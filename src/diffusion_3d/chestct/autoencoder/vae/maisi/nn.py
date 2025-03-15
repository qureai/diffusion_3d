import math
import os

import torch
from einops import rearrange
from monai.apps import download_url
from torch import nn
from vision_architectures.layers.embeddings import AbsolutePositionEmbeddings3D
from vision_architectures.nets.perceiver_3d import (
    Perceiver3DChannelMapping,
    Perceiver3DChannelMappingConfig,
    Perceiver3DDecoder,
    Perceiver3DEncoder,
)

from diffusion_3d.chestct.autoencoder.vae.maisi.maisi import AutoencoderKlMaisi


class AdaptiveVAE(nn.Module):
    def __init__(self, model_config: dict, checkpointing_level: int = 0):
        super().__init__()

        self.model_config = model_config

        # input_channel_mapping_config = Perceiver3DChannelMappingConfig(
        #     in_channels={stage.dim for stage in model_config.swin.stages},
        #     out_channels=model_config.adaptor.dim,
        # )
        input_channel_mapping_config = Perceiver3DChannelMappingConfig(
            in_channels=model_config.maisi.latent_channels,
            out_channels=model_config.adaptor.dim,
        )
        input_channel_mapping = Perceiver3DChannelMapping(input_channel_mapping_config)

        self.maisi = AutoencoderKlMaisi(
            spatial_dims=model_config.maisi.spatial_dims,
            in_channels=model_config.maisi.in_channels,
            latent_channels=model_config.maisi.latent_channels,
            out_channels=model_config.maisi.out_channels,
            num_res_blocks=model_config.maisi.num_res_blocks,
            num_channels=model_config.maisi.num_channels,
            norm_num_groups=model_config.maisi.norm_num_groups,
            norm_eps=model_config.maisi.norm_eps,
            attention_levels=model_config.maisi.attention_levels,
            with_encoder_nonlocal_attn=model_config.maisi.with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=model_config.maisi.with_decoder_nonlocal_attn,
            use_checkpointing=model_config.maisi.use_checkpointing,
            use_convtranspose=model_config.maisi.use_convtranspose,
            norm_float16=model_config.maisi.norm_float16,
            num_splits=model_config.maisi.num_splits,
            dim_split=model_config.maisi.dim_split,
        )
        self.perceiver_position_embeddings = AbsolutePositionEmbeddings3D()
        self.adapt = Perceiver3DEncoder(model_config.adaptor, input_channel_mapping, checkpointing_level)
        self.unadapt = Perceiver3DDecoder(model_config.adaptor, checkpointing_level)

        self.load_maisi_checkpoint()
        self.freeze_maisi()

    def load_maisi_checkpoint(self):
        trained_autoencoder_path = r"/raid3/arjun/checkpoints/maisi/autoencoder_epoch273.pt"
        trained_autoencoder_path_url = "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_maisi_autoencoder_epoch273_alternative.pt"
        if not os.path.exists(trained_autoencoder_path):
            download_url(url=trained_autoencoder_path_url, filepath=trained_autoencoder_path)
        state_dict = torch.load(trained_autoencoder_path, map_location="cpu", weights_only=False)
        self.maisi.load_state_dict(state_dict)

    def freeze_maisi(self):
        self.maisi.eval()
        for param in self.maisi.parameters():
            param.requires_grad = False

    def encode(self, x: torch.Tensor, crop_offsets: torch.Tensor = None):
        encoded, _ = self.maisi.encode(x)  # discard sigma and assume mu is the output

        scaled_crop_offsets = []
        cur_crop_offset = crop_offsets.clone()
        for _ in range(len(self.model_config.maisi.num_res_blocks) - 1):
            cur_crop_offset = cur_crop_offset // 2
            scaled_crop_offsets.append(cur_crop_offset)

        # embedded_encoded = encoded + self.perceiver_position_embeddings(
        #     batch_size=encoded.shape[0],
        #     dim=encoded.shape[1],
        #     grid_size=encoded.shape[2:],
        #     device=encoded.device,
        #     crop_offsets=scaled_crop_offsets[-1],
        # )
        # adapted = self.adapt(embedded_encoded)
        adapted = self.adapt(encoded)

        return adapted, encoded, scaled_crop_offsets

    def decode(self, z: torch.Tensor, maisi_encoder_output, scaled_crop_offsets):
        decoder_in_shape = maisi_encoder_output.shape[2:]
        decoder_input = self.unadapt(z, out_shape=decoder_in_shape, crop_offsets=scaled_crop_offsets[-1])
        decoded = self.maisi.decode(decoder_input)
        return decoded

    def forward(self, x, crop_offsets, run_type="val"):
        adapted, encoded, scaled_crop_offsets = self.encode(x, crop_offsets)
        decoded = self.decode(adapted, encoded, scaled_crop_offsets)
        reconstructed = decoded
        return {"reconstructed": reconstructed}


if __name__ == "__main__":
    from time import perf_counter

    import psutil
    from config import get_config
    from neuro_utils.describe import describe_model

    config = get_config()

    device = torch.device("cpu")
    device = torch.device("cuda:0")

    autoencoder = AdaptiveVAE(config.model, 2).to(device)
    print("MAISI:")
    describe_model(autoencoder.maisi)
    print("Adapt:")
    describe_model(autoencoder.adapt)
    print("Unadapt:")
    describe_model(autoencoder.unadapt)

    # Track memory before execution
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    initial_mem = process.memory_info().rss  # in bytes

    sample_input = torch.zeros((1, 1, *config.image_size)).to(device)
    crop_offsets = torch.Tensor([[0, 0, 0]]).to(device)

    tic = perf_counter()
    sample_output = autoencoder(sample_input, crop_offsets, "train")
    toc = perf_counter()
    forward_time = toc - tic

    loss: torch.Tensor = sample_output["reconstructed"].sum()
    tic = perf_counter()
    loss.backward()
    toc = perf_counter()
    backward_time = toc - tic

    final_mem = process.memory_info().rss  # in bytes

    print()
    print("Input shape:", sample_input.shape)
    print()
    print("Output shapes:", *[f"{key}: {value.shape}" for key, value in sample_output.items()], sep="\n")
    print()
    print(f"GPU: {torch.cuda.max_memory_allocated() / 2**30} GB peak mem used")
    print(f"RAM: {(final_mem - initial_mem) / 2**30} GB peak mem used")
    print(f"Time (forward): {forward_time:.3f} s")
    print(f"Time (backward): {backward_time:.3f} s")

    # losses = autoencoder.training_step({"scan": sample_input, "spacing": ...}, 0)
    # from pprint import pprint
    # pprint(losses)
    # pprint(losses)
