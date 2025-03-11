import math

import torch
from einops import rearrange
from torch import nn
from vision_architectures.layers.embeddings import AbsolutePositionEmbeddings3D
from vision_architectures.nets.perceiver_3d import (
    Perceiver3DChannelMapping,
    Perceiver3DChannelMappingConfig,
    Perceiver3DDecoder,
    Perceiver3DEncoder,
)
from vision_architectures.nets.swinv2_3d import SwinV23DDecoder, SwinV23DModel


class SigmoidScheduler:
    def __init__(self, min_y=0.0, max_y=1.0, min_x=-7, max_x=7):
        assert min_x < max_x, "min_x must be less than max_x"
        assert min_y < max_y, "min_y must be less than max_y"

        self.min_y = min_y
        self.max_y = max_y
        self.min_x = min_x
        self.max_x = max_x
        self.num_steps = ...
        self.x_step_size = ...

        self.x = min_x

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def set_num_steps(self, num_steps):
        if self.num_steps == ...:
            self.num_steps = num_steps
            self.x_step_size = (self.max_x - self.min_x) / self.num_steps

    def is_ready(self):
        return self.num_steps != ...

    def is_completed(self):
        return self.x >= self.max_x

    def get(self):
        if not self.is_ready():
            raise ValueError("Call set_num_steps first")
        y = self._sigmoid(self.x)
        scaled_y = self._scale(y)
        return scaled_y

    def step(self):
        if not self.is_ready():
            raise ValueError("Call set_num_steps first")
        if self.is_completed():
            return
        self.x = self.x + self.x_step_size

    def _scale(self, y):
        scaled_y = self.min_y + y * (self.max_y - self.min_y)
        return scaled_y


class AdaptorResidualConnection(nn.Module):
    def __init__(self, pathway_drop_prob=0.0):
        super().__init__()

        self.pathway_drop_prob = pathway_drop_prob  # helps avoid co-adaptation
        self.weight_scheduler = SigmoidScheduler()

    def set_num_steps(self, num_steps):
        self.weight_scheduler.set_num_steps(num_steps)

    def forward(self, swin_encoder_output: torch.Tensor, perceiver_decoder_output):
        perceiver_weight = self.weight_scheduler.get()
        if self.training:
            self.weight_scheduler.step()

        device = swin_encoder_output.device
        b = swin_encoder_output.shape[0]
        perceiver_weight: torch.Tensor = torch.full((b, 1, 1, 1, 1), perceiver_weight, device=device)
        swin_weight = 1 - perceiver_weight

        # Implement dropout
        if self.training and self.pathway_drop_prob > 0:
            dropout_mask = torch.rand_like(perceiver_weight) < self.pathway_drop_prob
            if dropout_mask.any():
                swin_encoder_output[dropout_mask.squeeze()] = swin_encoder_output[dropout_mask.squeeze()].detach()

        output = swin_weight * swin_encoder_output + perceiver_weight * perceiver_decoder_output
        return output


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(3, out_channels),
            nn.GELU(),
        )

        # Frequency separation block
        # Low-frequency path with standard convolution
        self.low_freq_branch = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.GroupNorm(3, out_channels),
            nn.GELU(),
        )
        # High-frequency path with depthwise separable convolution for efficiency
        self.high_freq_branch = nn.Sequential(
            nn.Conv3d(out_channels, out_channels * 2, kernel_size=1, padding=0, stride=1),
            nn.GroupNorm(3, out_channels * 2),
            nn.GELU(),
            # Depthwise convolution for spatial feature extraction
            nn.Conv3d(
                out_channels * 2,
                out_channels * 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=out_channels,
            ),
            nn.GroupNorm(3, out_channels * 2),
            nn.GELU(),
            # Pointwise convolution for feature mixing
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=1, padding=0, stride=1),
        )
        # Adaptive frequency gating mechanism
        self.fsb_gate = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        x = self.upsample(x)

        low_freq = self.low_freq_branch(x)
        high_freq = self.high_freq_branch(x)
        gate = self.fsb_gate(x)
        x = low_freq + high_freq * gate

        return x


class UnembeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        if len(config.upsample_channels) > 0:
            upsample_fsb_channels = zip([config.in_channels] + config.upsample_channels[:-1], config.upsample_channels)
            self.upsample_fsbs = nn.ModuleList(
                [UpsampleLayer(in_channels, out_channels) for in_channels, out_channels in upsample_fsb_channels]
            )
            final_in_chanels = config.upsample_channels[-1]
        else:
            self.upsample_fsbs = []
            final_in_chanels = config.in_channels

        self.finalize = nn.Sequential(
            nn.Conv3d(final_in_chanels, config.out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        for upsample_fsb in self.upsample_fsbs:
            x = upsample_fsb(x)

        x = self.finalize(x)
        # (b, d, z, y, x)

        return x


class AdaptiveAE(nn.Module):
    def __init__(self, model_config: dict, checkpointing_level: int = 0):
        super().__init__()

        self.model_config = model_config

        latent_channels = model_config.adaptor.dim

        # input_channel_mapping_config = Perceiver3DChannelMappingConfig(
        #     in_channels={stage.dim for stage in model_config.swin.stages},
        #     out_channels=latent_channels,
        # )
        input_channel_mapping_config = Perceiver3DChannelMappingConfig(
            in_channels=model_config.swin.stages[-1].dim,
            out_channels=latent_channels,
        )
        input_channel_mapping = Perceiver3DChannelMapping(input_channel_mapping_config)
        perceiver_position_embeddings = AbsolutePositionEmbeddings3D()

        self.encoder = SwinV23DModel(model_config.swin, checkpointing_level)
        self.perceiver_position_embeddings = perceiver_position_embeddings
        self.adapt = Perceiver3DEncoder(model_config.adaptor, input_channel_mapping, checkpointing_level)
        self.unadapt = Perceiver3DDecoder(model_config.adaptor, perceiver_position_embeddings, checkpointing_level)
        self.residual_connection = AdaptorResidualConnection(model_config.pathway_drop_prob)
        self.decoder = SwinV23DDecoder(model_config.decoder, checkpointing_level)
        self.unembedding = UnembeddingLayer(model_config.unembedding)

    def encode(self, x: torch.Tensor, crop_offsets: torch.Tensor = None, return_stage_outputs=False):
        encoded, stage_outputs, _ = self.encoder(x, crop_offsets=crop_offsets)
        # return encoded, encoded, encoded

        sliding_window, sliding_stride = None, None
        # if not self.training:
        #     sliding_window = 2**14
        #     sliding_stride = sliding_window // 2

        scaled_crop_offsets = []
        cur_crop_offset = crop_offsets.clone()
        cur_crop_offset = cur_crop_offset // torch.tensor(
            self.model_config.swin.patch_size, device=cur_crop_offset.device
        )
        for stage_config in self.model_config.swin.stages:
            scaled_crop_offsets.append(
                cur_crop_offset // torch.tensor(stage_config.spatial_compression_ratio, device=cur_crop_offset.device)
            )
            cur_crop_offset = scaled_crop_offsets[-1].clone()

        # embedded_stage_outputs: list[torch.Tensor] = []
        # for stage_output, scaled_crop_offset in zip(stage_outputs, scaled_crop_offsets):
        #     embedded_stage_output = stage_output + self.perceiver_position_embeddings(
        #         batch_size=stage_output.shape[0],
        #         dim=stage_output.shape[1],
        #         grid_size=stage_output.shape[2:],
        #         device=stage_output.device,
        #         crop_offsets=scaled_crop_offset,
        #     )
        #     embedded_stage_outputs.append(embedded_stage_output)
        embedded_encoded = encoded + self.perceiver_position_embeddings(
            batch_size=encoded.shape[0],
            dim=encoded.shape[1],
            grid_size=encoded.shape[2:],
            device=encoded.device,
            crop_offsets=scaled_crop_offsets[-1],
        )

        adapted = self.adapt(
            # list(
            #     reversed(embedded_stage_outputs)
            # ),  # Show low frequency features first, then show higher frequency features
            embedded_encoded,
            sliding_window=sliding_window,
            sliding_stride=sliding_stride,
        )

        return adapted, encoded, scaled_crop_offsets

    def decode(self, z: torch.Tensor, swin_encoder_output, scaled_crop_offsets):
        # z = rearrange(z, "b d z y x -> b z y x d")
        # dec, _, _ = self.decoder(z)
        # dec = rearrange(dec, "b z y x d -> b d z y x")
        # return dec

        decoder_in_shape = swin_encoder_output.shape[2:]
        unadapted = self.unadapt(z, out_shape=decoder_in_shape, crop_offsets=scaled_crop_offsets[-1])

        if self.training and not self.residual_connection.weight_scheduler.is_completed():
            decoder_input = self.residual_connection(swin_encoder_output, unadapted)
        else:
            decoder_input = unadapted

        decoder_input = rearrange(decoder_input, "b d z y x -> b z y x d")
        decoded, _, _ = self.decoder(decoder_input)
        decoded = rearrange(decoded, "b z y x d -> b d z y x")

        return decoded

    def forward(self, x, crop_offsets, run_type="val"):
        # x: (b, d1, z1, y1, x1)

        adapted, encoded, scaled_crop_offsets = self.encode(x, crop_offsets)

        decoded = self.decode(adapted, encoded, scaled_crop_offsets)

        reconstructed = self.unembedding(decoded)

        return {
            "reconstructed": reconstructed,
            "encoded": encoded,
        }


if __name__ == "__main__":
    from time import perf_counter

    import psutil
    from config import get_config
    from neuro_utils.describe import describe_model

    config = get_config()

    device = torch.device("cpu")
    # device = torch.device("cuda:0")

    autoencoder = AdaptiveAE(config.model, 2).to(device)
    autoencoder.residual_connection.set_num_steps(100)
    print("Encoder:")
    describe_model(autoencoder.encoder)
    print("Adapt:")
    describe_model(autoencoder.adapt)
    print("Unadapt:")
    describe_model(autoencoder.unadapt)
    print("Decoder:")
    describe_model(autoencoder.decoder)
    print("Final layer:")
    describe_model(autoencoder.unembedding)

    autoencoder.train()

    # Track memory before execution
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    initial_mem = process.memory_info().rss  # in bytes

    sample_input = torch.zeros((1, 1, *config.image_size)).to(device)
    crop_offsets = torch.Tensor([[100, 100, 100]]).to(device)

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
