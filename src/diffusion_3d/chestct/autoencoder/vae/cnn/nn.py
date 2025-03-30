import torch
from munch import Munch
from torch import nn
from vision_architectures.blocks.cnn import CNNBlock3D, MultiResCNNBlock3D
from vision_architectures.layers.latent_space import GaussianLatentSpace, LatentDecoder, LatentEncoder
from vision_architectures.nets.swinv2_3d import SwinV23DPatchMerging, SwinV23DPatchSplitting
from vision_architectures.utils.activation_checkpointing import ActivationCheckpointing
from vision_architectures.utils.rearrange import rearrange_channels
from vision_architectures.utils.residuals import add_stochastic_depth_dropout


class StageBlock(nn.Module):
    def __init__(self, config, depth, checkpointing_level: int = 0):
        super().__init__()

        assert config.in_channels == config.out_channels, config

        self.layers = nn.Sequential(*[MultiResCNNBlock3D(config, checkpointing_level) for _ in range(depth)])

        self.checkpointing_level3 = ActivationCheckpointing(3, checkpointing_level)

    def _forward(self, x: torch.Tensor, channels_first: bool = True):
        # x: (b, [dim], z, y, x, [dim])

        x = rearrange_channels(x, channels_first, True)
        # (b, dim, z, y, x)

        x = self.layers(x)
        # (b, dim, z, y, x)

        x = rearrange_channels(x, True, channels_first)
        # (b, [dim], z, y, x, [dim])

        return x

    def forward(self, *args, **kwargs):
        return self.checkpointing_level3(self._forward, *args, **kwargs)


class Encoder(nn.Module):
    def __init__(self, config: Munch, checkpointing_level: int = 0):
        super().__init__()

        self.stages = nn.ModuleList([])
        for i in range(len(config.depths)):
            depth = config.depths[i]
            channels = config.num_channels[i]

            stage = nn.Sequential()

            if i > 0:
                stage.append(
                    SwinV23DPatchMerging(
                        (2, 2, 2), config.num_channels[i - 1], channels, checkpointing_level=checkpointing_level
                    )
                )

            block_config = config.copy()
            block_config["in_channels"] = channels
            block_config["out_channels"] = channels
            block_config["filter_ratios"] = [3, 2, 1]

            stage.append(StageBlock(block_config, depth, checkpointing_level))
            self.stages.append(stage)

        self.latent_encoder = LatentEncoder(config.latent, checkpointing_level)

    def forward(self, x: torch.Tensor):
        stage_outputs = []
        for stage in self.stages:
            x = stage(x)
            stage_outputs.append(x)
        x = self.latent_encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config, checkpointing_level: int = 0):
        super().__init__()

        self.latent_decoder = LatentDecoder(config.latent, checkpointing_level)

        config = config.copy()
        config.depths = config.depths[::-1]
        config.num_channels = config.num_channels[::-1]

        self.stages = nn.ModuleList([])
        for i in range(len(config.depths)):
            depth = config.depths[i]
            channels = config.num_channels[i]

            stage = nn.Sequential()

            if i > 0:
                stage.append(
                    SwinV23DPatchSplitting(
                        (2, 2, 2), config.num_channels[i - 1], channels, checkpointing_level=checkpointing_level
                    )
                )

            block_config = config.copy()
            block_config["in_channels"] = channels
            block_config["out_channels"] = channels
            block_config["filter_ratios"] = [3, 2, 1]

            stage.append(StageBlock(block_config, depth, checkpointing_level))
            self.stages.append(stage)

    def forward(self, x: torch.Tensor):
        x = self.latent_decoder(x)
        stage_outputs = []
        for stage in self.stages:
            x = stage(x)
            stage_outputs.append(x)
        return x


class VAE(nn.Module):
    def __init__(self, model_config: dict, checkpointing_level: int = 0):
        super().__init__()

        self.model_config = model_config

        self.encoder_mapping = CNNBlock3D(
            model_config,
            checkpointing_level,
            in_channels=model_config.in_channels,
            out_channels=model_config.num_channels[0],
            kernel_size=3,
        )
        self.encoder = Encoder(model_config, checkpointing_level)
        self.latent_space = GaussianLatentSpace()
        self.decoder = Decoder(model_config, checkpointing_level)
        self.decoder_mapping = CNNBlock3D(
            model_config,
            checkpointing_level,
            in_channels=model_config.num_channels[0],
            out_channels=model_config.in_channels,
            kernel_size=3,
        )

        if model_config.survival_prob > 0.0:
            self.encoder = add_stochastic_depth_dropout(self.encoder, model_config.survival_prob)
            self.decoder = add_stochastic_depth_dropout(self.decoder, model_config.survival_prob)

    def forward(self, x):
        x = self.encoder_mapping(x)
        z_mu, z_sigma = self.encoder(x)
        z, kl_loss = self.latent_space(z_mu, z_sigma)
        decoded = self.decoder(z)
        reconstructed = self.decoder_mapping(decoded)

        return {
            "reconstructed": reconstructed,
            "z_mu": z_mu,
            "z_sigma": z_sigma,
            "kl_loss": kl_loss,
        }


if __name__ == "__main__":
    from time import perf_counter

    import psutil
    from arjcode.visualize import describe_model
    from config import get_config
    from monai.networks.nets.autoencoderkl import AutoencoderKL

    config = get_config()

    device = torch.device("cpu")
    device = torch.device("cuda:0")

    # autoencoder = VAE(config.model, 0).to(device)
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=config.model.in_channels,
        out_channels=config.model.in_channels,
        num_res_blocks=config.model.depths,
        channels=config.model.num_channels,
        attention_levels=[False] * len(config.model.depths),
        norm_num_groups=config.model.num_channels[0],
        latent_channels=config.model.latent.latent_dim,
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    ).to(device)
    print("Encoder:")
    describe_model(autoencoder.encoder)
    print("Decoder:")
    describe_model(autoencoder.decoder)

    autoencoder.train()

    # Track memory before execution
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    initial_mem = process.memory_info().rss  # in bytes

    sample_input = torch.zeros((1, 1, *config.image_size)).to(device)

    tic = perf_counter()
    sample_output = autoencoder(sample_input)
    toc = perf_counter()
    forward_time = toc - tic

    loss: torch.Tensor = sample_output[0].sum()
    tic = perf_counter()
    loss.backward()
    toc = perf_counter()
    backward_time = toc - tic

    final_mem = process.memory_info().rss  # in bytes

    print()
    print("Input shape:", sample_input.shape)
    print()
    print("Outputs:")
    print("Reconstructed:", sample_output[0].shape)
    print("z_mu:", sample_output[1].shape)
    print("z_sigma:", sample_output[2].shape)
    print()
    print(f"GPU: {torch.cuda.max_memory_allocated() / 2**30} GB peak mem used")
    print(f"RAM: {(final_mem - initial_mem) / 2**30} GB peak mem used")
    print(f"Time (forward): {forward_time:.3f} s")
    print(f"Time (backward): {backward_time:.3f} s")

    # -----
    # loss: torch.Tensor = sample_output["reconstructed"].sum()
    # tic = perf_counter()
    # loss.backward()
    # toc = perf_counter()
    # backward_time = toc - tic

    # final_mem = process.memory_info().rss  # in bytes

    # print()
    # print("Input shape:", sample_input.shape)
    # print()
    # print("Outputs:")
    # print("Reconstructed:", sample_output["reconstructed"].shape)
    # print("z_mu:", sample_output["z_mu"].shape)
    # print("z_sigma:", sample_output["z_sigma"].shape)
    # print("kl_loss:", sample_output["kl_loss"])
    # print()
    # print(f"GPU: {torch.cuda.max_memory_allocated() / 2**30} GB peak mem used")
    # print(f"RAM: {(final_mem - initial_mem) / 2**30} GB peak mem used")
    # print(f"Time (forward): {forward_time:.3f} s")
    # print(f"Time (backward): {backward_time:.3f} s")
