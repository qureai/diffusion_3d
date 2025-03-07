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
from vision_architectures.nets.swinv2_3d import SwinV23DConfig, SwinV23DDecoder, SwinV23DModel


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

        upsample_fsb_channels = zip([config.in_channels] + config.upsample_channels[:-1], config.upsample_channels)
        self.upsample_fsbs = nn.ModuleList(
            [UpsampleLayer(in_channels, out_channels) for in_channels, out_channels in upsample_fsb_channels]
        )

        self.finalize = nn.Sequential(
            nn.Conv3d(config.upsample_channels[-1], config.out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        for upsample_fsb in self.upsample_fsbs:
            x = upsample_fsb(x)

        x = self.finalize(x)
        # (b, d, z, y, x)

        return x


class AdaptiveVAE(nn.Module):
    def __init__(self, model_config: dict, checkpointing_level: int = 0):
        super().__init__()

        self.model_config = model_config

        latent_channels = model_config.adaptor.dim

        input_channel_mapping_config = Perceiver3DChannelMappingConfig(
            in_channels={stage.dim for stage in model_config.swin.stages},
            out_channels=latent_channels,
        )
        input_channel_mapping = Perceiver3DChannelMapping(input_channel_mapping_config)
        decode_position_embeddings = AbsolutePositionEmbeddings3D(dim=latent_channels)

        self.encoder = SwinV23DModel(model_config.swin, checkpointing_level)
        self.adapt = Perceiver3DEncoder(model_config.adaptor, input_channel_mapping, checkpointing_level)
        self.quant_conv_mu = nn.Linear(latent_channels, latent_channels)
        self.quant_conv_log_sigma = nn.Linear(latent_channels, latent_channels)
        self.post_quant_conv = nn.Linear(latent_channels, latent_channels)
        self.unadapt = Perceiver3DDecoder(model_config.adaptor, decode_position_embeddings, checkpointing_level)

        self.decoder = SwinV23DDecoder(model_config.decoder, checkpointing_level)
        self.unembedding = UnembeddingLayer(model_config.unembedding)

    def encode(self, x: torch.Tensor, return_stage_outputs=False):
        encoded, stage_outputs, _ = self.encoder(x)

        encoder_out_shape = encoded.shape[2:]

        sliding_window, sliding_stride = None, None
        if not self.training:
            sliding_window = 2**16
            sliding_stride = sliding_window // 2

        adapted = self.adapt(
            list(reversed(stage_outputs)),  # Show low frequency features first, then show higher frequency features
            sliding_window=sliding_window,
            sliding_stride=sliding_stride,
        )

        z_mu = self.quant_conv_mu(adapted)
        z_log_var = self.quant_conv_log_sigma(adapted)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        if return_stage_outputs:
            return z_mu, z_sigma, encoder_out_shape, stage_outputs
        return z_mu, z_sigma, encoder_out_shape

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def decode(self, z: torch.Tensor, decoder_in_shape: tuple[int, int, int]):
        z = self.post_quant_conv(z)

        unadapted = self.unadapt(z, out_shape=decoder_in_shape)

        unadapted = rearrange(unadapted, "b d z y x -> b z y x d")
        dec, _, _ = self.decoder(unadapted)
        dec = rearrange(dec, "b z y x d -> b d z y x")

        return dec

    def forward(self, x, run_type="val"):
        # x: (b, d1, z1, y1, x1)

        encoded_mu, encoded_sigma, encoder_out_shape = self.encode(x)
        if run_type == "train":
            encoded = self.sampling(encoded_mu, encoded_sigma)
        else:
            encoded = encoded_mu
        # (b, d2, z2, y2, x2)

        decoded = self.decode(encoded, decoder_in_shape=encoder_out_shape)
        # (b, d3, z3, y3, x3)

        reconstructed = self.unembedding(decoded)
        # (b, d1, z1, y1, x1)

        return {
            "reconstructed": reconstructed,
            "encoded_mu": encoded_mu,
            "encoded_sigma": encoded_sigma,
        }


if __name__ == "__main__":
    from time import perf_counter

    import psutil
    from config import get_config
    from neuro_utils.describe import describe_model

    config = get_config()

    device = torch.device("cpu")
    device = torch.device("cuda:0")

    autoencoder = AdaptiveVAE(config.model, 2).to(device)
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

    tic = perf_counter()
    sample_output = autoencoder(sample_input, "train")
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
