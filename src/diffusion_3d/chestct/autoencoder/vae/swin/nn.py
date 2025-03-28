import torch
from einops import rearrange
from torch import nn
from vision_architectures.nets.swinv2_3d import SwinV23DDecoder, SwinV23DModel
from vision_architectures.utils.normalizations import LayerNorm3D


class UpsampleLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(3, out_dim),
            nn.GELU(),
        )

        # Frequency separation block
        # Low-frequency path with standard convolution
        self.low_freq_branch = nn.Sequential(
            nn.Conv3d(out_dim, out_dim, kernel_size=5, padding=2, stride=1),
            nn.GroupNorm(3, out_dim),
            nn.GELU(),
        )
        # High-frequency path with depthwise separable convolution for efficiency
        self.high_freq_branch = nn.Sequential(
            nn.Conv3d(out_dim, out_dim * 2, kernel_size=1, padding=0, stride=1),
            nn.GroupNorm(3, out_dim * 2),
            nn.GELU(),
            # Depthwise convolution for spatial feature extraction
            nn.Conv3d(
                out_dim * 2,
                out_dim * 2,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=out_dim,
            ),
            nn.GroupNorm(3, out_dim * 2),
            nn.GELU(),
            # Pointwise convolution for feature mixing
            nn.Conv3d(out_dim * 2, out_dim, kernel_size=1, padding=0, stride=1),
        )
        # Adaptive frequency gating mechanism
        self.fsb_gate = nn.Sequential(nn.Conv3d(out_dim, out_dim, kernel_size=1), nn.Sigmoid())

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

        if len(config.upsample_dims) > 0:
            upsample_fsb_dims = zip([config.in_dim] + config.upsample_dims[:-1], config.upsample_dims)
            self.upsample_fsbs = nn.ModuleList(
                [UpsampleLayer(in_dim, out_dim) for in_dim, out_dim in upsample_fsb_dims]
            )
            final_in_chanels = config.upsample_dims[-1]
        else:
            self.upsample_fsbs = []
            final_in_chanels = config.in_dim

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


class VAE(nn.Module):
    def __init__(self, model_config: dict, checkpointing_level: int = 0):
        super().__init__()

        self.model_config = model_config

        latent_in_dim = model_config.encoder.stages[-1].dim
        latent_dim = model_config.latent_dim
        latent_out_dim = model_config.decoder.dim

        self.encoder = SwinV23DModel(model_config.encoder, checkpointing_level)
        self.encoder_mapping = nn.Sequential(
            nn.Conv3d(latent_in_dim, latent_dim, kernel_size=3, padding=1),
            LayerNorm3D(latent_dim),
            nn.GELU(),
        )
        self.quant_conv_mu = nn.Conv3d(latent_dim, latent_dim, 1)
        self.quant_conv_log_var = nn.Conv3d(latent_dim, latent_dim, 1)
        self.post_quant_conv = nn.Conv3d(latent_dim, latent_dim, 1)
        self.decoder_mapping = nn.Sequential(
            nn.Conv3d(latent_dim, latent_out_dim, kernel_size=3, padding=1),
            LayerNorm3D(latent_out_dim),
            nn.GELU(),
        )
        self.decoder = SwinV23DDecoder(model_config.decoder, checkpointing_level)
        self.unembedding = UnembeddingLayer(model_config.unembedding)

        self.init()

    def init(self):
        # nn.init.ones_(self.quant_conv_mu.weight)
        nn.init.zeros_(self.quant_conv_mu.bias)
        nn.init.zeros_(self.quant_conv_log_var.weight)
        nn.init.constant_(self.quant_conv_log_var.bias, 0.0)

    def encode(self, x: torch.Tensor, crop_offsets: torch.Tensor = None):
        encoded = self.encoder(x, crop_offsets=crop_offsets)

        scaled_crop_offsets = []
        cur_crop_offset = crop_offsets.clone()
        for stage_config in self.model_config.encoder.stages:
            scaled_crop_offsets.append(
                cur_crop_offset // torch.tensor(stage_config.spatial_compression_ratio, device=cur_crop_offset.device)
            )
            cur_crop_offset = scaled_crop_offsets[-1].clone()

        encoded = self.encoder_mapping(encoded)

        z_mu = self.quant_conv_mu(encoded)
        z_log_var = self.quant_conv_log_var(encoded)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma, encoded, scaled_crop_offsets

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def decode(self, z: torch.Tensor, encoder_output, scaled_crop_offsets):
        sampled = self.post_quant_conv(z)
        sampled = self.decoder_mapping(sampled)
        decoded = self.decoder(sampled)
        return decoded

    def forward(self, x, crop_offsets, run_type="val"):
        # x: (b, d1, z1, y1, x1)

        encoded_mu, encoded_sigma, encoder_output, scaled_crop_offsets = self.encode(x, crop_offsets)
        # encoded = encoded_mu
        if run_type == "train":
            encoded = self.sampling(encoded_mu, encoded_sigma)
        else:
            encoded = encoded_mu
        # (b, d2, z2, y2, x2)

        decoded = self.decode(encoded, encoder_output, scaled_crop_offsets)
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
    from arjcode.visualize import describe_model
    from config import get_config

    config = get_config()

    device = torch.device("cpu")
    device = torch.device("cuda:0")

    autoencoder = VAE(config.model, 1).to(device)
    state_dict = torch.load(
        r"/raid3/arjun/checkpoints/adaptive_autoencoder/v47__2025_03_25/version_0/checkpoints/last.ckpt",
        map_location=device,
    )["state_dict"]
    for key in state_dict.copy().keys():
        value = state_dict.pop(key)
        if key.startswith("autoencoder.") and not "quant" in key:
            state_dict[key.removeprefix("autoencoder.")] = value
    autoencoder.load_state_dict(state_dict, strict=False)
    autoencoder.init()
    print("Encoder:")
    describe_model(autoencoder.encoder)
    print("Encoder mapper:")
    describe_model(autoencoder.encoder_mapping)
    print("Decoder mapper:")
    describe_model(autoencoder.decoder_mapping)
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
