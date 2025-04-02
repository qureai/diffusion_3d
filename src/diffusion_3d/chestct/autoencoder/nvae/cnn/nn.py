import numpy as np
import torch
from einops import repeat
from munch import Munch, munchify
from torch import nn
from vision_architectures.blocks.cnn import CNNBlock3D
from vision_architectures.layers.latent_space import GaussianLatentSpace, LatentDecoder, LatentEncoder
from vision_architectures.utils.activation_checkpointing import ActivationCheckpointing
from vision_architectures.utils.rearrange import rearrange_channels
from vision_architectures.utils.residuals import Residual, add_stochastic_depth_dropout


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, model_config, checkpointing_level: int = 0):
        super().__init__()

        self.conv1 = CNNBlock3D(
            model_config,
            checkpointing_level,
            in_channels=in_channels,
            out_channels=out_channels,
            sequence="NACD",
        )
        self.conv2 = CNNBlock3D(
            model_config,
            checkpointing_level,
            in_channels=out_channels,
            out_channels=out_channels,
            sequence="NACD",
        )
        self.conv_res = CNNBlock3D(
            model_config,
            checkpointing_level,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            activation=None,
            normalization=None,
        )
        self.residual = Residual()

    def forward(self, x):
        res = self.conv_res(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual(x, res)
        return x


class StageBlock(nn.Module):
    def __init__(self, num_channels, config, depth, has_skip_conn: bool, checkpointing_level: int = 0):
        super().__init__()

        in_channels = [num_channels] * depth
        out_channels = [num_channels] * depth
        if has_skip_conn:
            in_channels[0] = num_channels * 2

        self.layers = nn.Sequential(
            *[
                ResBlock(in_channel, out_channel, config, checkpointing_level)
                for in_channel, out_channel in zip(in_channels, out_channels)
            ]
        )

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
                    CNNBlock3D(  # Downsample
                        config,
                        checkpointing_level,
                        in_channels=config.num_channels[i - 1],
                        out_channels=channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        activation=None,
                        normalization=None,
                    )
                )

            stage.append(StageBlock(channels, config, depth, checkpointing_level, False))
            self.stages.append(stage)

    def forward(self, x: torch.Tensor):
        encodings = []
        for stage in self.stages:
            x = stage(x)
            encodings.append(x)
        return encodings


class LatentSpace(nn.Module):
    def __init__(self, config: Munch, checkpointing_level: int = 0):
        super().__init__()

        self.config = config

        num_channels = config.num_channels
        latent_dims = config.latent_dims
        stages_with_latents = np.argwhere(np.array(latent_dims) != None).flatten()
        self.stages_with_latents = stages_with_latents

        # deepest "latent" on which the deepest prior is estimated, I am calling this hidden latent
        self.hidden_latent = nn.Parameter(torch.randn(latent_dims[-1]), requires_grad=True)

        self.prior_estimators = nn.ModuleList([None] * len(latent_dims))  # Predicts prior distributions
        for target_i in stages_with_latents:
            target_dim = latent_dims[target_i]

            # source_i and target_i are used to upsample the latent to the correct spatial resolution
            # source_dim and target_dim are used to create the prior estimate

            # prior estimator requires next latent
            if target_i == len(latent_dims) - 1:  # if deepest latent
                source_i = target_i
                source_dim = target_dim
            else:
                source_i = stages_with_latents[stages_with_latents > target_i].min()
                source_dim = latent_dims[source_i]

            upscale_factor = 2 ** (source_i - target_i)

            prior_estimator_config = config.latent.copy()
            prior_estimator_config.dim = source_dim
            prior_estimator_config.latent_dim = target_dim
            self.prior_estimators[target_i] = nn.ModuleDict(
                {
                    "upsample": nn.Upsample(scale_factor=upscale_factor, mode="trilinear"),
                    "estimate": LatentEncoder(prior_estimator_config, checkpointing_level),
                }
            )

        self.latent_estimators = nn.ModuleList([None] * len(latent_dims))  # Predicts posterior distributions
        for i in stages_with_latents:
            latent_dim = latent_dims[i]
            dim = num_channels[i]

            latent_encoder_config = config.latent.copy()
            latent_encoder_config.dim = dim
            latent_encoder_config.latent_dim = latent_dim
            self.latent_estimators[i] = LatentEncoder(latent_encoder_config, checkpointing_level)

        self.latent_encoder = GaussianLatentSpace()

    def forward(self, encodings: torch.Tensor):
        b, _, z, y, x = encodings[-1].shape
        last_latent = repeat(self.hidden_latent, "d -> b d z y x", b=b, z=z, y=y, x=x)

        prior_distributions = [None] * len(self.config.latent_dims)
        posterior_distributions = [None] * len(self.config.latent_dims)
        latents = [None] * len(self.config.latent_dims)
        kl_divergences = [None] * len(self.config.latent_dims)
        for i in self.stages_with_latents[::-1]:
            x = encodings[i]

            upsampled_last_latent = self.prior_estimators[i]["upsample"](last_latent)
            prior_mu, prior_sigma, prior_logvar = self.prior_estimators[i]["estimate"](
                upsampled_last_latent, return_logvar=True
            )
            prior_distributions[i] = (prior_mu, prior_sigma)

            posterior_mu, posterior_sigma = self.latent_estimators[i](x, prior_mu, prior_logvar)
            posterior_distributions[i] = (posterior_mu, posterior_sigma)

            latent, kl_divergence = self.latent_encoder(posterior_mu, posterior_sigma, prior_mu, prior_sigma)
            latents[i] = latent
            kl_divergences[i] = kl_divergence

            last_latent = latent

        return latents, kl_divergences, prior_distributions, posterior_distributions


class Decoder(nn.Module):
    def __init__(self, config, checkpointing_level: int = 0):
        super().__init__()

        config = config.copy()
        depths = config.depths[::-1]
        num_channels = config.num_channels[::-1]
        latent_dims = config.latent_dims[::-1]
        stages_with_latents = np.argwhere(np.array(latent_dims) != None).flatten()

        self.latent_decoders = nn.ModuleList([None] * len(latent_dims))
        for i in stages_with_latents:
            latent_dim = latent_dims[i]
            dim = num_channels[i]

            latent_decoder_config = config.latent.copy()
            latent_decoder_config.latent_dim = latent_dim
            latent_decoder_config.dim = dim
            self.latent_decoders[i] = LatentDecoder(latent_decoder_config, checkpointing_level)

        self.stages = nn.ModuleList([])
        for i in range(len(depths)):
            dim = num_channels[i]
            depth = depths[i]
            channels = num_channels[i]
            has_skip_conn = True
            if latent_dims[i] is None or i == 0:  # if not concatenating with sampled latent or if deepest
                has_skip_conn = False

            stage = nn.Sequential()

            stage.append(StageBlock(dim, config, depth, has_skip_conn, checkpointing_level))

            if i < len(depths) - 1:
                stage.append(
                    CNNBlock3D(  # Upsample
                        config,
                        checkpointing_level,
                        in_channels=channels,
                        out_channels=num_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        conv_kwargs={"output_padding": 1},
                        activation=None,
                        normalization=None,
                        transposed=True,
                    )
                )

            self.stages.append(stage)

    def forward(self, latents: torch.Tensor):
        latents = latents[::-1]

        x = None
        for i in range(len(latents)):
            if latents[i] is not None:
                latent = latents[i]
                x_latent = self.latent_decoders[i](latent)
                if x is None:
                    x = x_latent
                else:
                    x = torch.cat((x, x_latent), dim=1)
            x = self.stages[i](x)

        return x


class VAE(nn.Module):
    def __init__(self, model_config: dict, checkpointing_level: int = 0):
        super().__init__()

        self.model_config = model_config

        assert self.model_config.latent_dims[-1] is not None
        assert (
            len(self.model_config.latent_dims) == len(self.model_config.depths) == len(self.model_config.num_channels)
        )

        self.encoder_mapping = CNNBlock3D(
            checkpointing_level=checkpointing_level,
            in_channels=model_config.in_channels,
            out_channels=model_config.num_channels[0],
            kernel_size=3,
            normalization=None,
            activation=None,
        )
        self.encoder = Encoder(model_config, checkpointing_level)
        self.latent_space = LatentSpace(model_config, checkpointing_level)
        self.decoder = Decoder(model_config, checkpointing_level)
        self.decoder_mapping = CNNBlock3D(
            model_config,
            checkpointing_level,
            in_channels=model_config.num_channels[0],
            out_channels=model_config.in_channels,
            kernel_size=3,
            normalization=None,
            activation="tanh",
        )

        if model_config.survival_prob < 1.0:
            self.encoder = add_stochastic_depth_dropout(self.encoder, model_config.survival_prob)
            self.decoder = add_stochastic_depth_dropout(self.decoder, model_config.survival_prob)

    def forward(self, x):
        x = self.encoder_mapping(x)
        encodings = self.encoder(x)
        latents, kl_divergences, prior_distributions, posterior_distributions = self.latent_space(encodings)
        decoded = self.decoder(latents)
        reconstructed = self.decoder_mapping(decoded)

        return {
            "reconstructed": reconstructed,
            "latents": latents,
            "kl_divergences": kl_divergences,
            "prior_distributions": prior_distributions,
            "posterior_distributions": posterior_distributions,
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

    autoencoder = VAE(config.model, 0).to(device)
    print("Encoder:")
    describe_model(autoencoder.encoder)
    print("Latent Space:")
    describe_model(autoencoder.latent_space)
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

    loss: torch.Tensor = sample_output["reconstructed"].sum()
    tic = perf_counter()
    loss.backward()
    toc = perf_counter()
    backward_time = toc - tic

    final_mem = process.memory_info().rss  # in bytes

    def replace_with_shapes(arr):
        for i in range(len(arr)):
            if arr[i] is not None:
                if isinstance(arr[i], tuple):
                    arr[i] = arr[i][0]
                arr[i] = arr[i].shape
        return arr

    print()
    print("Input shape:", sample_input.shape)
    print()
    print("Outputs:")
    print("Reconstructed:", sample_output["reconstructed"].shape)
    print("Latents:", replace_with_shapes(sample_output["latents"]))
    print("kl_loss:", ["kl_divergences"])
    print("Prior distributions:", replace_with_shapes(sample_output["prior_distributions"]))
    print("Posterior distributions:", replace_with_shapes(sample_output["posterior_distributions"]))
    print()
    print(f"GPU: {torch.cuda.max_memory_allocated() / 2**30} GB peak mem used")
    print(f"RAM: {(final_mem - initial_mem) / 2**30} GB peak mem used")
    print(f"Time (forward): {forward_time:.3f} s")
    print(f"Time (backward): {backward_time:.3f} s")
