import numpy as np
import torch
from einops import repeat
from munch import Munch
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
            sequence="NAC",
        )
        self.conv2 = CNNBlock3D(
            model_config,
            checkpointing_level,
            in_channels=out_channels,
            out_channels=out_channels,
            sequence="NAC",
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

            stage.append(StageBlock(channels, config, depth, False, checkpointing_level))
            self.stages.append(stage)

    def forward(self, x: torch.Tensor):
        encodings = []
        for stage in self.stages:
            x = stage(x)
            encodings.append(x)
        return encodings


class LatentSpaceOps(nn.Module):
    def __init__(self, config: Munch, checkpointing_level: int = 0):
        super().__init__()

        self.config = config

        num_channels = config.num_channels
        latent_dims = config.latent_dims
        stages_with_latents = np.argwhere(np.array(latent_dims) != None).flatten()
        self.stages_with_latents = stages_with_latents

        self.prior_estimators = nn.ModuleList([None] * len(latent_dims))  # Predicts prior distributions
        self.latent_encoders = nn.ModuleList([None] * len(latent_dims))  # Predicts posterior distributions
        self.latent_decoders = nn.ModuleList([None] * len(latent_dims))
        for i in stages_with_latents:
            latent_dim = latent_dims[i]
            dim = num_channels[i]

            latent_config = config.latent.copy()
            latent_config.dim = dim
            latent_config.latent_dim = latent_dim
            self.latent_encoders[i] = LatentEncoder(latent_config, checkpointing_level)
            self.latent_decoders[i] = LatentDecoder(latent_config, checkpointing_level)
            if i != len(latent_dims) - 1:  # if not deepest latent
                self.prior_estimators[i] = LatentEncoder(latent_config, checkpointing_level)

        self.latent_space = GaussianLatentSpace()

    def forward(self, i: int, encoding: torch.Tensor, previous_decoder_output: torch.Tensor):
        prior_estimator = self.prior_estimators[i]
        latent_encoder = self.latent_encoders[i]
        latent_decoder = self.latent_decoders[i]

        if latent_encoder is None:
            return previous_decoder_output, None, None, (None, None), (None, None)

        # Create prior
        if i == len(self.config.latent_dims) - 1:  # if deepest latent
            prior_mu, prior_sigma, prior_log_var = None, None, None
        else:
            prior_mu, prior_sigma, prior_log_var = prior_estimator(previous_decoder_output, return_log_var=True)

        # Create posterior
        posterior_mu, posterior_sigma = latent_encoder(encoding, prior_mu, prior_log_var)

        # Create latent
        latent, kl_divergence = self.latent_space(
            posterior_mu, posterior_sigma, prior_mu, prior_sigma, kl_divergence_reduction=None
        )

        # if i == 2:
        #     print(f"a{i}")
        #     latent = torch.randn_like(latent)
        #     if prior_mu is not None:
        #         latent = prior_mu + prior_sigma * latent

        # Decode latent
        latent_decoded = latent_decoder(latent)

        # Combine with previous decoder output
        output = torch.cat((previous_decoder_output, latent_decoded), dim=1)

        return output, latent, kl_divergence, (prior_mu, prior_sigma), (posterior_mu, posterior_sigma)


class Decoder(nn.Module):
    def __init__(self, config, checkpointing_level: int = 0):
        super().__init__()

        depths = config.depths
        num_channels = config.num_channels
        latent_dims = config.latent_dims

        # deepest "latent decoding"
        self.hidden_decoding = nn.Parameter(torch.randn(num_channels[-1]), requires_grad=True)

        self.latent_space_ops = LatentSpaceOps(config, checkpointing_level)

        self.stages = nn.ModuleList([])
        for i in range(len(depths)):
            depth = depths[i]
            channels = num_channels[i]
            has_skip_conn = True
            if latent_dims[i] is None:  # if not concatenating with sampled latent
                has_skip_conn = False

            stage = nn.Sequential()
            stage.append(StageBlock(channels, config, depth, has_skip_conn, checkpointing_level))
            if i > 0:
                stage.append(
                    CNNBlock3D(  # Upsample
                        config,
                        checkpointing_level,
                        in_channels=channels,
                        out_channels=num_channels[i - 1],
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

    def forward(self, encodings: torch.Tensor):
        b, _, z, y, x = encodings[-1].shape

        latents = [None] * len(encodings)
        kl_divergences = [None] * len(encodings)
        prior_distributions = [(None, None)] * len(encodings)
        posterior_distributions = [(None, None)] * len(encodings)

        decoder_output = repeat(self.hidden_decoding, "d -> b d z y x", b=b, z=z, y=y, x=x)
        for i in range(len(encodings) - 1, -1, -1):
            encoding = encodings[i]
            encoding_after_latent, latent, kl_divergence, prior_distribution, posterior_distribution = (
                self.latent_space_ops(i, encoding, decoder_output)
            )
            decoder_output = self.stages[i](encoding_after_latent)

            latents[i] = latent
            kl_divergences[i] = kl_divergence
            prior_distributions[i] = prior_distribution
            posterior_distributions[i] = posterior_distribution

        return decoder_output, latents, kl_divergences, prior_distributions, posterior_distributions


class NVAE(nn.Module):
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
        decoded, latents, kl_divergences, prior_distributions, posterior_distributions = self.decoder(encodings)
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

    config = get_config()

    device = torch.device("cpu")
    device = torch.device("cuda:0")

    autoencoder = NVAE(config.model, 0).to(device)
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

    loss: torch.Tensor = sample_output["reconstructed"].sum()
    tic = perf_counter()
    loss.backward()
    toc = perf_counter()
    backward_time = toc - tic

    final_mem = process.memory_info().rss  # in bytes

    def replace_with_shapes(arr):
        for i in range(len(arr)):
            if isinstance(arr[i], tuple):
                arr[i] = arr[i][0]
            if arr[i] is not None:
                arr[i] = arr[i].shape
        return arr

    print()
    print("Input shape:", sample_input.shape)
    print()
    print("Outputs:")
    print("Reconstructed:", sample_output["reconstructed"].shape)
    print("Latents:", replace_with_shapes(sample_output["latents"]))
    print("kl_loss:", replace_with_shapes(sample_output["kl_divergences"]))
    print("Prior distributions:", replace_with_shapes(sample_output["prior_distributions"]))
    print("Posterior distributions:", replace_with_shapes(sample_output["posterior_distributions"]))
    print()
    print(f"GPU: {torch.cuda.max_memory_allocated() / 2**30} GB peak mem used")
    print(f"RAM: {(final_mem - initial_mem) / 2**30} GB peak mem used")
    print(f"Time (forward): {forward_time:.3f} s")
    print(f"Time (backward): {backward_time:.3f} s")
