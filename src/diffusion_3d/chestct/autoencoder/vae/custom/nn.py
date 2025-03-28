from collections import OrderedDict

import torch
from torch import nn
from vision_architectures.blocks.cnn import CNNBlock3D
from vision_architectures.blocks.mbconv_3d import MBConv3D
from vision_architectures.layers.latent_space import GaussianLatentSpace, LatentDecoder, LatentEncoder
from vision_architectures.nets.maxvit_3d import MaxViT3DStem0
from vision_architectures.nets.swinv2_3d import SwinV23DBlock, SwinV23DPatchMerging, SwinV23DPatchSplitting
from vision_architectures.utils.activation_checkpointing import ActivationCheckpointing
from vision_architectures.utils.rearrange import rearrange_channels
from vision_architectures.utils.residuals import Residual


class ResBlock(nn.Module):
    def __init__(self, config, checkpointing_level: int = 0):
        super().__init__()

        self.layers = nn.ModuleList([])
        for i in range(config.depth):
            self.layers.append(MBConv3D(config, checkpointing_level=checkpointing_level))
            self.layers.append(SwinV23DBlock(config, checkpointing_level=checkpointing_level))

        self.residual = Residual()

        self.checkpointing_level3 = ActivationCheckpointing(3, checkpointing_level)

    def _forward(self, x: torch.Tensor, channels_first: bool = True):
        # x: (b, [dim], z, y, x, [dim])

        x = rearrange_channels(x, channels_first, True)
        # (b, dim, z1, y1, x1)

        for i in range(0, len(self.layers), 2):
            residual = x

            x = self.layers[i](x, channels_first=True)  # mbconv
            x = self.layers[i + 1](x, channels_first=True)  # attention
            # (b, dim, z1, y1, x1)

            x = self.residual(residual, x)

        x = rearrange_channels(x, True, channels_first)
        # (b, [dim], z1, y1, x1, [dim])

        return x

    def forward(self, *args, **kwargs):
        return self.checkpointing_level3(self._forward, *args, **kwargs)


class UnembeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        if len(config.upsample_dims) > 0:
            upsample_fsb_dims = zip([config.dim] + config.upsample_dims[:-1], config.upsample_dims)
            self.upsample_fsbs = nn.ModuleList(
                [
                    nn.Sequential(
                        CNNBlock3D(
                            config,
                            in_channels=in_dim,
                            out_channels=out_dim,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            transposed=True,
                        ),
                        MBConv3D(
                            config,
                            dim=out_dim,
                            out_dim=out_dim,
                        ),
                    )
                    for in_dim, out_dim in upsample_fsb_dims
                ]
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


class Encoder(nn.Sequential):
    def __init__(self, config, checkpointing_level: int = 0):
        embedding = MaxViT3DStem0(config, checkpointing_level)
        res_blocks = [
            nn.Sequential(
                ResBlock(block_config, checkpointing_level),
                (
                    SwinV23DPatchMerging(
                        merge_window_size=block_config.merge_window_size,
                        in_dim=block_config.dim,
                        out_dim=block_config.dim * block_config.out_dim_ratio,
                        checkpointing_level=checkpointing_level,
                    )
                    if "merge_window_size" in block_config
                    else nn.Identity()
                ),
            )
            for block_config in config.blocks
        ]
        latent_encoder = LatentEncoder(config.latent)

        return super().__init__(embedding, *res_blocks, latent_encoder)


class Decoder(nn.Sequential):
    def __init__(self, config, checkpointing_level: int = 0):
        latent_decoder = LatentDecoder(config.latent)
        res_blocks = [
            nn.Sequential(
                (
                    SwinV23DPatchSplitting(
                        final_window_size=block_config.final_window_size,
                        in_dim=block_config.dim * block_config.out_dim_ratio,
                        out_dim=block_config.dim,
                        checkpointing_level=checkpointing_level,
                    )
                    if "final_window_size" in block_config
                    else nn.Identity()
                ),
                ResBlock(block_config, checkpointing_level),
            )
            for block_config in config.blocks
        ]
        unembedding = UnembeddingLayer(config.unembedding)

        return super().__init__(latent_decoder, *res_blocks, unembedding)


class VAE(nn.Module):
    def __init__(self, model_config: dict, checkpointing_level: int = 0):
        super().__init__()

        self.model_config = model_config

        self.encoder = Encoder(model_config.encoder, checkpointing_level)
        self.latent_space = GaussianLatentSpace()
        self.decoder = Decoder(model_config.decoder, checkpointing_level)

    def forward(self, x):
        # x: (b, d1, z1, y1, x1)

        z_mu, z_sigma = self.encoder(x)
        # (b, d2, z2, y2, x2)

        z, kl_loss = self.latent_space(z_mu, z_sigma)

        reconstructed = self.decoder(z)
        # (b, d1, z1, y1, x1)

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

    config = get_config()

    device = torch.device("cpu")
    device = torch.device("cuda:0")

    autoencoder = VAE(config.model, 2).to(device)
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

    print()
    print("Input shape:", sample_input.shape)
    print()
    print("Outputs:")
    print("Reconstructed:", sample_output["reconstructed"].shape)
    print("z_mu:", sample_output["z_mu"].shape)
    print("z_sigma:", sample_output["z_sigma"].shape)
    print("kl_loss:", sample_output["kl_loss"])
    print()
    print(f"GPU: {torch.cuda.max_memory_allocated() / 2**30} GB peak mem used")
    print(f"RAM: {(final_mem - initial_mem) / 2**30} GB peak mem used")
    print(f"Time (forward): {forward_time:.3f} s")
    print(f"Time (backward): {backward_time:.3f} s")

    # losses = autoencoder.training_step({"scan": sample_input, "spacing": ...}, 0)
    # from pprint import pprint
    # pprint(losses)
    # pprint(losses)
    # pprint(losses)
