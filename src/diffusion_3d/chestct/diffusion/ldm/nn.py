import numpy as np
import torch
from einops import rearrange
from munch import Munch
from torch import nn
from vision_architectures.blocks.cnn import CNNBlock3D
from vision_architectures.blocks.transformer import TransformerEncoderBlock3D
from vision_architectures.layers.embeddings import (
    AbsolutePositionEmbeddings3D,
    get_absolute_position_embeddings_3d,
    get_timestep_embeddings_1d,
)
from vision_architectures.layers.scale import PixelShuffleDownsample3D, PixelShuffleUpsample3D
from vision_architectures.utils.activation_checkpointing import ActivationCheckpointing
from vision_architectures.utils.activations import get_act_layer
from vision_architectures.utils.rearrange import rearrange_channels
from vision_architectures.utils.residuals import Residual, add_stochastic_depth_dropout


class Conditioning(nn.Module):
    def __init__(self, out_channels: int, activation: str = "silu"):
        super().__init__()

        self.out_channels = out_channels

        def get_embedding_module(in_channels: int, out_channels: int):
            return nn.Sequential(
                nn.Linear(in_channels, out_channels),
                get_act_layer(activation),
                nn.Linear(out_channels, out_channels),
            )

        self.proj_noise_param = get_embedding_module(out_channels, out_channels)
        self.proj_spacings = get_embedding_module(3, out_channels)
        self.proj_offset = get_embedding_module(out_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        noise_params: torch.Tensor,
        spacings: torch.Tensor,
        offsets: torch.Tensor,
        guidance: torch.Tensor,
    ):
        # guidance shape: (num_conditioning_factors, b)

        # noise_param conditioning
        # noise_params: (b,)
        noise_param = get_timestep_embeddings_1d(self.out_channels, noise_params)[0]
        # (b, d)
        noise_param_cond = self.proj_noise_param(noise_param)
        # (b, d)
        noise_param_cond = guidance[0, :, None] * noise_param_cond
        # (b, d)
        noise_param_cond = rearrange(noise_param_cond, "b d -> b d 1 1 1")
        # (b, d, 1, 1, 1)
        x = x + noise_param_cond

        # spacings conditioning
        spacings_cond = self.proj_spacings(spacings)
        # (b, d)
        spacings_cond = guidance[1, :, None] * spacings_cond
        # (b, d)
        spacings_cond = rearrange(spacings_cond, "b d -> b d 1 1 1")
        # (b, d, 1, 1, 1)
        x = x + spacings_cond

        # Offset conditioning
        # offsets: (b, 3)
        offsets_emb = []
        for offset in offsets:
            offsets_emb.append(
                get_absolute_position_embeddings_3d(self.out_channels, (1, 1, 1), crop_offset=offset.cpu().tolist())[
                    :, :, 0, 0, 0
                ]
            )
        offsets_emb = torch.cat(offsets_emb, dim=0).to(x.device)
        # (b, d)
        offsets_cond = self.proj_offset(offsets_emb)
        # (b, d)
        offsets_cond = guidance[2, :, None] * offsets_cond
        # (b, d)
        offsets_cond = rearrange(offsets_cond, "b d -> b d 1 1 1")
        # (b, d, 1, 1, 1)
        x = x + offsets_cond

        return x


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

        self.condition = Conditioning(out_channels, model_config.activation)

    def forward(self, x, noise_params, spacings, offsets, guidance: torch.Tensor):
        res = self.conv_res(x)
        x = self.conv1(x)
        x = self.condition(x, noise_params, spacings, offsets, guidance)
        x = self.conv2(x)
        x = self.residual(x, res)
        return x


class StageBlock(nn.Module):
    def __init__(
        self, num_channels, config, depth, has_skip_conn: bool, attn_num_heads: bool, checkpointing_level: int = 0
    ):
        super().__init__()

        in_channels = [num_channels] * depth
        out_channels = [num_channels] * depth
        if has_skip_conn:
            in_channels[0] = num_channels * 2

        self.position_embeddings = AbsolutePositionEmbeddings3D()

        self.layers = nn.ModuleList([])
        for in_channel, out_channel in zip(in_channels, out_channels):
            self.layers.append(ResBlock(in_channel, out_channel, config, checkpointing_level))
            if attn_num_heads is not None:
                self.layers.append(
                    TransformerEncoderBlock3D(
                        dim=num_channels, num_heads=attn_num_heads, checkpointing_level=checkpointing_level
                    ),
                )

        self.checkpointing_level3 = ActivationCheckpointing(3, checkpointing_level)

    def _forward(
        self,
        x: torch.Tensor,
        noise_params: torch.Tensor,
        spacings: torch.Tensor,
        offsets: torch.Tensor,
        guidance: torch.Tensor,
        channels_first: bool = True,
    ):
        # x: (b, [dim], z, y, x, [dim])

        x = rearrange_channels(x, channels_first, True)
        # (b, dim, z, y, x)

        for layer in self.layers:
            if isinstance(layer, TransformerEncoderBlock3D):
                x = layer(self.position_embeddings(x))
            else:
                x = layer(x, noise_params, spacings, offsets, guidance)
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

            stage = nn.ModuleList()
            if i > 0:
                stage.append(
                    PixelShuffleDownsample3D(  # Downsample
                        config,
                        checkpointing_level,
                        in_channels=config.num_channels[i - 1],
                        out_channels=channels,
                        kernel_size=1,
                        activation=None,
                        normalization=None,
                    )
                )

            stage.append(StageBlock(channels, config, depth, False, config.attn_num_heads[i], checkpointing_level))
            self.stages.append(stage)

    def forward(
        self,
        x: torch.Tensor,
        noise_params: torch.Tensor,
        spacings: torch.Tensor,
        offsets: torch.Tensor,
        guidance: torch.Tensor,
    ):
        encodings = []
        for stage in self.stages:
            for i in range(len(stage)):
                if i == len(stage) - 1:
                    x = stage[i](x, noise_params, spacings, offsets, guidance)
                else:
                    x = stage[i](x)
            encodings.append(x)
        return encodings


class Processor(nn.Module):
    def __init__(self, config: Munch, checkpointing_level: int = 0):
        super().__init__()

        channels = config.num_channels[-1]

        self.stage = StageBlock(
            channels, config, config.mid_depth, False, config.attn_num_heads[-1], checkpointing_level
        )

    def forward(
        self,
        x: torch.Tensor,
        noise_params: torch.Tensor,
        spacings: torch.Tensor,
        offsets: torch.Tensor,
        guidance: torch.Tensor,
    ):
        encodings = self.stage(x, noise_params, spacings, offsets, guidance)
        return encodings


class Decoder(nn.Module):
    def __init__(self, config, checkpointing_level: int = 0):
        super().__init__()

        depths = config.depths
        num_channels = config.num_channels

        self.stages = nn.ModuleList([])
        for i in range(len(depths)):
            depth = depths[i]
            channels = num_channels[i]
            has_skip_conn = i < len(depths) - 1

            stage = nn.ModuleList()
            stage.append(
                StageBlock(
                    channels,
                    config,
                    depth,
                    has_skip_conn,
                    config.attn_num_heads[i],
                    checkpointing_level,
                )
            )
            if i > 0:
                stage.append(
                    PixelShuffleUpsample3D(  # Upsample
                        config,
                        checkpointing_level,
                        in_channels=channels,
                        out_channels=num_channels[i - 1],
                        kernel_size=1,
                        activation=None,
                        normalization=None,
                    )
                )

            self.stages.append(stage)

    def forward(
        self,
        mid_out: torch.Tensor,
        encodings: torch.Tensor,
        noise_params: torch.Tensor,
        spacings: torch.Tensor,
        offsets: torch.Tensor,
        guidance: torch.Tensor,
    ):
        for i in range(len(encodings) - 1, -1, -1):
            if i == len(encodings) - 1:
                decoder_input = mid_out
            else:
                encoding = encodings[i]
                decoder_input = torch.cat([decoder_output, encoding], dim=1)

            for j in range(len(self.stages[i])):
                if j == 0:
                    decoder_output = self.stages[i][j](decoder_input, noise_params, spacings, offsets, guidance)
                else:
                    decoder_output = self.stages[i][j](decoder_output)

        return decoder_output


class LDM3D(nn.Module):
    def __init__(self, model_config: dict, checkpointing_level: int = 0):
        super().__init__()

        self.model_config = model_config

        self.encoder_mapping = CNNBlock3D(
            checkpointing_level=checkpointing_level,
            in_channels=model_config.in_channels,
            out_channels=model_config.num_channels[0],
            kernel_size=3,
            normalization=None,
            activation=None,
        )
        self.encoder = Encoder(model_config, checkpointing_level)
        self.processor = Processor(model_config, checkpointing_level)
        self.decoder = Decoder(model_config, checkpointing_level)
        self.decoder_mapping = CNNBlock3D(
            model_config,
            checkpointing_level,
            in_channels=model_config.num_channels[0],
            out_channels=model_config.in_channels,
            kernel_size=3,
            normalization=None,
            activation=None,
        )

        if model_config.survival_prob < 1.0:
            self.encoder = add_stochastic_depth_dropout(self.encoder, model_config.survival_prob)
            self.decoder = add_stochastic_depth_dropout(self.decoder, model_config.survival_prob)

    def forward(self, x, noise_params, spacings, offsets, guidance):
        x = self.encoder_mapping(x)
        encodings = self.encoder(x, noise_params, spacings, offsets, guidance)
        mid_out = self.processor(encodings[-1], noise_params, spacings, offsets, guidance)
        decoded = self.decoder(mid_out, encodings, noise_params, spacings, offsets, guidance)
        noise = self.decoder_mapping(decoded)

        return noise


if __name__ == "__main__":
    from time import perf_counter

    import psutil
    from arjcode.visualize import describe_model
    from config import get_config

    config = get_config()

    device = torch.device("cpu")
    # device = torch.device("cuda:0")

    ldm = LDM3D(config.model, 0).to(device)
    describe_model(ldm)

    ldm.train()

    # Track memory before execution
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    initial_mem = process.memory_info().rss  # in bytes

    sample_input = torch.zeros((1, 1, *[d // 4 for d in config.input_size])).to(device)
    noise_param = torch.randint(0, config.model.timesteps, (1,)).to(device)  # Dummy noise_param
    spacings = torch.rand((1, 3)).to(device)
    offsets = torch.rand((1, 3)).to(device)
    guidance = torch.randint(0, 5, (3, 1)).to(device)

    tic = perf_counter()
    sample_output = ldm(sample_input, noise_param, spacings, offsets, guidance)
    toc = perf_counter()
    forward_time = toc - tic

    loss: torch.Tensor = sample_output.sum()
    tic = perf_counter()
    loss.backward()
    toc = perf_counter()
    backward_time = toc - tic

    final_mem = process.memory_info().rss  # in bytes

    print()
    print("Input shape:", sample_input.shape)
    print("Output shape:", sample_output.shape)
    print()
    print(f"GPU: {torch.cuda.max_memory_allocated() / 2**30} GB peak mem used")
    print(f"RAM: {(final_mem - initial_mem) / 2**30} GB peak mem used")
    print(f"Time (forward): {forward_time:.3f} s")
    print(f"Time (backward): {backward_time:.3f} s")
