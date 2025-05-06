import torch
from arjcode.model import freeze_module
from hydra.utils import instantiate
from torch import nn


class FrozenMaisiVAE(nn.Module):
    maisi_vae_config = {
        "_target_": "monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi",
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": 4,
        "num_channels": [64, 128, 256],
        "num_res_blocks": [2, 2, 2],
        "norm_num_groups": 32,
        "norm_eps": 1e-06,
        "attention_levels": [False, False, False],
        "with_encoder_nonlocal_attn": False,
        "with_decoder_nonlocal_attn": False,
        "use_checkpointing": False,
        "use_convtranspose": False,
        "norm_float16": False,
        "num_splits": 8,
        "dim_split": 1,
        "save_mem": False,
    }

    def __init__(self, checkpoint_path: str = r"/raid3/arjun/checkpoints/maisi/autoencoder_epoch273.pt"):
        super().__init__()
        self.checkpoint_path = checkpoint_path

        self.model: nn.Module = instantiate(self.maisi_vae_config)
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"))
        freeze_module(self)

    @torch.no_grad()
    def encode(self, *args, sample: bool = True, **kwargs):
        if sample:
            return self.model.encode_stage_2_inputs(*args, **kwargs)
        else:
            return self.model.encode(*args, **kwargs)[0]

    @torch.no_grad()
    def decode(self, *args, **kwargs):
        return self.model.decode_stage_2_outputs(*args, **kwargs)


if __name__ == "__main__":
    from arjcode.visualize import describe_model

    vae = FrozenMaisiVAE()
    print(vae.model)
    describe_model(vae.model, True)

    x = torch.randn(1, 1, 96, 96, 96)
    print(x.shape)
    encoded = vae.encode(x)
    print(encoded.shape)
    decoded = vae.decode(encoded)
    print(decoded.shape)
