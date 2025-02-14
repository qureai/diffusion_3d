import lightning as L
import numpy as np
import torch
from einops import rearrange, repeat
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from monai.networks.nets.autoencoderkl import AutoencoderKL, Decoder
from monai.utils import set_determinism
from munch import Munch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.nn import functional as F
from vision_architectures.nets.swinv2_3d import SwinV23DDecoder, SwinV23DModel
from vision_architectures.nets.vit_3d import ViT3DDecoder


class AdaptiveAutoEncoder(AutoencoderKL, L.LightningModule):
    def __init__(self, model_config: dict, training_config: Munch):
        super().__init__(spatial_dims=3, latent_channels=model_config.adaptor.dim)
        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config

        self.encoder = SwinV23DModel(model_config.swin)
        self.adaptor = ViT3DDecoder(model_config.adaptor)
        # self.decoder = Decoder(**model_config.decoder)
        self.decoder = SwinV23DDecoder(model_config.decoder)
        self.final_layer = nn.ConvTranspose3d(
            in_channels=model_config.final_layer.in_channels,
            out_channels=model_config.final_layer.out_channels,
            kernel_size=model_config.final_layer.kernel_size,
            stride=model_config.final_layer.kernel_size,
        )
        self.discriminator = PatchDiscriminator(
            spatial_dims=3,
            num_layers_d=3,
            channels=32,
            in_channels=model_config.final_layer.out_channels,
            out_channels=model_config.final_layer.out_channels,
            norm="INSTANCE",
        )

        self.adaptive_queries = nn.Parameter(
            torch.rand(model_config.adaptor.dim, *model_config.adaptor.adaptive_queries_size),
            requires_grad=True,
        )

        self.l2_loss = MSELoss()
        self.perceptual_loss = PerceptualLoss(
            spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
        ).eval()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")

        self.automatic_optimization = False  # Manual optimization required as there are two optimizers

        self.train_losses = []
        self.val_losses = []

    def calculate_reconstruction_loss(self, reconstructed, x):
        return self.l2_loss(reconstructed, x)

    def calculate_kl_loss(self, z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
            dim=list(range(1, len(z_sigma.shape))),
        )
        return torch.sum(kl_loss) / kl_loss.shape[0]

    def calculate_reconstruction_loss(self, reconstructed, x):
        return self.perceptual_loss(reconstructed.float(), x.float())

    def calculate_basic_losses(self, x, reconstructed, z_mu, z_sigma):
        return {
            "reconstruction_loss": self.calculate_reconstruction_loss(reconstructed, x),
            "kl_loss": self.calculate_kl_loss(z_mu, z_sigma),
            "perceptual_loss": self.calculate_reconstruction_loss(reconstructed, x),
        }

    def process_step(self, batch, losses: list, prefix, batch_idx):
        x = batch

        reconstructed, decoded, adapted, encoded, encoded_mu, encoded_sigma = self(x)

        # Reshape back to input shape
        reconstructed = F.interpolate(reconstructed, batch.shape[2:], mode="trilinear")

        basic_losses = self.calculate_basic_losses(x, reconstructed, encoded_mu, encoded_sigma)

        all_losses = basic_losses
        if prefix == "train":
            optimizer_main, optimizer_disc = self.optimizers()
            scheduler_main, scheduler_disc = self.lr_schedulers()

            logits_fake = self.discriminator(reconstructed.contiguous().float())[-1]
            generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

            loss_main = (
                basic_losses["reconstruction_loss"]
                + 1e-7 * basic_losses["kl_loss"]
                + 0.3 * basic_losses["perceptual_loss"]
                + 0.1 * generator_loss
            )
            self.manual_backward(loss_main)
            self.clip_gradients(optimizer_main, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            if (batch_idx + 1) % 5 == 0:
                optimizer_main.step()
                optimizer_main.zero_grad()

            logits_fake = self.discriminator(reconstructed.contiguous().detach())[-1]
            loss_disc_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.discriminator(x.contiguous().detach())[-1]
            loss_disc_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            loss_disc = (loss_disc_fake + loss_disc_real) * 0.5
            self.manual_backward(loss_disc)
            self.clip_gradients(optimizer_disc, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            if (batch_idx + 1) % 5 == 0:
                optimizer_disc.step()
                optimizer_disc.zero_grad()

            scheduler_main.step()
            scheduler_disc.step()

            all_losses["generator_loss"] = generator_loss
            all_losses["loss_disc_fake"] = loss_disc_fake
            all_losses["loss_disc_real"] = loss_disc_real

            losses_log = {}
            for key, value in all_losses.items():
                losses_log[f"train_step/{key}"] = value
            # self.log_dict(losses_log, sync_dist=True)

        losses.append(all_losses)

        return all_losses

    def process_epoch(self, losses, prefix):
        losses = {key: np.mean([d[key] for d in losses]).item() for key in losses[0]}
        for key, loss in losses.items():
            self.log(f"{prefix}_loss/{key}", loss, sync_dist=True)

        self.print_metrics(f"{prefix}", losses)

        return loss

    def on_after_backward(self):
        # Log gradient info
        norm = 0.0
        max_abs = 0.0
        for param in self.parameters():
            if param.grad is not None:
                norm += param.grad.detach().norm(2).item() ** 2
                max_abs = max(max_abs, param.grad.detach().abs().max().item())
        norm = norm**0.5
        self.log("train_grad/norm", norm, sync_dist=True)
        self.log("train_grad/max_abs", max_abs, sync_dist=True)

    def training_step(self, batch, batch_idx):
        return self.process_step(batch["scan"], self.train_losses, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.process_step(batch["scan"], self.val_losses, "val", batch_idx)

    def on_train_epoch_end(self):
        self.process_epoch(self.train_losses, "train")
        self.train_losses.clear()

    def on_validation_epoch_end(self):
        self.process_epoch(self.val_losses, "val")
        self.val_losses.clear()

    def configure_optimizers(self):
        optimizer_main = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.adaptor.parameters())
            + list(self.decoder.parameters())
            + list(self.final_layer.parameters()),
            lr=self.training_config.inital_lr,
        )
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.training_config.inital_lr)

        total_steps = self.trainer.estimated_stepping_batches
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_main, total_steps)
        scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc, total_steps)
        return [
            {"optimizer": optimizer_main, "lr_scheduler": scheduler_main},
            {"optimizer": optimizer_disc, "lr_scheduler": scheduler_disc},
        ]

    def print_metrics(self, prefix, losses):
        if self.global_rank != 0:
            return

        print()
        print()
        print("{}: Epoch = {:<4} | Loss = {:.5f} |".format(prefix, self.current_epoch))
        for key, loss in losses.items():
            print("{} = {:.5f} |".format(key.ljust(20), loss))
        print()

    def encode(self, x: torch.Tensor):
        h, _, _ = self.encoder(x)

        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma

    def decode(self, z: torch.Tensor):
        z = self.post_quant_conv(z)
        z = rearrange(z, "b d z y x -> b z y x d")
        dec, _, _ = self.decoder(z)
        dec = rearrange(dec, "b z y x d -> b d z y x")
        return dec

    def forward(self, x):
        # x: (b, d1, z1, y1, x1)

        encoded_mu, encoded_sigma = self.encode(x)
        encoded = self.sampling(encoded_mu, encoded_sigma)
        # (b, d2, z2, y2, x2)

        batch_size = x.shape[0]
        adaptive_queries = repeat(self.adaptive_queries, f"d z y x -> {batch_size} (z y x) d")
        encoded = rearrange(encoded, "b d z y x -> b (z y x) d")
        adapted, _ = self.adaptor(adaptive_queries, encoded)
        # (b, d2, z2, y2, x2)

        adapted = rearrange(
            adapted,
            "b (z y x) d -> b d z y x",
            z=self.adaptive_queries.shape[1],
            y=self.adaptive_queries.shape[2],
            x=self.adaptive_queries.shape[3],
        )
        decoded = self.decode(adapted)
        # (b, d3, z3, y3, x3)

        reconstructed = self.final_layer(decoded)
        # (b, d1, z1, y1, x1)

        return reconstructed, decoded, adapted, encoded, encoded_mu, encoded_sigma

    # def on_before_zero_grad(self, optimizer):
    #     """Will print all unused parameters"""
    #     if self.global_rank == 0:
    #         print("Zero grad params")
    #         for name, param in self.named_parameters():
    #             if param.grad is None:
    #                 print(name)
    #         print()


if __name__ == "__main__":
    from neuro_utils.describe import describe_model

    from diffusion_3d.chestct.autoencoder.config import get_config

    cfg = get_config()

    device = torch.device("cpu")
    # device = torch.device("cuda:0")

    autoencoder = AdaptiveAutoEncoder(cfg.model, cfg.training).to(device)
    print("Encoder:")
    describe_model(autoencoder.encoder)
    print(f"Adaptive queries:: {autoencoder.adaptive_queries.numel()}")
    print("Adaptor:")
    describe_model(autoencoder.adaptor)
    print("Decoder:")
    describe_model(autoencoder.decoder)
    print("Final layer:")
    describe_model(autoencoder.final_layer)
    print("Discriminator:")
    describe_model(autoencoder.discriminator)

    autoencoder.train()
    torch.cuda.reset_peak_memory_stats()

    sample_input = torch.zeros((2, 1, 256, 256, 256)).to(device)
    sample_output = autoencoder(sample_input)
    print([x.shape for x in sample_output])
    print(f"{torch.cuda.max_memory_allocated() / 2**30} GB GPU mem used")

    # losses = autoencoder.training_step({"scan": sample_input, "spacing": ...}, 0)
    # from pprint import pprint
    # pprint(losses)
