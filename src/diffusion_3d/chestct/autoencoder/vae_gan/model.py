import lightning as L
import numpy as np
import torch
from einops import rearrange, repeat
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.blocks.convolutions import Convolution
from monai.networks.nets import PatchDiscriminator
from monai.networks.nets.autoencoderkl import AutoencoderKL, Decoder
from monai.utils import set_determinism
from munch import Munch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.nn import functional as F
from vision_architectures.layers.embeddings import AbsolutePositionEmbeddings3D
from vision_architectures.nets.swinv2_3d import SwinV23DConfig, SwinV23DDecoder, SwinV23DModel
from vision_architectures.nets.vit_3d import ViT3DDecoder


class AdaptiveVAEGAN(AutoencoderKL, L.LightningModule):
    def __init__(self, model_config: dict, training_config: Munch):
        super().__init__(spatial_dims=3, latent_channels=model_config.adaptor.dim)
        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config

        self.encoder = SwinV23DModel(model_config.swin)

        self.adaptive_queries = nn.Parameter(
            torch.empty(model_config.adaptor.dim, *model_config.adaptor.adaptive_queries_size),
            requires_grad=True,
        )
        torch.nn.init.uniform_(self.adaptive_queries, -0.02, 0.02)
        self.adaptor = ViT3DDecoder(model_config.adaptor)

        if isinstance(model_config.decoder, SwinV23DConfig):
            self.decoder = SwinV23DDecoder(model_config.decoder)
            self.final_layer = nn.Conv3d(
                in_channels=model_config.final_layer.in_channels,
                out_channels=model_config.final_layer.out_channels,
                kernel_size=1,
                stride=1,
            )
        else:
            self.decoder = Decoder(**model_config.decoder)
            self.final_layer = nn.Sequential(
                Convolution(
                    spatial_dims=3,
                    in_channels=model_config.final_layer.in_channels,
                    out_channels=model_config.final_layer.out_channels,
                    kernel_size=3,
                    strides=2,
                    is_transposed=True,
                ),
                Convolution(
                    spatial_dims=3,
                    in_channels=model_config.final_layer.out_channels,
                    out_channels=model_config.final_layer.out_channels,
                    kernel_size=3,
                    strides=2,
                    is_transposed=True,
                ),
            )

        self.discriminator = PatchDiscriminator(
            spatial_dims=3,
            num_layers_d=3,
            channels=32,
            in_channels=model_config.final_layer.out_channels,
            out_channels=model_config.final_layer.out_channels,
            # in_channels=model_config.decoder.out_channels,
            # out_channels=model_config.decoder.out_channels,
            norm="INSTANCE",
        )

        self.l2_loss = L1Loss()
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
        z_mu_squared = z_mu.pow(2)
        z_sigma_squared = z_sigma.pow(2)
        kl_loss = 0.5 * (z_mu_squared + z_sigma_squared - torch.log(z_sigma.pow(2)) - 1)

        # additional plotting
        self.log_dict(
            {
                "train_kl_step/mu": z_mu_squared.mean(),
                "train_kl_step/sigma": z_sigma_squared.mean(),
            },
            sync_dist=True,
        )

        return torch.mean(kl_loss)

    def calculate_perceptual_loss(self, reconstructed, x):
        return self.perceptual_loss(reconstructed.float(), x.float())

    def calculate_basic_losses(self, x, reconstructed, z_mu, z_sigma):
        return {
            "reconstruction_loss": self.calculate_reconstruction_loss(reconstructed, x),
            "kl_loss": self.calculate_kl_loss(z_mu, z_sigma),
            "perceptual_loss": self.calculate_perceptual_loss(reconstructed, x),
        }

    def process_step(self, batch, losses: list, prefix, batch_idx):
        x = batch

        reconstructed, decoded, adapted_encoded, encoded_mu, encoded_sigma = self(x, prefix)

        # Reshape back to input shape
        reconstructed = F.interpolate(reconstructed, batch.shape[2:], mode="trilinear")

        all_losses = self.calculate_basic_losses(x, reconstructed, encoded_mu, encoded_sigma)

        kl_beta = min(1, self.current_epoch / 10)
        all_losses["kl_loss"] = kl_beta * all_losses["kl_loss"]

        if prefix == "train":
            optimizer_main, optimizer_disc = self.optimizers()
            scheduler_main, scheduler_disc = self.lr_schedulers()

        logits_fake = self.discriminator(reconstructed.contiguous().float())[-1]

        generator_loss_time_weight = min(1, self.current_epoch / 20)
        all_losses["generator_loss"] = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        all_losses["generator_loss"] = generator_loss_time_weight * all_losses["generator_loss"]

        all_losses["autoencoder_loss"] = (
            self.training_config.loss_weights["reconstruction_loss"] * all_losses["reconstruction_loss"]
            + self.training_config.loss_weights["kl_loss"] * all_losses["kl_loss"]
            + self.training_config.loss_weights["perceptual_loss"] * all_losses["perceptual_loss"]
            + self.training_config.loss_weights["generator_loss"] * all_losses["generator_loss"]
        )

        update_main_every_n_steps = 5
        update_disc_every_n_steps = 50
        update_main_step = (batch_idx + 1) % update_main_every_n_steps == 0
        update_disc_step = (batch_idx + 1) % update_disc_every_n_steps == 0

        if prefix == "train":
            self.manual_backward(all_losses["autoencoder_loss"])
            if update_main_step:
                self.clip_gradients(optimizer_main, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                optimizer_main.step()
                optimizer_main.zero_grad(set_to_none=True)

        logits_fake = self.discriminator(reconstructed.contiguous().detach())[-1]
        all_losses["disc_fake_loss"] = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(x.contiguous().detach())[-1]
        all_losses["disc_real_loss"] = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        all_losses["discriminator_loss"] = (
            self.training_config.loss_weights["disc_fake_loss"] * all_losses["disc_fake_loss"]
            + self.training_config.loss_weights["disc_real_loss"] * all_losses["disc_real_loss"]
        )

        if prefix == "train":
            self.manual_backward(all_losses["discriminator_loss"])
            if update_disc_step:
                self.clip_gradients(optimizer_disc, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=True)

        if prefix == "train":
            if update_main_step:
                scheduler_main.step()
            if update_disc_step:
                scheduler_disc.step()

        if prefix == "train":
            losses_log = {}
            for key, value in all_losses.items():
                losses_log[f"train_step/{key}"] = float(value)
                losses_log[f"train_step_scaled/{key}"] = float(self.training_config.loss_weights.get(key, 1.0) * value)
            try:
                self.log_dict(losses_log, sync_dist=True)
            except:
                print(f"Error in logging losses {losses_log}")

        losses.append({key: value.detach().cpu() for key, value in all_losses.items()})

        return all_losses, reconstructed, decoded, adapted_encoded, encoded_mu, encoded_sigma

    def process_epoch(self, losses, prefix):
        losses = {key: [d[key].item() for d in losses] for key in losses[0]}
        losses = {key: np.mean(value).item() for key, value in losses.items()}
        for key, loss in losses.items():
            try:
                self.log(f"{prefix}_epoch/{key}", float(loss), sync_dist=True)
                self.log(
                    f"{prefix}_epoch_scaled/{key}",
                    float(self.training_config.loss_weights.get(key, 1.0) * loss),
                    sync_dist=True,
                )
            except:
                print(f"Error in logging losses {loss}, {type(loss)}")

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
        try:
            self.log("train_grad/norm", norm, sync_dist=True)
            self.log("train_grad/max_abs", max_abs, sync_dist=True)
        except:
            print(f"Error in logging gradients {norm}, {max_abs}")

    def training_step(self, batch, batch_idx):
        self.process_step(batch["scan"], self.train_losses, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        self.process_step(batch["scan"], self.val_losses, "val", batch_idx)

    def on_train_epoch_end(self):
        self.process_epoch(self.train_losses, "train")
        self.train_losses.clear()

    def on_validation_epoch_end(self):
        self.process_epoch(self.val_losses, "val")
        self.val_losses.clear()

    def configure_optimizers(self):
        main_parameters = (
            list(self.encoder.parameters())
            + [self.adaptive_queries]
            + list(self.adaptor.parameters())
            + list(self.decoder.parameters())
        )
        if hasattr(self, "final_layer"):
            main_parameters += list(self.final_layer.parameters())

        optimizer_main = torch.optim.Adam(main_parameters, lr=self.training_config.inital_lr)
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
        print("{}: Epoch = {:<4}".format(prefix, self.current_epoch))
        for key, loss in losses.items():
            print("{} = {:.5f}".format(key.ljust(20), loss))
        print()

    def encode(self, x: torch.Tensor):
        encoded, _, _ = self.encoder(x)

        # Add position embeddings
        position_embeddings = AbsolutePositionEmbeddings3D(encoded.shape[1], encoded.shape[2:], learnable=False)(
            batch_size=encoded.shape[0]
        ).to(encoded.device)
        encoded = encoded + position_embeddings

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

        z_mu = self.quant_conv_mu(adapted)
        z_log_var = self.quant_conv_log_sigma(adapted)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma

    def decode(self, z: torch.Tensor):
        z = self.post_quant_conv(z)
        if isinstance(self.decoder, SwinV23DDecoder):
            z = rearrange(z, "b d z y x -> b z y x d")
            dec, _, _ = self.decoder(z)
            dec = rearrange(dec, "b z y x d -> b d z y x")
        else:
            dec = self.decoder(z)
        return dec

    def forward(self, x, run_type):
        # x: (b, d1, z1, y1, x1)

        encoded_mu, encoded_sigma = self.encode(x)
        if run_type == "train":
            adapted_encoded = self.sampling(encoded_mu, encoded_sigma)
        else:
            adapted_encoded = encoded_mu
        # (b, d2, z2, y2, x2)

        decoded = self.decode(adapted_encoded)
        # (b, d3, z3, y3, x3)

        if hasattr(self, "final_layer"):
            reconstructed = self.final_layer(decoded)
        else:
            reconstructed = decoded
        # (b, d1, z1, y1, x1)

        reconstructed = torch.tanh(reconstructed)  # this fits it into (-1., 1.)

        return reconstructed, decoded, adapted_encoded, encoded_mu, encoded_sigma

    # def on_before_zero_grad(self, optimizer):
    #     """Will print all unused parameters"""
    #     if self.global_rank == 0:
    #         print("Zero grad params")
    #         for name, param in self.named_parameters():
    #             if param.grad is None:
    #                 print(name)
    #         print()


if __name__ == "__main__":
    import psutil
    from neuro_utils.describe import describe_model

    from diffusion_3d.chestct.autoencoder.config import get_config

    config = get_config()

    device = torch.device("cpu")
    # device = torch.device("cuda:0")

    autoencoder = AdaptiveVAEGAN(config.model, config.training).to(device)
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

    # Track memory before execution
    process = psutil.Process()
    initial_mem = process.memory_info().rss  # in bytes
    torch.cuda.reset_peak_memory_stats()

    sample_input = torch.zeros((1, 1, *config.image_size)).to(device)
    sample_output = autoencoder(sample_input, "train")

    final_mem = process.memory_info().rss  # in bytes

    print(sample_input.shape)
    print([x.shape for x in sample_output])
    print(f"GPU: {torch.cuda.max_memory_allocated() / 2**30} GB peak mem used")
    print(f"RAM: {(final_mem - initial_mem) / 2**30} GB peak mem used")

    # losses = autoencoder.training_step({"scan": sample_input, "spacing": ...}, 0)
    # from pprint import pprint
    # pprint(losses)
