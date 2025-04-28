import torch
from arjcode.model import MyLightningModule, freeze_module, freeze_modules, unfreeze_modules
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.metrics import MultiScaleSSIMMetric, PSNRMetric
from monai.networks.nets import PatchDiscriminator
from munch import Munch
from torch.nn import L1Loss
from vision_architectures.schedulers.cyclic import CyclicAnnealingScheduler
from vision_architectures.schedulers.lrs import ConstantLRWithWarmup
from vision_architectures.utils.clamping import floor_softplus_clamp

from diffusion_3d.chestct.autoencoder.nvae.cnn.nn import NVAE


class NVAELightning(MyLightningModule):
    def __init__(self, model_config: dict, training_config: Munch):
        super().__init__(
            # identify_unused_parameters=True,
            # print_small_gradient_norms=True,
            # print_large_gradient_norms=True,
        )
        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config

        self.autoencoder = NVAE(model_config, training_config.checkpointing_level)

        self.discriminator = PatchDiscriminator(
            spatial_dims=3,
            channels=32,
            in_channels=model_config.in_channels,
            norm="INSTANCE",
        )
        self.automatic_optimization = False

        self.reconstruction_loss = L1Loss()
        self.perceptual_loss = PerceptualLoss(
            spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
        )
        freeze_module(self.perceptual_loss)

        self.adv_loss = PatchAdversarialLoss()

        self.free_nats_per_dim = training_config.free_nats_per_dim

        self.psnr_metric = PSNRMetric(max_val=2.0)
        self.ms_ssim_metric = MultiScaleSSIMMetric(spatial_dims=3, data_range=2.0, kernel_size=7)

        self.kl_beta_schedulers = {key: CyclicAnnealingScheduler(0, 1) for key in training_config.kl_annealing.keys()}
        self.discriminator_scheduler = CyclicAnnealingScheduler(0, 1)

        self.freeze_scales(training_config.freeze_scales)

    def freeze_scales(self, scales):
        scales = sorted(scales)
        if 0 in scales:
            freeze_module(self.autoencoder.encoder_mapping)
            freeze_module(self.autoencoder.decoder_mapping)
        for scale in scales:
            freeze_modules(
                [
                    self.autoencoder.encoder.stages[scale],
                    self.autoencoder.decoder.stages[scale],
                ]
            )
            print(f"Freezing scale: {scale}")
        unfreeze_modules(
            [
                self.autoencoder.decoder.stages[scales[-1]][0].layers[0].conv1.conv,
                self.autoencoder.decoder.stages[scales[-1]][0].layers[0].conv_res.conv,
            ]
        )
        print(f"Unfreezing autoencoder.decoder.stages[{scales[-1]}][0].layers[0].conv1.conv and conv_res.conv")

    def calculate_reconstruction_loss(self, reconstructed, x):
        return self.reconstruction_loss(reconstructed, x)

    def calculate_perceptual_loss(self, reconstructed, x):
        return self.perceptual_loss(reconstructed.float(), x.float()).clamp(min=0.0)

    def calculate_ms_ssim_loss(self, reconstructed, x):
        return 1 - self.calculate_ms_ssim(reconstructed, x)

    def calculate_kl_losses(self, kl_divergences, prior_distributions, posterior_distributions):
        # Prepare beta annealing and get betas
        kl_betas = {}
        for key in self.kl_beta_schedulers:
            kl_beta_scheduler = self.kl_beta_schedulers[key]
            if not kl_beta_scheduler.is_ready():
                try:
                    steps_per_epoch = self.get_steps_per_epoch()
                except RuntimeError:
                    steps_per_epoch = 100
                kl_annealing_wavelength = self.training_config.kl_annealing[key]["wavelength"]
                kl_annealing_wavelength_steps = kl_annealing_wavelength * steps_per_epoch
                self.kl_beta_schedulers[key].set_num_annealing_steps(
                    2 * kl_annealing_wavelength_steps // 3, kl_annealing_wavelength_steps // 3, 0, 0
                )

            if self.current_epoch >= self.training_config.kl_annealing[key]["start_epoch"]:
                kl_betas[key] = kl_beta_scheduler.get()
                if self.training:
                    kl_beta_scheduler.step()
            else:
                kl_betas[key] = 0.0

        # free nats clamping
        free_nats_ratios = {}
        clamped_kl_divergences = []
        for i, kl_divergence in enumerate(kl_divergences):
            if kl_divergence is not None:
                free_nats = self.free_nats_per_dim[f"scale_{i}"]
                free_nats_ratios[f"free_nats_ratio_scale_{i}"] = (
                    kl_divergence < free_nats
                ).sum() / kl_divergence.numel()
                kl_divergence: torch.Tensor = floor_softplus_clamp(kl_divergence, free_nats).mean(dim=0).sum()
                clamped_kl_divergences.append(kl_divergence)
            else:
                clamped_kl_divergences.append(None)

        # Scale all kl divergences
        kl_losses = {}
        for i, kl_divergence in enumerate(clamped_kl_divergences):
            if kl_divergence is not None:
                kl_beta = kl_betas[f"scale_{i}"]
                kl_losses[f"kl_loss_scale_{i}"] = kl_beta * kl_divergence

        # logging
        if self.training:
            with torch.no_grad():
                log_dict = (
                    {f"train_schedulers/kl_beta_{key}": kl_beta for key, kl_beta in kl_betas.items()}
                    | {f"train_kl_step/{k}": v for k, v in free_nats_ratios.items()}
                    | {
                        f"train_kl_step/kl_per_dim_scale_{i}": kl_divergence.mean()
                        for i, kl_divergence in enumerate(kl_divergences)
                        if kl_divergence is not None
                    }
                )
                for i in range(len(posterior_distributions)):
                    posterior_mu, posterior_sigma = posterior_distributions[i]
                    prior_mu, prior_sigma = prior_distributions[i]
                    if posterior_mu is not None:
                        log_dict[f"train_dist_step/posterior_mu{i}"] = posterior_mu.mean()
                        log_dict[f"train_dist_step/posterior_sigma{i}"] = posterior_sigma.mean()
                        if prior_mu is not None:
                            log_dict[f"train_dist_step/prior_mu{i}"] = prior_mu.mean()
                            log_dict[f"train_dist_step/prior_sigma{i}"] = prior_sigma.mean()

                            d_mu = posterior_mu - prior_mu
                            d_sigma = posterior_sigma / prior_sigma
                            log_dict[f"train_dist_step/d_mu{i}"] = d_mu.mean()
                            log_dict[f"train_dist_step/d_sigma{i}"] = d_sigma.mean()
                self.log_dict(log_dict, sync_dist=True, on_step=True, on_epoch=False)

        return kl_losses

    def calculate_psnr(self, reconstructed, x):
        return self.psnr_metric(reconstructed, x).mean()

    def calculate_ms_ssim(self, reconstructed, x):
        return self.ms_ssim_metric(reconstructed, x).mean()

    def calculate_adv_losses(self, reconstructed, x):
        if not self.discriminator_scheduler.is_ready():
            try:
                steps_per_epoch = self.get_steps_per_epoch()
            except RuntimeError:
                steps_per_epoch = 100
            discriminator_annealing_epochs = self.training_config.discriminator_annealing_wavelength
            discriminator_annealing_steps = discriminator_annealing_epochs * steps_per_epoch
            self.discriminator_scheduler.set_num_annealing_steps(
                2 * discriminator_annealing_steps // 3, discriminator_annealing_steps // 3, 0, 0
            )

        adv_beta = self.discriminator_scheduler.get()
        if self.training:
            self.discriminator_scheduler.step()

        try:
            optimizer_main, optimizer_disc = self.optimizers()
        except:  # not attached to trainer
            return {}

        self.toggle_optimizer(optimizer_main)
        gen_fool_disc_loss = adv_beta * self.adv_loss(
            self.discriminator(reconstructed)[-1],
            target_is_real=True,
            for_discriminator=False,
        )
        self.untoggle_optimizer(optimizer_main)

        self.toggle_optimizer(optimizer_disc)
        disc_catch_gen_loss = self.adv_loss(
            self.discriminator(reconstructed.detach())[-1],
            target_is_real=False,
            for_discriminator=True,
        )
        disc_identify_real_loss = self.adv_loss(
            self.discriminator(x)[-1],
            target_is_real=True,
            for_discriminator=True,
        )
        self.untoggle_optimizer(optimizer_disc)

        # logging
        if self.training:
            with torch.no_grad():
                self.log(f"train_schedulers/adv_beta", adv_beta, sync_dist=True, on_step=True, on_epoch=False)

        return {
            "gen_fool_disc_loss": gen_fool_disc_loss,
            "disc_catch_gen_loss": disc_catch_gen_loss,
            "disc_identify_real_loss": disc_identify_real_loss,
        }

    def calculate_basic_losses(self, x, reconstructed, kl_divergences, prior_distributions, posterior_distributions):
        return (
            {
                "reconstruction_loss": self.calculate_reconstruction_loss(reconstructed, x),
                "perceptual_loss": self.calculate_perceptual_loss(reconstructed, x),
                "ms_ssim_loss": self.calculate_ms_ssim_loss(reconstructed, x),
                # "spectral_loss": self.calculate_spectral_loss(z_mu, z_sigma),
            }
            | self.calculate_kl_losses(kl_divergences, prior_distributions, posterior_distributions)
            | self.calculate_adv_losses(reconstructed, x)
        )

    def calculate_aurs(self, kl_divergences):
        aurs = {}
        for i, kl_divergence in enumerate(kl_divergences):
            if kl_divergence is None:
                continue
            aurs[f"aur_scale_{i}"] = (kl_divergence > self.training_config.aur_threshold_per_dim).float().mean()
        return aurs

    @torch.no_grad()
    def calculate_metrics(self, x, reconstructed, kl_divergences):
        metrics = {
            "psnr": self.calculate_psnr(reconstructed, x),
            "ms_ssim": self.calculate_ms_ssim(reconstructed, x),
        } | self.calculate_aurs(kl_divergences)
        return metrics

    def calculate_autoencoder_loss(self, all_losses):
        all_losses["autoencoder_loss"] = sum(
            self.training_config.loss_weights[key] * all_losses[key]
            for key in all_losses
            if not key.startswith("disc_")
        )

    def calculate_discriminator_loss(self, all_losses):
        all_losses["discriminator_loss"] = sum(
            self.training_config.loss_weights[key] * all_losses[key] for key in all_losses if key.startswith("disc_")
        )

    def process_step(self, batch, prefix, batch_idx=-1):
        x = batch["image"]
        # crop_offsets = batch["crop_offset"]

        autoencoder_output = self(x)
        reconstructed = autoencoder_output["reconstructed"]
        kl_divergences = autoencoder_output["kl_divergences"]
        prior_distributions = autoencoder_output["prior_distributions"]
        posterior_distributions = autoencoder_output["posterior_distributions"]

        all_losses = self.calculate_basic_losses(
            x, reconstructed, kl_divergences, prior_distributions, posterior_distributions
        )
        all_metrics = self.calculate_metrics(x, reconstructed, kl_divergences)

        self.calculate_autoencoder_loss(all_losses)
        self.calculate_discriminator_loss(all_losses)

        # Log
        with torch.no_grad():
            step_log = {}
            epoch_log = {}

            for key, value in all_losses.items():
                scaled_value = self.training_config.loss_weights.get(key, 1.0) * value
                if prefix == "train":
                    step_log[f"train_step/{key}"] = float(value)
                    step_log[f"train_step_scaled/{key}"] = float(scaled_value)
                epoch_log[f"{prefix}_epoch_loss/{key}"] = float(value)
                epoch_log[f"{prefix}_epoch_loss_scaled/{key}"] = float(scaled_value)

            for key, value in all_metrics.items():
                epoch_log[f"{prefix}_epoch_metrics/{key}"] = float(value)

            try:
                self.log_dict(step_log, sync_dist=True, on_step=True, on_epoch=False)
            except:
                print(f"Error in logging steps {step_log}")

            try:
                self.log_dict(epoch_log, sync_dist=True, on_step=False, on_epoch=True)
            except:
                print(f"Error in logging epochs {epoch_log}")

        return {
            "all_losses": all_losses,
            "all_metrics": all_metrics,
            "reconstructed": reconstructed,
            "kl_divergences": kl_divergences,
            "prior_distributions": prior_distributions,
            "posterior_distributions": posterior_distributions,
            "step_log": step_log,
            "epoch_log": epoch_log,
        }

    def training_step(self, batch, batch_idx):
        return_value = self.process_step(batch, "train", batch_idx)
        all_losses = return_value["all_losses"]

        # Manual optimization
        optimizer_main, optimizer_disc = self.optimizers()
        scheduler_main, scheduler_disc = self.lr_schedulers()

        # Accumulate gradients
        self.manual_backward(all_losses["autoencoder_loss"])
        self.manual_backward(all_losses["discriminator_loss"])

        # Update weights if it's time
        update_main = (batch_idx + 1) % self.training_config.accumulate_grad_batches == 0
        update_disc = (batch_idx + 1) % (
            self.training_config.accumulate_grad_batches * self.training_config.train_discriminator_every_gen_steps
        ) == 0

        if update_main:
            # VAE
            self.clip_gradients(
                optimizer_main, gradient_clip_val=self.training_config.gradient_clip_val, gradient_clip_algorithm="norm"
            )
            optimizer_main.step()
            self.on_before_zero_grad(optimizer_main)
            optimizer_main.zero_grad(set_to_none=True)
            scheduler_main.step()

        if update_disc:
            # Discriminator
            self.clip_gradients(
                optimizer_disc, gradient_clip_val=self.training_config.gradient_clip_val, gradient_clip_algorithm="norm"
            )
            optimizer_disc.step()
            self.on_before_zero_grad(optimizer_disc)
            optimizer_disc.zero_grad(set_to_none=True)
            scheduler_disc.step()

    def validation_step(self, batch, batch_idx):
        return self.process_step(batch, "val", batch_idx)

    def on_train_epoch_end(self):
        self.print_log()

    def configure_optimizers(self):
        all_params = set(filter(lambda p: p.requires_grad, self.parameters()))
        disc_params = set(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
        main_params = all_params - disc_params

        optimizer_main = torch.optim.Adam(main_params, lr=self.training_config.lr)
        optimizer_disc = torch.optim.Adam(disc_params, lr=2 * self.training_config.lr, betas=(0.5, 0.999))

        total_steps = self.get_total_steps()
        scheduler_main = ConstantLRWithWarmup(optimizer_main, max(1, total_steps // 10))
        scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc, T_max=total_steps, eta_min=1e-6)

        return [
            {"optimizer": optimizer_main, "lr_scheduler": {"scheduler": scheduler_main, "interval": "step"}},
            {"optimizer": optimizer_disc, "lr_scheduler": {"scheduler": scheduler_disc, "interval": "step"}},
        ]

    def forward(self, x):
        o = self.autoencoder(x)
        return o

    def get_total_steps(self):
        # Divide by accumulate_grad_batches because of manual optimization
        return self.trainer.estimated_stepping_batches // self.training_config.accumulate_grad_batches

    def get_steps_per_epoch(self):
        return self.trainer.estimated_stepping_batches * self.trainer.accumulate_grad_batches // self.trainer.max_epochs


if __name__ == "__main__":
    from config import get_config

    config = get_config()

    device = torch.device("cpu")
    # device = torch.device("cuda:0")

    autoencoder = NVAELightning(config.model, config.training).to(device)

    sample_input = {
        "image": torch.zeros(1, 1, *config.image_size, device=device),
        # "crop_offset": torch.zeros(1, 3, device=device),
    }
    sample_output = autoencoder.process_step(sample_input, "train", 0)

    # print("Input shape: ", sample_input["image"].shape)
    # print()
    # # print("Output:", *[f"{key}:\n{value}\n" for key, value in sample_output.items()], sep="\n")
    # print(sample_output["all_losses"], sample_output["all_metrics"])
