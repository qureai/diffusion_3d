import torch
from arjcode.model import MyLightningModule, freeze_module, freeze_modules, unfreeze_module
from diffusion_3d.chestct.diffusion.unet.nn import Diffusion3D
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.metrics import MultiScaleSSIMMetric, PSNRMetric
from monai.networks.nets import PatchDiscriminator
from munch import Munch
from torch import nn
from torch.nn import L1Loss, MSELoss
from vision_architectures.schedulers.cyclic import CyclicAnnealingScheduler
from vision_architectures.schedulers.lrs import ConstantLRWithWarmup
from vision_architectures.schedulers.noise import CosineNoiseScheduler
from vision_architectures.utils.clamping import floor_softplus_clamp
from vision_architectures.utils.timesteps import TimestepSampler


class Diffusion3DLightning(MyLightningModule):
    def __init__(self, model_config: dict, training_config: Munch):
        super().__init__(
            # identify_unused_parameters=True,
            # print_small_gradient_norms=True,
            # print_large_gradient_norms=True,
        )
        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config

        self.denoiser = Diffusion3D(model_config, training_config.checkpointing_level)

        self.noise_scheduler = CosineNoiseScheduler(model_config.timesteps)
        self.timestep_sampler = TimestepSampler(model_config.timesteps, "uniform")

        self.l1_loss = L1Loss()
        self.l2_loss = MSELoss()

        self.cosine_similarity_metric = nn.CosineSimilarity(dim=1)
        self.psnr_metric = PSNRMetric(max_val=2.0)
        self.ms_ssim_metric = MultiScaleSSIMMetric(spatial_dims=3, data_range=2.0, kernel_size=5)
        self.perceptual_loss = PerceptualLoss(
            spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
        )
        freeze_module(self.perceptual_loss)

    def calculate_reconstruction_loss(self, noise_pred, noise):
        return {
            "l1_loss": self.l1_loss(noise_pred, noise),
            "l2_loss": self.l2_loss(noise_pred, noise),
        }

    def calculate_perceptual_loss(self, xt_minus_1_hat, xt_minus_1):
        # prediction = self.noise_scheduler.add_noise(x, timesteps - 1, noise_pred)
        # target = self.noise_scheduler.add_noise(x, timesteps - 1, noise)
        return self.perceptual_loss(xt_minus_1_hat, xt_minus_1).clamp(min=0.0)

    def calculate_psnr(self, xt_minus_1_hat, xt_minus_1):
        return self.psnr_metric(xt_minus_1_hat, xt_minus_1).mean()

    def calculate_ms_ssim(self, xt_minus_1_hat, xt_minus_1):
        return self.ms_ssim_metric(xt_minus_1_hat, xt_minus_1).mean()

    def calculate_basic_losses(self, noise_pred, noise):
        return self.calculate_reconstruction_loss(noise_pred, noise)

    @torch.no_grad()
    def calculate_train_metrics(self, noise_pred, noise):
        return {
            "cosine_similarity": self.cosine_similarity_metric(noise_pred, noise).mean(),
        }

    @torch.no_grad()
    def calculate_val_metrics(self, x0_hat, x0):
        metrics = {
            "perceptual_loss": self.calculate_perceptual_loss(x0_hat, x0),
            "psnr": self.calculate_psnr(x0_hat, x0),
            "ms_ssim": self.calculate_ms_ssim(x0_hat, x0),
        }
        return metrics

    def calculate_denoiser_loss(self, all_losses):
        all_losses["denoiser_loss"] = sum(
            self.training_config.loss_weights[key] * all_losses[key] for key in all_losses
        )

    def process_batch(self, batch):
        x0 = batch["image"]
        spacings = torch.stack(batch["Spacing"], dim=1)
        # crop_offsets = batch["crop_offset"]
        return x0, spacings

    def process_training_step(self, batch, batch_idx=-1):
        x0, spacings = self.process_batch(batch)

        batch_size = x0.shape[0]

        timesteps = self.timestep_sampler(batch_size).to(x0.device)
        noise = torch.randn_like(x0)
        guidance = torch.ones((2, batch_size)).to(x0)

        xt = self.noise_scheduler.add_noise(x0, timesteps, noise)

        noise_pred = self(xt, timesteps, spacings, guidance)

        all_losses = self.calculate_basic_losses(noise_pred, noise)
        all_metrics = self.calculate_train_metrics(noise_pred, noise)

        self.calculate_denoiser_loss(all_losses)

        # Log
        with torch.no_grad():
            step_log = {}
            epoch_log = {}

            for key, value in all_losses.items():
                scaled_value = self.training_config.loss_weights.get(key, 1.0) * value
                step_log[f"train_step_loss/{key}"] = float(value)
                step_log[f"train_step_loss_scaled/{key}"] = float(scaled_value)
                epoch_log[f"train_epoch_loss/{key}"] = float(value)
                epoch_log[f"train_epoch_loss_scaled/{key}"] = float(scaled_value)

            for key, value in all_metrics.items():
                step_log[f"train_step_metrics/{key}"] = float(value)
                epoch_log[f"train_epoch_metrics/{key}"] = float(value)

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
            "step_log": step_log,
            "epoch_log": epoch_log,
        }

    def training_step(self, batch, batch_idx):
        return_value = self.process_training_step(batch, batch_idx)
        all_losses = return_value["all_losses"]
        return all_losses["denoiser_loss"]

    def process_validation_step(self, batch, batch_idx=-1):
        x0, spacings = self.process_batch(batch)

        batch_size = x0.shape[0]

        timesteps = torch.full((batch_size,), self.training_config.val_timesteps + 1, device=x0.device)
        noise = torch.randn_like(x0)
        guidance = torch.ones((2, batch_size)).to(x0)

        xt = self.noise_scheduler.add_noise(x0, timesteps, noise)

        for t in range(self.training_config.val_timesteps + 1, 0, -self.training_config.val_ddim_skip_steps):
            timesteps = torch.full((batch_size,), t, device=x0.device)
            noise_pred = self(xt, timesteps, spacings, guidance)
            _, xt_minus_1 = self.noise_scheduler.remove_noise(
                xt, noise_pred, timesteps, eta=self.training_config.val_ddim_eta
            )

            xt = xt_minus_1

        assert t == 1, f"t={t} should be 1, but got {t}"
        all_metrics = self.calculate_val_metrics(xt, x0)

        # Log
        epoch_log = {}
        for key, value in all_metrics.items():
            epoch_log[f"val_epoch_metrics/{key}"] = float(value)

        try:
            self.log_dict(epoch_log, sync_dist=True, on_step=False, on_epoch=True)
        except:
            print(f"Error in logging epochs {epoch_log}")

        return {
            "x0": x0,
            "x0_hat": xt,
            "all_metrics": all_metrics,
            "epoch_log": epoch_log,
        }

    def validation_step(self, batch, batch_idx):
        return self.process_validation_step(batch, batch_idx)

    def on_train_epoch_end(self):
        self.print_log()

    def configure_optimizers(self):
        all_params = set(filter(lambda p: p.requires_grad, self.parameters()))

        optimizer = torch.optim.Adam(all_params, lr=self.training_config.lr)

        total_steps = self.get_total_steps()
        scheduler = ConstantLRWithWarmup(optimizer, max(1, total_steps // 10))

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def forward(self, xt, t, spacings, guidance):
        o = self.denoiser(xt, t, spacings, guidance)
        return o

    def get_total_steps(self):
        return self.trainer.estimated_stepping_batches


if __name__ == "__main__":
    from config import get_config

    config = get_config()
    config.input_size = (32, 32, 32)
    config.training.val_timesteps = 1

    device = torch.device("cpu")
    # device = torch.device("cuda:0")

    autoencoder = Diffusion3DLightning(config.model, config.training).to(device)

    sample_input = {
        "image": torch.zeros(1, 1, *config.input_size, device=device),
        "Spacing": [torch.tensor([0.0])] * 3,
        # "crop_offset": torch.zeros(1, 3, device=device),
    }
    sample_output = autoencoder.process_training_step(sample_input)

    print("Input shape: ", sample_input["image"].shape)
    print()
    print(sample_output["all_losses"], sample_output["all_metrics"])

    sample_output = autoencoder.process_validation_step(sample_input)

    print("Input shape: ", sample_input["image"].shape)
    print()
    print(sample_output["all_metrics"])
