import lightning as L
import torch
from monai.losses.perceptual import PerceptualLoss
from monai.metrics import MultiScaleSSIMMetric, PSNRMetric
from munch import Munch
from prettytable import PrettyTable
from torch.nn import L1Loss

from diffusion_3d.chestct.autoencoder.ae.nn import AdaptiveAE


class AdaptiveAELightning(L.LightningModule):
    def __init__(self, model_config: dict, training_config: Munch):
        super().__init__()
        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config

        self.autoencoder = AdaptiveAE(model_config, training_config.checkpointing_level)

        self.reconstruction_loss = L1Loss()
        self.perceptual_loss = PerceptualLoss(
            spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
        ).eval()
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False

        # self.free_bits_per_channel = training_config.get("free_bits", 0) / model_config.swin.stages[-1].out_dim

        self.train_losses = []
        self.val_losses = []

        self.psnr_metric = PSNRMetric(max_val=2.0)
        self.ms_ssim_metric = MultiScaleSSIMMetric(spatial_dims=3, data_range=2.0, kernel_size=4)

        self.train_metrics = []
        self.val_metrics = []

    def calculate_reconstruction_loss(self, reconstructed, x):
        return self.reconstruction_loss(reconstructed, x)

    def calculate_perceptual_loss(self, reconstructed, x):
        return self.perceptual_loss(reconstructed.float(), x.float()).clamp(min=0.0)

    def calculate_tv_loss(self, reconstructed):
        # Calculate Total Variation loss across all three dimensions
        diff_i = torch.abs(reconstructed[:, :, :, :, :-1] - reconstructed[:, :, :, :, 1:])
        diff_j = torch.abs(reconstructed[:, :, :, :-1, :] - reconstructed[:, :, :, 1:, :])
        diff_k = torch.abs(reconstructed[:, :, :-1, :, :] - reconstructed[:, :, 1:, :, :])

        return torch.mean(diff_i) + torch.mean(diff_j) + torch.mean(diff_k)

    def calculate_ms_ssim_loss(self, reconstructed, x):
        return 1 - self.calculate_ms_ssim(reconstructed, x)

    def calculate_psnr(self, reconstructed, x):
        return self.psnr_metric(reconstructed, x).mean()

    def calculate_ms_ssim(self, reconstructed, x):
        return self.ms_ssim_metric(reconstructed, x).mean()

    def calculate_basic_losses(self, reconstructed, x):
        return {
            "reconstruction_loss": self.calculate_reconstruction_loss(reconstructed, x),
            "perceptual_loss": self.calculate_perceptual_loss(reconstructed, x),
            "ms_ssim_loss": self.calculate_ms_ssim_loss(reconstructed, x),
        }

    def calculate_metrics(self, reconstructed, x):
        return {
            "psnr": self.calculate_psnr(reconstructed, x),
            "ms_ssim": self.calculate_ms_ssim(reconstructed, x),
        }

    def calculate_autoencoder_loss(self, all_losses):
        all_losses["autoencoder_loss"] = (
            self.training_config.loss_weights["reconstruction_loss"] * all_losses["reconstruction_loss"]
            + self.training_config.loss_weights["perceptual_loss"] * all_losses["perceptual_loss"]
            + self.training_config.loss_weights["ms_ssim_loss"] * all_losses["ms_ssim_loss"]
        )

    def process_step(self, batch, prefix, batch_idx):
        x = batch["image"]
        crop_offsets = batch["crop_offset"]

        autoencoder_output = self(x, crop_offsets, prefix)
        reconstructed = autoencoder_output["reconstructed"]
        encoded = autoencoder_output["encoded"]

        all_losses = self.calculate_basic_losses(reconstructed, x)
        all_metrics = self.calculate_metrics(reconstructed, x)

        self.calculate_autoencoder_loss(all_losses)

        # Log
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
            "encoded": encoded,
        }

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
        return self.process_step(batch, "train", batch_idx)["all_losses"]["autoencoder_loss"]

    def validation_step(self, batch, batch_idx):
        return self.process_step(batch, "val", batch_idx)

    def on_train_epoch_end(self):
        # self.process_epoch("train")
        self.print_numbers()

    # def on_validation_epoch_end(self):
    #     self.process_epoch("val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.training_config.lr)

        total_steps = self.trainer.estimated_stepping_batches
        # scheduler = DecayingSineLR(optimizer, 1e-6, self.training_config.lr, total_steps // 4, 0.5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.training_config.lr,
            total_steps=total_steps,
            pct_start=0.1,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def print_numbers(self):
        if self.global_rank != 0:
            return

        numbers = self.trainer.logged_metrics
        numbers = list(numbers.items())
        for i in range(len(numbers)):
            numbers[i] = [*numbers[i][0].split("/"), round(float(numbers[i][1]), 5)]
        numbers = sorted(numbers)

        print()
        print()
        print("Epoch = {:<4}".format(self.current_epoch))
        table = PrettyTable(["Header", "Metric", "Value"])
        table.add_rows(numbers)
        print(table)
        print()

    def forward(self, x, crop_offsets, run_type="val"):
        # x: (b, d1, z1, y1, x1)
        # crop_offsets: (b, 3)

        residual_connection = self.autoencoder.residual_connection
        try:
            if self.training and not residual_connection.weight_scheduler.is_ready():
                steps_per_epoch = (
                    self.trainer.estimated_stepping_batches
                    * self.trainer.accumulate_grad_batches
                    // self.trainer.max_epochs
                )
                residual_connection_epochs = self.training_config.residual_connection_epochs
                residual_connection_steps = residual_connection_epochs * steps_per_epoch
                residual_connection.set_num_steps(residual_connection_steps)

            if run_type == "train":
                self.log(
                    "train_sigmoids/residual_connection_weight",
                    residual_connection.weight_scheduler.get(),
                    sync_dist=True,
                    on_step=True,
                    on_epoch=False,
                )
        except:  # Gives an error when called outside of training because accessing self.trainer
            residual_connection.set_num_steps(100)  # dummy number

        return self.autoencoder(x, crop_offsets, run_type)

    # def on_before_zero_grad(self, optimizer):
    #     """Will print all unused parameters"""
    #     if self.global_rank == 0:
    #         print("Zero grad params")
    #         for name, param in self.named_parameters():
    #             if param.requires_grad and param.grad is None:
    #                 print(name)
    #         print()


if __name__ == "__main__":
    from config import get_config

    config = get_config()

    autoencoder = AdaptiveAELightning(config.model, config.training)

    sample_input = {
        "image": torch.zeros((1, 1, *config.image_size)),
        "crop_offset": torch.zeros(1, 3),
    }
    sample_output = autoencoder.process_step(sample_input, "train", 0)

    print("Input shape: ", sample_input["image"].shape)
    print()
    print("Output:", *[f"{key}:\n{value}\n" for key, value in sample_output.items()], sep="\n")
