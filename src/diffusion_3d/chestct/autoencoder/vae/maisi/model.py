import math

import torch
from arjcode.model import MyLightningModule
from monai.losses.perceptual import PerceptualLoss
from monai.metrics import MultiScaleSSIMMetric, PSNRMetric
from munch import Munch
from torch.nn import L1Loss

from diffusion_3d.chestct.autoencoder.vae.maisi.nn import AdaptiveVAE


class AdaptiveVAELightning(MyLightningModule):
    def __init__(self, model_config: dict, training_config: Munch):
        super().__init__()
        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config

        self.autoencoder = AdaptiveVAE(model_config, training_config.checkpointing_level)

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

        # self.kl_beta_scheduler = SigmoidScheduler()

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

    def calculate_kl_loss(self, z_mu, z_sigma):
        z_mu_squared = z_mu.pow(2)
        z_sigma_squared = z_sigma.pow(2)
        kl_loss: torch.Tensor = 0.5 * (z_mu_squared + z_sigma_squared - torch.log(z_sigma_squared + 1e-8) - 1)

        # free_bits_ratio = (kl_loss < self.free_bits_per_channel).sum() / kl_loss.numel()
        # kl_loss = kl_loss.clamp(min=self.free_bits_per_channel)

        # additional plotting
        self.log_dict(
            {
                "train_kl_step/mu": z_mu_squared.mean(),
                "train_kl_step/sigma": z_sigma_squared.mean(),
                # "train_kl_step/free_bits_ratio": free_bits_ratio,
            },
            sync_dist=True,
        )

        return kl_loss.sum(dim=1).mean()

    def calculate_spectral_loss(self, mu, logvar, eps=1e-6, normalize_by_dim=True, spatial_average=True):
        b, dim, z, y, x = mu.shape
        num_voxels = z * y * x

        # Flatten spatial dimensions
        mu_flat = mu.reshape(b, dim, num_voxels)  # (b, dim, num_voxels)
        logvar_flat = logvar.reshape(b, dim, num_voxels)  # (b, dim, num_voxels)

        # Compute empirical covariance matrix from mu across all samples and spatial locations
        # We'll compute this separately for each batch and average to reduce batch dependency
        batch_cov_matrices = []
        for i in range(b):
            # Treat each spatial location as an independent sample (dim, num_voxels)
            mu_batch = mu_flat[i].permute(1, 0)  # (num_voxels, dim)

            # Center the data for this batch
            mu_centered = mu_batch - mu_batch.mean(dim=0, keepdim=True)

            # Scale by 1/(N-1) for unbiased estimation when sample size is large
            cov_factor = 1.0 / max(num_voxels - 1, 1)
            batch_cov = cov_factor * mu_centered.T @ mu_centered  # (dim, dim)
            batch_cov_matrices.append(batch_cov)

        # Average covariance matrices across batches
        cov_mu = torch.stack(batch_cov_matrices).mean(dim=0)  # (dim, dim)

        # Properly incorporate the variance from logvar
        # First average variances per batch, then across batches
        var_flat = torch.exp(logvar_flat)  # (b, dim, num_voxels)

        # Average variance across spatial dimensions for each latent dimension and batch
        batch_var_diags = var_flat.mean(dim=2)  # (b, dim)

        # Then average across batches
        var_diag = batch_var_diags.mean(dim=0)  # (dim,)

        # Add diagonal variance to covariance matrix
        cov_matrix = cov_mu + torch.diag(var_diag)  # (dim, dim)

        # Add small epsilon to diagonal for numerical stability
        # This is more robust than simple addition since it scales with eigenvalue magnitudes
        diag_eps = eps * (1.0 + torch.diag(cov_matrix).abs())
        cov_matrix = cov_matrix + torch.diag(diag_eps)

        # Compute eigenvalues with robust approach
        try:
            eigvals = torch.linalg.eigvalsh(cov_matrix)  # (dim,)
            # Apply safe clamping with a scaled minimum value
            min_eig = eps * (1.0 + eigvals.abs().max().item())
            eigvals = torch.clamp(eigvals, min=min_eig)
        except RuntimeError:
            # Fallback for numerical instability: add larger epsilon and retry
            cov_matrix = cov_matrix + torch.eye(dim, device=cov_matrix.device) * (eps * 10.0)
            eigvals = torch.linalg.eigvalsh(cov_matrix)
            min_eig = eps * 10.0 * (1.0 + eigvals.abs().max().item())
            eigvals = torch.clamp(eigvals, min=min_eig)

        # Spectral KL loss: 0.5 * sum(λᵢ - 1 - log(λᵢ))
        per_dim_loss = 0.5 * (eigvals - 1.0 - torch.log(eigvals + eps))

        # Apply normalization based on latent dimension to ensure consistent scaling
        if normalize_by_dim:
            spectral_loss = per_dim_loss.sum() / dim
        else:
            spectral_loss = per_dim_loss.sum()

        # Further scale by spatial dimensions if requested
        # This makes the loss invariant to the number of spatial points
        if spatial_average:
            # Apply a scaling factor that's inversely proportional to num_voxels
            # This keeps the loss magnitude consistent regardless of spatial resolution
            # We use sqrt because covariance already has a quadratic relationship with sample count
            spectral_loss = spectral_loss * (100.0 / max(math.sqrt(num_voxels), 1.0))

        # Participation Ratio: (sum(λᵢ))² / sum(λᵢ²)
        # Measures effective dimensionality of the latent representation
        participation_ratio = (eigvals.sum() ** 2) / (eigvals.pow(2).sum() + eps)
        scaled_participation_ratio = participation_ratio / dim
        self.log("train_kl_step/participation_ratio", participation_ratio, sync_dist=True)
        self.log("train_kl_step/scaled_participation_ratio", scaled_participation_ratio, sync_dist=True)

        return spectral_loss

    def calculate_psnr(self, reconstructed, x):
        return self.psnr_metric(reconstructed, x).mean()

    def calculate_ms_ssim(self, reconstructed, x):
        return self.ms_ssim_metric(reconstructed, x).mean()

    def calculate_basic_losses(self, x, output):
        reconstructed = output["reconstructed"]
        # z_mu = output["encoded_mu"]
        # z_sigma = output["encoded_sigma"]
        perceiver_in = output["encoded"]
        perceiver_out = output["unadapted"]

        self.log_dict(
            {
                "train_perceiver/in_mean": perceiver_in.mean(),
                "train_perceiver/in_std": perceiver_in.std(),
                "train_perceiver/in_min": perceiver_in.min(),
                "train_perceiver/in_max": perceiver_in.max(),
                "train_perceiver/out_mean": perceiver_out.mean(),
                "train_perceiver/out_std": perceiver_out.std(),
                "train_perceiver/out_min": perceiver_out.min(),
                "train_perceiver/out_max": perceiver_out.max(),
            },
            sync_dist=True,
            on_step=True,
            on_epoch=False,
        )

        return {
            "reconstruction_loss": self.calculate_reconstruction_loss(reconstructed, x),
            "perceiver_loss": self.calculate_reconstruction_loss(perceiver_out, perceiver_in),
            "perceptual_loss": self.calculate_perceptual_loss(reconstructed, x),
            "ms_ssim_loss": self.calculate_ms_ssim_loss(reconstructed, x),
            # "kl_loss": self.calculate_kl_loss(z_mu, z_sigma),
            # "spectral_loss": self.calculate_spectral_loss(z_mu, z_sigma),
        }

    def calculate_metrics(self, x, reconstructed):
        return {
            "psnr": self.calculate_psnr(reconstructed, x),
            "ms_ssim": self.calculate_ms_ssim(reconstructed, x),
        }

    def calculate_autoencoder_loss(self, all_losses):
        # # Apply beta annealing
        # if not self.kl_beta_scheduler.is_ready():
        #     steps_per_epoch = (
        #         self.trainer.estimated_stepping_batches
        #         * self.trainer.accumulate_grad_batches
        #         // self.trainer.max_epochs
        #     )
        #     kl_annealing_epochs = self.training_config.kl_annealing_epochs
        #     kl_annealing_steps = kl_annealing_epochs * steps_per_epoch
        #     self.kl_beta_scheduler.set_num_steps(kl_annealing_steps)

        # kl_beta = 0.0
        # if self.current_epoch >= self.training_config.kl_annealing_start_epoch:
        #     kl_beta = self.kl_beta_scheduler.get()
        #     if self.training:
        #         self.kl_beta_scheduler.step()
        # all_losses["kl_loss"] = kl_beta * all_losses["kl_loss"]
        # # all_losses["spectral_loss"] = kl_beta * all_losses["spectral_loss"]

        # if self.training:
        #     self.log("train_kl_step/beta", kl_beta, sync_dist=True, on_step=True, on_epoch=False)

        all_losses["autoencoder_loss"] = (
            self.training_config.loss_weights["reconstruction_loss"] * all_losses["reconstruction_loss"]
            + self.training_config.loss_weights["perceiver_loss"] * all_losses["perceiver_loss"]
            + self.training_config.loss_weights["perceptual_loss"] * all_losses["perceptual_loss"]
            + self.training_config.loss_weights["ms_ssim_loss"] * all_losses["ms_ssim_loss"]
            # + self.training_config.loss_weights["kl_loss"] * all_losses["kl_loss"]
            # + self.training_config.loss_weights["spectral_loss"] * all_losses["spectral_loss"]
        )

    def process_step(self, batch, prefix, batch_idx):
        x = batch["image"]
        crop_offsets = batch["crop_offset"]

        autoencoder_output = self(x, crop_offsets, prefix)
        all_losses = self.calculate_basic_losses(x, autoencoder_output)
        reconstructed = autoencoder_output["reconstructed"]
        all_metrics = self.calculate_metrics(x, reconstructed)

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
            # "encoded_mu": encoded_mu,
            # "encoded_sigma": encoded_sigma,
        }

    def training_step(self, batch, batch_idx):
        return self.process_step(batch, "train", batch_idx)["all_losses"]["autoencoder_loss"]

    def validation_step(self, batch, batch_idx):
        return self.process_step(batch, "val", batch_idx)

    def on_train_epoch_end(self):
        # self.process_epoch("train")
        self.print_log()

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

    def forward(self, x, crop_offsets, run_type="val"):
        # x: (b, d1, z1, y1, x1)
        # crop_offsets: (b, 3)

        return self.autoencoder(x, crop_offsets, run_type)


if __name__ == "__main__":
    from config import get_config

    config = get_config()

    autoencoder = AdaptiveVAELightning(config.model, config.training)

    sample_input = {
        "image": torch.zeros((1, 1, *config.image_size)),
        "crop_offset": torch.zeros(1, 3),
    }
    sample_output = autoencoder.process_step(sample_input, "train", 0)

    print("Input shape: ", sample_input["image"].shape)
    print()
    print("Output shapes:")
    for key, value in sample_output.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
