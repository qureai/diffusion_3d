import lightning as L
import numpy as np
import torch
from ct_pretraining.schedulers.decaying_sine import DecayingSineLR
from einops import rearrange, reduce
from monai.losses.perceptual import PerceptualLoss
from monai.metrics import MultiScaleSSIMMetric, PSNRMetric
from monai.networks.nets.autoencoderkl import AutoencoderKL, Decoder
from munch import Munch
from torch import nn
from torch.nn import L1Loss, MSELoss
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from vision_architectures.layers.embeddings import AbsolutePositionEmbeddings3D
from vision_architectures.nets.swinv2_3d import SwinV23DConfig, SwinV23DDecoder, SwinV23DModel


class UnembeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        unembedding_channels = config.in_channels // 2

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(config.in_channels, unembedding_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(3, unembedding_channels),  # More stable than InstanceNorm
            nn.SiLU(inplace=True),  # SiLU (Swish) activation for smoother gradients
        )

        # Frequency separation block
        # Low-frequency path with standard convolution
        self.low_freq_branch = nn.Sequential(
            spectral_norm(nn.Conv3d(unembedding_channels, unembedding_channels, kernel_size=5, padding=2, stride=1)),
            nn.GroupNorm(3, unembedding_channels),
            nn.SiLU(inplace=True),
        )
        # High-frequency path with depthwise separable convolution for efficiency
        self.high_freq_branch = nn.Sequential(
            spectral_norm(nn.Conv3d(unembedding_channels, unembedding_channels, kernel_size=1, padding=0, stride=1)),
            nn.GroupNorm(3, unembedding_channels),
            nn.SiLU(inplace=True),
            # Depthwise convolution for spatial feature extraction
            spectral_norm(
                nn.Conv3d(
                    unembedding_channels,
                    unembedding_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    groups=unembedding_channels,
                )
            ),
            nn.GroupNorm(3, unembedding_channels),
            nn.SiLU(inplace=True),
            # Pointwise convolution for feature mixing
            spectral_norm(nn.Conv3d(unembedding_channels, unembedding_channels, kernel_size=1, padding=0, stride=1)),
        )
        # Adaptive frequency gating mechanism
        self.fsb_gate = nn.Sequential(
            nn.Conv3d(unembedding_channels, unembedding_channels, kernel_size=1), nn.Sigmoid()
        )

        self.finalize = nn.Sequential(
            spectral_norm(nn.Conv3d(unembedding_channels, config.out_channels, kernel_size=3, padding=1)),
            nn.Tanh(),
        )

    def forward(self, x):
        # x: (b, 2d1, z, y, x)
        x = self.upsample(x)
        # (b, d1, 2z, 2y, 2x)

        low_freq = self.low_freq_branch(x)
        high_freq = self.high_freq_branch(x)
        gate = self.fsb_gate(x)
        x = low_freq + high_freq * gate
        # (b, d1, 2z, 2y, 2x)

        x = self.finalize(x)
        # (b, d2, 2z, 2y, 2x)

        return x


class AdaptiveVAE(AutoencoderKL, L.LightningModule):
    def __init__(self, model_config: dict, training_config: Munch):
        super().__init__(spatial_dims=3, latent_channels=model_config.swin.stages[-1]._out_dim)
        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config

        self.encoder = SwinV23DModel(model_config.swin)

        self.aggregator = nn.ModuleList(
            [
                nn.Conv3d(stage._out_dim, model_config.swin.stages[-1]._out_dim, kernel_size=1)
                for stage in model_config.swin.stages
            ]
        )

        if isinstance(model_config.decoder, SwinV23DConfig):
            self.decoder = SwinV23DDecoder(model_config.decoder)
        else:
            self.decoder = Decoder(**model_config.decoder)

        self.unembedding = UnembeddingLayer(model_config.unembedding)

        self.reconstruction_loss = L1Loss()
        self.perceptual_loss = PerceptualLoss(
            spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
        ).eval()
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False

        self.train_losses = []
        self.val_losses = []

        self.psnr_metric = PSNRMetric(max_val=2.0)
        self.ms_ssim_metric = MultiScaleSSIMMetric(spatial_dims=3, data_range=2.0, kernel_size=7)

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

    def calculate_kl_loss(self, z_mu, z_sigma):
        z_mu_squared = z_mu.pow(2)
        z_sigma_squared = z_sigma.pow(2)
        kl_loss = 0.5 * (z_mu_squared + z_sigma_squared - torch.log(z_sigma.pow(2)) - 1 + 1e-8)

        # additional plotting
        self.log_dict(
            {
                "train_kl_step/mu": z_mu_squared.mean(),
                "train_kl_step/sigma": z_sigma_squared.mean(),
            },
            sync_dist=True,
        )

        return kl_loss.sum(dim=1).mean()

    def calculate_psnr(self, reconstructed, x):
        return self.psnr_metric(reconstructed, x).mean()

    def calculate_ms_ssim(self, reconstructed, x):
        return self.ms_ssim_metric(reconstructed, x).mean()

    def calculate_basic_losses(self, x, reconstructed, z_mu, z_sigma):
        return {
            "reconstruction_loss": self.calculate_reconstruction_loss(reconstructed, x),
            "perceptual_loss": self.calculate_perceptual_loss(reconstructed, x),
            # "tv_loss": self.calculate_tv_loss(reconstructed),
            "ms_ssim_loss": self.calculate_ms_ssim_loss(reconstructed, x),
            "kl_loss": self.calculate_kl_loss(z_mu, z_sigma),
        }

    def calculate_metrics(self, x, reconstructed):
        return {
            "psnr": self.calculate_psnr(reconstructed, x),
            "ms_ssim": self.calculate_ms_ssim(reconstructed, x),
        }

    def process_step(self, batch, losses: list, metrics: list, prefix, batch_idx):
        x = batch

        reconstructed, decoded, adapted_encoded, encoded_mu, encoded_sigma = self(x, prefix)

        # Reshape back to input shape
        # reconstructed = F.interpolate(reconstructed, batch.shape[2:], mode="trilinear")

        all_losses = self.calculate_basic_losses(x, reconstructed, encoded_mu, encoded_sigma)
        all_metrics = self.calculate_metrics(x, reconstructed)

        kl_beta = min(1, (1 + self.current_epoch) / self.training_config.kl_annealing_epochs)
        all_losses["kl_loss"] = kl_beta * all_losses["kl_loss"]

        all_losses["autoencoder_loss"] = (
            self.training_config.loss_weights["reconstruction_loss"] * all_losses["reconstruction_loss"]
            + self.training_config.loss_weights["ms_ssim_loss"] * all_losses["ms_ssim_loss"]
            # + self.training_config.loss_weights["tv_loss"] * all_losses["tv_loss"]
            + self.training_config.loss_weights["kl_loss"] * all_losses["kl_loss"]
        )

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
        metrics.append({key: value.detach().cpu() for key, value in all_metrics.items()})

        return {
            "all_losses": all_losses,
            "all_metrics": all_metrics,
            "reconstructed": reconstructed,
            "decoded": decoded,
            "adapted_encoded": adapted_encoded,
            "encoded_mu": encoded_mu,
            "encoded_sigma": encoded_sigma,
        }

    def process_epoch(self, losses, metrics, prefix):
        losses = {key: [d[key].item() for d in losses] for key in losses[0]}
        losses = {key: np.mean(value).item() for key, value in losses.items()}
        for key, loss in losses.items():
            try:
                self.log(f"{prefix}_epoch_loss/{key}", float(loss), sync_dist=True)
                self.log(
                    f"{prefix}_epoch_loss_scaled/{key}",
                    float(self.training_config.loss_weights.get(key, 1.0) * loss),
                    sync_dist=True,
                )
            except:
                print(f"Error in logging losses {loss}, {type(loss)}")

        metrics = {key: [d[key].item() for d in metrics] for key in metrics[0]}
        metrics = {key: np.mean(value).item() for key, value in metrics.items()}
        for key, metric in metrics.items():
            try:
                self.log(f"{prefix}_epoch_metrics/{key}", float(metric), sync_dist=True)
            except:
                print(f"Error in logging metrics {metric}, {type(metric)}")

        self.print_numbers(f"{prefix}", losses | metrics)

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
        return self.process_step(batch["scan"], self.train_losses, self.train_metrics, "train", batch_idx)[
            "all_losses"
        ]["autoencoder_loss"]

    def validation_step(self, batch, batch_idx):
        return self.process_step(batch["scan"], self.val_losses, self.val_metrics, "val", batch_idx)

    def on_train_epoch_end(self):
        self.process_epoch(self.train_losses, self.train_metrics, "train")
        self.train_losses.clear()

    def on_validation_epoch_end(self):
        self.process_epoch(self.val_losses, self.val_metrics, "val")
        self.val_losses.clear()

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

    def print_numbers(self, prefix, numbers):
        if self.global_rank != 0:
            return

        print()
        print()
        print("{}: Epoch = {:<4}".format(prefix, self.current_epoch))
        for key, number in numbers.items():
            print("{} = {:.5f}".format(key.ljust(20), number))
        print()

    def encode(self, x: torch.Tensor):
        encoded, stage_outputs, _ = self.encoder(x)

        latent_shape = encoded.shape[2:]
        aggregated_latents = []
        for i, stage_output in enumerate(stage_outputs):
            aggregated_latents.append(self.aggregator[i](F.adaptive_avg_pool3d(stage_output, latent_shape)))
        encoded = reduce(aggregated_latents, "s ... -> ...", "sum")

        adapted = encoded

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

    def forward(self, x, run_type="val"):
        # x: (b, d1, z1, y1, x1)

        # encoded, _, _ = self.encoder(x)
        # decoded = self.decode(encoded)
        # reconstructed = self.unembedding(decoded)
        # return reconstructed, decoded, encoded, None, None

        encoded_mu, encoded_sigma = self.encode(x)
        if run_type == "train":
            adapted_encoded = self.sampling(encoded_mu, encoded_sigma)
        else:
            adapted_encoded = encoded_mu
        # (b, d2, z2, y2, x2)

        decoded = self.decode(adapted_encoded)
        # (b, d3, z3, y3, x3)

        reconstructed = self.unembedding(decoded)
        # (b, d1, z1, y1, x1)

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
    from config import get_config
    from neuro_utils.describe import describe_model

    config = get_config()

    device = torch.device("cpu")
    # device = torch.device("cuda:0")

    autoencoder = AdaptiveVAE(config.model, config.training).to(device)
    print("Encoder:")
    describe_model(autoencoder.encoder)
    # print("Adaptor:")
    # describe_model(autoencoder.adaptor)
    print("Aggregator:")
    describe_model(autoencoder.aggregator)
    print("Decoder:")
    describe_model(autoencoder.decoder)
    print("Final layer:")
    describe_model(autoencoder.unembedding)

    autoencoder.train()

    # Track memory before execution
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    initial_mem = process.memory_info().rss  # in bytes

    sample_input = torch.zeros((1, 1, *config.image_size)).to(device)
    sample_output = autoencoder(sample_input, "train")

    final_mem = process.memory_info().rss  # in bytes

    print(sample_input.shape)
    # print([x.shape for x in sample_output])
    print(f"GPU: {torch.cuda.max_memory_allocated() / 2**30} GB peak mem used")
    print(f"RAM: {(final_mem - initial_mem) / 2**30} GB peak mem used")

    # losses = autoencoder.training_step({"scan": sample_input, "spacing": ...}, 0)
    # from pprint import pprint
    # pprint(losses)
