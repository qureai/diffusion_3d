import lightning as L
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from taming.modules.losses.lpips import LPIPS
from torch import nn


class AutoEncoderLoss(L.LightningModule):
    def __init__(self, num_channels):
        super().__init__()

        # Perceptual loss
        self.reconstruction_loss = nn.L1Loss()
        self.perceptual_loss = LPIPS().eval()

        self.adv_loss = nn.BCEWithLogitsLoss()
        self.kl_loss = nn.KLDivLoss()

        self.discriminator = NLayerDiscriminator(input_nc=num_channels, ndf=1).apply(weights_init)

    def get_reconstruction_loss(self, y_hat, y):
        reconstruction_loss = self.reconstruction_loss(y_hat, y)
        perceptual_loss = self.perceptual_loss(y_hat, y)
        return reconstruction_loss + perceptual_loss  # perceptual weight is 1

    def forward(self, y_hat, y):
        pass
