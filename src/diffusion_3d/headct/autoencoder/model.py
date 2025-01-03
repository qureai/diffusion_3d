from typing import Literal

import lightning as L
from vision_architectures.blocks.heads_3d import SegmentationHead3D
from vision_architectures.nets.swinv2_3d import SwinV23DDecoder, SwinV23DModel

from diffusion_3d.headct.autoencoder.loss import AutoEncoderLoss


class AutoEncoder(L.LightningModule):
    def __init__(self, run_type: Literal["train", "freeze", "predict"], decoder_config, encoder_config=None):
        super().__init__()

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.run_type = run_type

        self.loss = None
        self.encoder = None
        self.decoder = None
        self.reconstruction_head = None
        self.discriminator = None

        # If train: encoder, decoder+reconstruction_head, loss
        # If freeze: encoder, decoder+reconstruction_head
        # If predict: decoder+reconstruction_head

        if run_type in {"train", "freeze"}:
            self.encoder = SwinV23DModel(encoder_config)
            if run_type == "train":
                self.loss = AutoEncoderLoss()
            else:  # in freeze mode
                self.encoder.eval()
        else:
            assert encoder_config is None, "Encoder should be None for prediction"

        self.decoder = SwinV23DDecoder(decoder_config)
        self.reconstruction_head = SegmentationHead3D(decoder_config["stages"][-1]["_out_dim"], 1)
        if run_type in {"freeze", "predict"}:
            self.decoder.eval()

    def encode(self, x):
        # x: (b, c1, z1, y1, x1)

        assert self.run_type in {"train", "freeze"}

        embedding, _, _ = self.encoder(x)
        # (b, c2, z2, y2, x2)

        return embedding

    def decode(self, embedding):
        # embedding: (b, c2, z2, y2, x2)

        hidden_states = self.decoder(embedding)
        # (b, c1, z1, y1, x1)

        reconstruction = self.reconstruction_head(hidden_states)
        # (b, c1, z1, y1, x1)

        return reconstruction

    def forward(self, x, y):
        # y: (b, c1, z1, y1, x1)
        #
        # If train or freeze:
        # x: (b, c1, z1, y1, x1)
        # If predict:
        # x: (b, c2, z2, y2, x2)

        if self.run_type in {"train", "freeze"}:
            x = self.encode(x)

        y_hat = self.decode(x)

        return_value = {"y_hat": y_hat}

        if self.run_type == "train":
            loss = self.loss(y_hat, y)
            return_value["loss"] = loss

        return return_value


if __name__ == "__main__":
    from config import get_config

    sample_config = get_config().model

    model = AutoEncoder("train", sample_config.decoder, sample_config.encoder)
    # model = AutoEncoder("freeze", sample_config.decoder, sample_config.encoder)
    # model = AutoEncoder("predict", sample_config.decoder, None)

    print(model)
    print(f"{sum(p.numel() for p in model.parameters()):,d} parameters")
