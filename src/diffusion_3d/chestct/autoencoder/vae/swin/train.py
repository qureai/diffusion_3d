import lightning as L
import torch
from clearml import Task
from config import get_config
from einops import rearrange
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from munch import munchify
from neuro_utils.describe import describe_model

from diffusion_3d.chestct.autoencoder.vae.swin.model import AdaptiveVAELightning
from diffusion_3d.datasets.ct_rate import CTRATEDataModule

torch.set_float32_matmul_precision("medium")


# Load config
config = get_config()


# Seed everything
L.seed_everything(config.training.seed, workers=True)


# Start ClearML task if applicable
task = None
if not config.training.fast_dev_run and config.additional.log_on_clearml:
    task: Task = Task.init(
        project_name=config.additional.clearml_project,
        task_name=config.additional.task_name,
    )
    task.connect_configuration(config, name="config")
    task.add_tags(config.additional.clearml_tags)


# Create model and datamodule
if config.training.start_from_checkpoint is not None:
    model = AdaptiveVAELightning.load_from_checkpoint(
        config.training.start_from_checkpoint,
        model_config=config.model,
        training_config=config.training,
        strict=False,
        map_location="cpu",
    )
    print(f"Started from: {config.training.start_from_checkpoint}")
else:
    model = AdaptiveVAELightning(config.model, config.training)
datamodule = CTRATEDataModule(config.data)


# Add model size tags to clearml
if task is not None:
    encoder_params = sum(p.numel() for p in model.autoencoder.encoder.parameters())
    # perceiver_params = sum(p.numel() for p in model.autoencoder.adapt.parameters()) + sum(
    #     p.numel() for p in model.autoencoder.unadapt.parameters()
    # )
    decoder_params = sum(p.numel() for p in model.autoencoder.decoder.parameters())
    total_params = sum(p.numel() for p in model.autoencoder.parameters())
    any_other_params = total_params - (encoder_params + decoder_params)
    task.add_tags(
        [
            f"Encoder: {encoder_params:,} params",
            # f"Perceiver: {perceiver_params:,} params",
            f"Decoder: {decoder_params:,} params",
            f"Other: {any_other_params:,} params",
            f"Total: {total_params:,} params",
        ]
    )


# Determine some trainer arguments
checkpoint_callback = ModelCheckpoint(
    filename="{epoch}",
    monitor="val_epoch_loss/autoencoder_loss",
    mode="min",
    save_last=True,
    save_top_k=1,
)
lr_callback = LearningRateMonitor("step")
callbacks = [checkpoint_callback, lr_callback]

logger = TensorBoardLogger(config.data.checkpointspath, name=config.additional.task_name)


devices = "auto"
plugins = None
num_nodes = 1
if config.distributed.distributed:
    devices = config.distributed.devices
    plugins = [config.distributed.cluster_environment]
    num_nodes = len(config.distributed.nodes)


# Create trainer object
trainer = L.Trainer(
    precision="16-mixed",
    max_epochs=config.training.max_epochs,
    logger=logger,
    enable_checkpointing=logger is not None,
    callbacks=callbacks,
    deterministic=False,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=config.training.check_val_every_n_epoch,
    fast_dev_run=config.training.fast_dev_run,
    log_every_n_steps=1,
    accumulate_grad_batches=config.training.accumulate_grad_batches if not config.training.fast_dev_run else 1,
    strategy=config.training.strategy,
    gradient_clip_val=config.training.gradient_clip_val,
    devices=devices,
    plugins=plugins,
    num_nodes=num_nodes,
    profiler="simple" if config.training.fast_dev_run else None,
)


# Print some useful numbers
if trainer.global_rank == 0:
    print("Tags:")
    for tag in config.additional.clearml_tags:
        print(" ", tag)
    print()


# Train
trainer.fit(model, datamodule)
