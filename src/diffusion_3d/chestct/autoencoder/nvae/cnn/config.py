import torch
from munch import munchify

from diffusion_3d.constants import SERVER_MAPPING
from diffusion_3d.utils.environment import set_multi_node_environment


def get_config(training_image_size=(128, 128, 128)):
    model_config = munchify(
        {
            "in_channels": 1,
            "num_channels": [16, 32, 64, 128, 256],
            "depths": [4, 4, 4, 4, 4],
            "latent_dims": [None, None, 4, None, 16],
            "kernel_size": 3,
            "normalization": "groupnorm",
            "normalization_pre_args": [4],
            "activation": "silu",
            "survival_prob": 1.0,
            "latent": {
                "kernel_size": 3,
                "normalization": "groupnorm",
                "normalization_pre_args_list": [None, None, 2, None, 4],
                "activation": "silu",
            },
        }
    )

    # 64x compression, 64 latent, 512 input, 1024 intermediate, effective 4096x compression
    # 32x compression, 32 latent, 256 input, 512 intermediate, effective 1024x compression
    # 16x compression, 16 latent, 128 input, 256 intermediate, effective 256x compression
    # 8x compression, 8 latent, 64 input, 128 intermediate, effective 64x compression
    # 4x compression, 4 latent, 32 input, 64 intermediate, effective 16x compression
    # 2x compression, 2 latent, 16 input, 32 intermediate, effective 4x compression
    # 1x compression, No latent, Not trained separately, 16 intermediate

    batch_size = 8
    num_train_samples_per_datapoint = 4
    num_val_samples_per_datapoint = batch_size

    transformsd_keys = ["image"]

    clipping_transform = {
        "_target_": "vision_architectures.transforms.clipping.Clipd",
        "keys": transformsd_keys,
        "min_value": -1.0,
        "max_value": 1.0,
    }

    def compose_with_clipping_tranform(transforms):
        if isinstance(transforms, dict):
            transforms = [transforms]
        return {
            "_target_": "monai.transforms.Compose",
            "transforms": [*transforms, clipping_transform],
        }

    data_config = munchify(
        dict(
            csvpath=r"/raid3/arjun/ct_pretraining/csvs/sources.csv",
            datapath=r"/raid3/arjun/ct_pretraining/scans/",
            checkpointspath=r"/raid3/arjun/checkpoints/adaptive_autoencoder/",
            #
            limited_dataset_size=None,
            #
            allowed_spacings=((0.4, 7), (-1, -1), (-1, -1)),
            allowed_shapes=((training_image_size[0], -1), (256, -1), (256, -1)),
            #
            train_augmentations={
                "_target_": "monai.transforms.Compose",
                "transforms": [
                    {
                        "_target_": "vision_architectures.transforms.croppad.CropForegroundWithCropTrackingd",
                        "keys": transformsd_keys,
                        "source_key": transformsd_keys[0],
                        "allow_smaller": True,
                    },
                    {
                        "_target_": "monai.transforms.ScaleIntensityRanged",  # Windowing
                        "keys": transformsd_keys,
                        "a_min": -1500,
                        "a_max": 1500,
                        "b_min": -1.0,
                        "b_max": 1.0,
                        "clip": True,
                    },
                    # {
                    #     "_target_": "monai.transforms.RandomOrder",
                    #     "transforms": [
                    #         {  # Rotate
                    #             "_target_": "monai.transforms.RandRotated",
                    #             "keys": transformsd_keys,
                    #             "range_x": 20,
                    #             "prob": 1.0,
                    #         },
                    #         {  # Shear
                    #             "_target_": "monai.transforms.RandAffined",
                    #             "keys": transformsd_keys,
                    #             "shear_range": (0, 0, -0.35, 0.35, -0.35, 0.35),
                    #             "prob": 0.2,
                    #         },
                    #     ],
                    # },
                    {
                        "_target_": "vision_architectures.transforms.croppad.RandSpatialCropSamplesWithCropTrackingd",
                        "keys": transformsd_keys,
                        "roi_size": tuple(int(size * 0.8) for size in training_image_size),
                        "max_roi_size": tuple(int(size * 1.2) for size in training_image_size),
                        "random_size": True,
                        "num_samples": num_train_samples_per_datapoint,
                    },
                    {
                        "_target_": "vision_architectures.transforms.spatial.ResizedWithCropTrackingd",
                        "keys": transformsd_keys,
                        "original_shape_key": "Shape",
                        "spatial_size": training_image_size,
                        "mode": "trilinear",
                        "anti_aliasing": True,
                    },
                    clipping_transform,
                    {
                        "_target_": "monai.transforms.RandomOrder",
                        "transforms": [
                            {
                                "_target_": "vision_architectures.transforms.spatial.RandFlipWithCropTrackingd",
                                "keys": transformsd_keys,
                                "original_shape_key": "Shape",
                                "prob": 0.5,
                            },
                            compose_with_clipping_tranform(
                                {
                                    "_target_": "monai.transforms.RandKSpaceSpikeNoised",
                                    "keys": transformsd_keys,
                                    "prob": 0.1,
                                }
                            ),
                            {
                                "_target_": "monai.transforms.OneOf",
                                "transforms": [
                                    compose_with_clipping_tranform(
                                        {
                                            "_target_": "monai.transforms.RandGaussianNoised",
                                            "keys": transformsd_keys,
                                            "prob": 0.5,
                                            "std": 0.03,
                                        }
                                    ),
                                    compose_with_clipping_tranform(
                                        {
                                            "_target_": "monai.transforms.RandGaussianSmoothd",
                                            "keys": transformsd_keys,
                                            "prob": 0.5,
                                        }
                                    ),
                                    compose_with_clipping_tranform(
                                        {
                                            "_target_": "monai.transforms.RandGaussianSharpend",
                                            "keys": transformsd_keys,
                                            "prob": 0.5,
                                        }
                                    ),
                                ],
                            },
                        ],
                    },
                    clipping_transform,
                ],
            },
            val_augmentations={
                "_target_": "monai.transforms.Compose",
                "transforms": [
                    {
                        "_target_": "vision_architectures.transforms.croppad.CropForegroundWithCropTrackingd",
                        "keys": transformsd_keys,
                        "source_key": transformsd_keys[0],
                        "allow_smaller": True,
                    },
                    {
                        "_target_": "monai.transforms.ScaleIntensityRanged",  # Windowing
                        "keys": transformsd_keys,
                        "a_min": -1500,
                        "a_max": 1500,
                        "b_min": -1.0,
                        "b_max": 1.0,
                        "clip": True,
                    },
                    {
                        "_target_": "vision_architectures.transforms.croppad.RandSpatialCropSamplesWithCropTrackingd",
                        "keys": transformsd_keys,
                        "roi_size": training_image_size,
                        "num_samples": num_val_samples_per_datapoint,
                    },
                    {  # In case image gets cropped to smaller size than required size
                        "_target_": "vision_architectures.transforms.spatial.ResizedWithCropTrackingd",
                        "keys": transformsd_keys,
                        "original_shape_key": "Shape",
                        "spatial_size": training_image_size,
                        "mode": "trilinear",
                        "anti_aliasing": True,
                    },
                    clipping_transform,
                ],
            },
            test_augmentations={
                "_target_": "monai.transforms.Compose",
                "transforms": [
                    {
                        "_target_": "vision_architectures.transforms.croppad.CropForegroundWithCropTrackingd",
                        "keys": transformsd_keys,
                        "source_key": transformsd_keys[0],
                        "allow_smaller": True,
                    },
                    {
                        "_target_": "monai.transforms.ScaleIntensityRanged",  # Windowing
                        "keys": transformsd_keys,
                        "a_min": -1500,
                        "a_max": 1500,
                        "b_min": -1.0,
                        "b_max": 1.0,
                        "clip": True,
                    },
                    {
                        "_target_": "monai.transforms.CenterSpatialCropd",
                        "keys": transformsd_keys,
                        "roi_size": training_image_size,
                    },
                    {  # In case image gets cropped to smaller size than required size
                        "_target_": "vision_architectures.transforms.spatial.ResizedWithCropTrackingd",
                        "keys": transformsd_keys,
                        "original_shape_key": "Shape",
                        "spatial_size": training_image_size,
                        "mode": "trilinear",
                        "anti_aliasing": True,
                    },
                    clipping_transform,
                ],
            },
            num_workers=12,
            train_batch_size=batch_size // num_train_samples_per_datapoint,
            val_batch_size=batch_size // num_val_samples_per_datapoint,
            test_batch_size=4,
            train_sample_size=4_800,
            sample_balance_cols=["Source", "BodyPart"],
        )
    )

    training_config = munchify(
        dict(
            # start_from_checkpoint=None,
            start_from_checkpoint=r"/raid3/arjun/checkpoints/adaptive_autoencoder/v69__2025_04_23__epoch330/version_0/checkpoints/last.ckpt",
            #
            max_epochs=1000,
            lr=1e-4,
            seed=42,
            check_val_every_n_epoch=1,
            #
            loss_weights={
                "reconstruction_loss": 0.9,
                "perceptual_loss": 0.3,
                "ms_ssim_loss": 0.1,
                "gen_fool_disc_loss": 0.15,
                #
                "kl_loss_scale_2": 1e-6,
                "kl_loss_scale_4": 1e-5,
                #
                "disc_catch_gen_loss": 0.5,
                "disc_identify_real_loss": 0.5,
            },
            kl_annealing={
                "scale_2": {
                    "start_epoch": 0,
                    "wavelength": 30,
                },
                "scale_4": {
                    "start_epoch": 0,
                    "wavelength": 20,
                },
            },
            free_nats_per_dim={
                "scale_2": 0.05,
                "scale_4": 0.05,
            },
            aur_threshold_per_dim=0.05,
            #
            discriminator_annealing_wavelength=25,
            train_discriminator_every_gen_steps=2,
            #
            checkpointing_level=0,
            freeze_scales=[0, 1, 2],
            #
            fast_dev_run=False,
            strategy="ddp_find_unused_parameters_true",
            #
            accumulate_grad_batches=5,
            log_every_n_steps=1,
            gradient_clip_val=5.0,
        )
    )

    compression_factor = 2 ** (len(model_config.depths) - 1)

    clearml_tags = [
        f"Training image size: {training_image_size}",
        f"Dimensions: {model_config.num_channels}",
        f"Latent dimensions: {model_config.latent_dims}",
        f"Train batch size: {data_config.train_batch_size}",
        f"Compression: {compression_factor}",
        f"Checkpointing level: {training_config.checkpointing_level}",
        f"Frozen scales: {training_config.freeze_scales}",
        #
        "NVAE",
        "ms_ssim kernel = 7",
        "Training next scale i.e. 16x compression",
        "Accumulate 5 batches",
        "Added back gaussian sharpening augmentation",
        "Note: first conv layer of decoder2 is trainable (as new input is comming from scale3)",
    ]

    additional_config = munchify(
        dict(
            task_name="v70__2025_04_28__4xv69",
            log_on_clearml=True,
            clearml_project="adaptive_autoencoder",
            clearml_tags=clearml_tags,
        )
    )

    distributed_config = munchify(
        dict(
            distributed=True,
            nodes=[
                (node, SERVER_MAPPING[node])
                for node in [
                    "qrnd10.internal.qure.ai",  # First one is master node
                    "qrnd21.l40.sr.internal.qure.ai",
                    "qrnd8.internal.qure.ai",
                ]
            ],
        )
    )
    if training_config.fast_dev_run:
        distributed_config.distributed = False

    if distributed_config.distributed:
        distributed_environment_dict = set_multi_node_environment(distributed_config.nodes)
        distributed_config.update(distributed_environment_dict)

    config = munchify(
        dict(
            data=data_config,
            model=model_config,
            training=training_config,
            distributed=distributed_config,
            additional=additional_config,
            #
            image_size=training_image_size,
        )
    )

    return config


if __name__ == "__main__":
    from pprint import pprint

    from hydra.utils import instantiate

    config = get_config()
    pprint(config.toDict())

    sample_input = torch.zeros((1, 100, 100, 100))
    transforms = instantiate(config.data.toDict()["train_augmentations"])
    print(transforms)
    sample_output = transforms(sample_input)
    print(sample_output.shape)
