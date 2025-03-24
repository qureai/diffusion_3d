import torch
from munch import munchify
from vision_architectures.nets.perceiver_3d import Perceiver3DConfig

from diffusion_3d.constants import SERVER_MAPPING
from diffusion_3d.utils.environment import set_multi_node_environment


def get_config(training_image_size=(64, 64, 64)):
    # minimum_input_size = (32, 128, 128)
    model_config = {
        "maisi": {
            "spatial_dims": 3,
            "in_channels": 1,
            "latent_channels": 192,
            "out_channels": 1,
            "num_res_blocks": [2, 2, 2],
            "num_channels": [64, 128, 256],
            "norm_num_groups": 32,
            "norm_eps": 1e-06,
            "attention_levels": [False, False, False],
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "use_checkpointing": False,
            "use_convtranspose": False,
            "norm_float16": False,
            "num_splits": 4,
            "dim_split": 0,
            "save_mem": False,
        }
    }
    model_config["adaptor"] = Perceiver3DConfig.model_validate(
        {
            "dim": 192,
            "latent_grid_size": (16, 16, 16),
            "num_heads": 16,
            "attn_drop_prob": 0.1,
            "proj_drop_prob": 0.1,
            "mlp_drop_prob": 0.1,
            "encode": {
                "num_layers": 2,
            },
            "process": {
                "num_layers": 4,
            },
            "decode": {
                "num_layers": 2,
                "out_channels": model_config["maisi"]["latent_channels"],
            },
        }
    )

    latent_patch_size = [2 ** (len(model_config["maisi"]["num_res_blocks"]) - 1)] * 3
    # # latent_patch_size = (64, 64, 64)
    training_latent_grid_size = tuple(size // patch for size, patch in zip(training_image_size, latent_patch_size))
    compression_factor = tuple(training_image_size[i] // training_latent_grid_size[i] for i in range(3))

    batch_size = 8
    num_train_samples_per_datapoint = 4
    num_val_samples_per_datapoint = batch_size

    transformsd_keys = ["image"]

    clipping_transform = {
        "_target_": "monai.transforms.ScaleIntensityRanged",
        "keys": transformsd_keys,
        "a_min": 0.0,
        "a_max": 1.0,
        "b_min": 0.0,
        "b_max": 1.0,
        "clip": True,
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
            allowed_shapes=((64, -1), (256, -1), (256, -1)),
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
                    # {
                    #     "_target_": "vision_architectures.transforms.spatial.ResizedWithCropTrackingd",
                    #     "keys": transformsd_keys,
                    #     "original_shape_key": "Shape",
                    #     "spatial_size": (-1, 256, 256),
                    #     "mode": "trilinear",
                    #     "anti_aliasing": True,
                    # },
                    {
                        "_target_": "monai.transforms.ScaleIntensityRanged",  # Windowing
                        "keys": transformsd_keys,
                        "a_min": -1000,
                        "a_max": 1000,
                        "b_min": 0.0,
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
                    #         {  # Scale
                    #             "_target_": "monai.transforms.RandZoomd",
                    #             "keys": transformsd_keys,
                    #             "min_zoom": 0.5,
                    #             "max_zoom": 1.1,
                    #             "prob": 0.9,
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
                                            "prob": 0.75,
                                        }
                                    ),
                                    compose_with_clipping_tranform(
                                        {
                                            "_target_": "monai.transforms.RandGaussianSmoothd",
                                            "keys": transformsd_keys,
                                            "prob": 0.75,
                                        }
                                    ),
                                    compose_with_clipping_tranform(
                                        {
                                            "_target_": "monai.transforms.RandGaussianSharpend",
                                            "keys": transformsd_keys,
                                            "prob": 0.75,
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
                    # {
                    #     "_target_": "vision_architectures.transforms.spatial.ResizedWithCropTrackingd",
                    #     "keys": transformsd_keys,
                    #     "original_shape_key": "Shape",
                    #     "spatial_size": (-1, 256, 256),
                    #     "mode": "trilinear",
                    #     "anti_aliasing": True,
                    # },
                    {
                        "_target_": "monai.transforms.ScaleIntensityRanged",  # Windowing
                        "keys": transformsd_keys,
                        "a_min": -1000,
                        "a_max": 1000,
                        "b_min": 0.0,
                        "b_max": 1.0,
                        "clip": True,
                    },
                    # {
                    #     "_target_": "monai.transforms.SpatialPadd",
                    #     "keys": transformsd_keys,
                    #     "spatial_size": minimum_input_size,
                    #     "mode": "constant",
                    #     "value": -1,
                    # },
                    # {
                    #     "_target_": "monai.transforms.DivisiblePadd",
                    #     "keys": transformsd_keys,
                    #     "k": latent_patch_size,
                    #     "mode": "constant",
                    #     "value": -1,
                    # },
                    {
                        "_target_": "vision_architectures.transforms.croppad.RandSpatialCropSamplesWithCropTrackingd",
                        "keys": transformsd_keys,
                        "roi_size": training_image_size,
                        "num_samples": num_val_samples_per_datapoint,
                    },
                    clipping_transform,
                ],
            },
            # test_augmentations={
            #     "_target_": "monai.transforms.Compose",
            #     "transforms": [
            #         {
            #             "_target_": "vision_architectures.transforms.croppad.CropForegroundWithCropTrackingd",
            #             "keys": transformsd_keys,
            #             "source_key": transformsd_keys[0],
            #             "allow_smaller": True,
            #         },
            #         {
            #             "_target_": "monai.transforms.ScaleIntensityRanged",  # Windowing
            #             "keys": transformsd_keys,
            #             "a_min": -1000,
            #             "a_max": 2000,
            #             "b_min": -1.0,
            #             "b_max": 1.0,
            #             "clip": True,
            #         },
            #         {
            #             "_target_": "monai.transforms.SpatialPadd",
            #             "keys": transformsd_keys,
            #             "spatial_size": minimum_input_size,
            #             "mode": "constant",
            #             "value": -1,
            #         },
            #         {
            #             "_target_": "monai.transforms.DivisiblePadd",
            #             "keys": transformsd_keys,
            #             "k": latent_patch_size,
            #             "mode": "constant",
            #             "value": -1,
            #         },
            #     ],
            # },
            #
            num_workers=16,
            train_batch_size=batch_size // num_train_samples_per_datapoint,
            val_batch_size=batch_size // num_val_samples_per_datapoint,
            train_sample_size=4_000,
            sample_balance_cols=["Source", "BodyPart"],
        )
    )

    training_config = munchify(
        dict(
            start_from_checkpoint=None,
            # start_from_checkpoint=r"/raid3/arjun/checkpoints/adaptive_autoencoder/v27__2025_03_07/version_0/checkpoints/last.ckpt",
            #
            max_epochs=200,
            lr=5e-4,
            seed=42,
            check_val_every_n_epoch=1,
            #
            loss_weights={
                "reconstruction_loss": 1.0,
                "perceiver_loss": 2.0,
                "perceptual_loss": 0.2,
                "ms_ssim_loss": 0.1,
                "kl_loss": 5e-6,
                # "spectral_loss": 1e-6,
            },
            kl_annealing_start_epoch=10,
            kl_annealing_epochs=50,
            # free_bits=1.0,
            #
            checkpointing_level=2,
            #
            fast_dev_run=20,
            strategy="ddp",
            #
            accumulate_grad_batches=10,
            gradient_clip_val=None,
        )
    )

    clearml_tags = [
        f"Training image size: {training_image_size}",
        f"Latent patch size: {latent_patch_size}",
        f"Latent grid size: {training_latent_grid_size}",
        f"Train batch size: {data_config.train_batch_size}",
        f"Compression: {compression_factor}",
        f"Checkpointing level: {training_config.checkpointing_level}",
        #
        "VAE",
        "Frozen MAISI",
        "Trainable perceiver",
        "Replaced compression and expansion layers in maisi with larger dim, trainable",
        "Removed quant layers in maisi",
        "Increased perceiver loss weight to 2.0",
        "Added position embeddings to perceiver input",
        "Added layernorms before and after perceiver",
        "Corrected relative position embeddings to all perceiver process layers",
        # "Added back gradient clipping",
    ]

    additional_config = munchify(
        dict(
            task_name="v40__2025_03_17",
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
    from hydra.utils import instantiate

    config = get_config()

    encoder = instantiate(config.model["encoder"])
    decoder = instantiate(config.model["decoder"])

    print(encoder)
    print(decoder)

    sample_input = torch.empty((1, 1, *config.image_size))
    sample_output = encoder(sample_input)
    print(sample_output.shape)
    sample_output = decoder(sample_output)
    print(sample_output.shape)

    # sample_input = torch.zeros((1, 100, 100, 100))
    # transforms = instantiate(config.data.toDict()["train_augmentations"])
    # print(transforms)
    # sample_output = transforms(sample_input)
    # print(sample_output.shape)
    # print(sample_output.shape)
