import torch
from munch import munchify
from vision_architectures.nets.perceiver_3d import Perceiver3DConfig
from vision_architectures.nets.swinv2_3d import SwinV23DConfig, SwinV23DDecoderConfig

from diffusion_3d.constants import SERVER_MAPPING
from diffusion_3d.utils.environment import set_multi_node_environment


def get_config():
    minimum_input_size = (32, 128, 128)
    training_image_size = (64, 64, 64)
    window_sizes = [
        (4, 4, 4),
        (4, 4, 4),
        (4, 4, 4),
        # (6, 6, 6),
        (4, 4, 4),
    ]
    model_config = {
        "swin": SwinV23DConfig.model_validate(
            {
                "patch_size": (2, 2, 2),
                "in_channels": 1,
                "dim": 24,
                "drop_prob": 0.1,
                "stages": [
                    {
                        "depth": 2,
                        "num_heads": 8,
                        "window_size": window_sizes[0],
                        "attn_drop_prob": 0.1,
                        "proj_drop_prob": 0.1,
                        "mlp_drop_prob": 0.1,
                        "max_attention_batch_size": 2**12,
                    },
                    {
                        "patch_merging": {
                            "merge_window_size": (2, 2, 2),
                            "out_dim_ratio": 2,
                        },
                        "depth": 2,
                        "num_heads": 8,
                        "window_size": window_sizes[1],
                        "attn_drop_prob": 0.1,
                        "proj_drop_prob": 0.1,
                        "mlp_drop_prob": 0.1,
                        "max_attention_batch_size": 2**12,
                    },
                    {
                        "patch_merging": {
                            "merge_window_size": (2, 2, 2),
                            "out_dim_ratio": 2,
                        },
                        "depth": 4,
                        "num_heads": 16,
                        "window_size": window_sizes[2],
                        "attn_drop_prob": 0.1,
                        "proj_drop_prob": 0.1,
                        "mlp_drop_prob": 0.1,
                        "max_attention_batch_size": 2**12,
                    },
                    # {
                    #     "patch_merging": {
                    #         "merge_window_size": (2, 2, 2),
                    #         "out_dim_ratio": 2,
                    #     },
                    #     "depth": 2,
                    #     "num_heads": 16,
                    #     "window_size": window_sizes[3],
                    #     "attn_drop_prob": 0.1,
                    #     "proj_drop_prob": 0.1,
                    #     "mlp_drop_prob": 0.1,
                    #     "max_attention_batch_size": 2**12,
                    # },
                ],
            }
        )
    }
    model_config["adaptor"] = Perceiver3DConfig.model_validate(
        {
            "encode": {
                "dim": model_config["swin"].stages[-1].out_dim,
                "num_latent_tokens": 1024,
                "num_layers": 2,
                "num_heads": 16,
                "attn_drop_prob": 0.1,
                "proj_drop_prob": 0.1,
                "mlp_drop_prob": 0.1,
            },
            "process": {
                "dim": model_config["swin"].stages[-1].out_dim,
                "num_layers": 4,
                "num_heads": 16,
                "attn_drop_prob": 0.1,
                "proj_drop_prob": 0.1,
                "mlp_drop_prob": 0.1,
            },
            "decode": {
                "dim": model_config["swin"].stages[-1].out_dim,
                "num_layers": 2,
                "num_heads": 16,
                "out_channels": model_config["swin"].stages[-1].out_dim,
                "attn_drop_prob": 0.1,
                "proj_drop_prob": 0.1,
                "mlp_drop_prob": 0.1,
            },
        }
    )
    model_config["decoder"] = SwinV23DDecoderConfig.model_validate(
        {
            "dim": model_config["swin"].stages[-1].out_dim,
            "drop_prob": 0.1,
            "stages": [
                # {
                #     "depth": 2,
                #     "num_heads": 16,
                #     "window_size": window_sizes[-1],
                #     "attn_drop_prob": 0.1,
                #     "proj_drop_prob": 0.1,
                #     "mlp_drop_prob": 0.1,
                #     "max_attention_batch_size": 2**12,
                #     "patch_splitting": {
                #         "final_window_size": (2, 2, 2),
                #         "out_dim_ratio": 2,
                #     },
                # },
                {
                    "depth": 4,
                    "num_heads": 16,
                    "window_size": window_sizes[-2],
                    "attn_drop_prob": 0.1,
                    "proj_drop_prob": 0.1,
                    "mlp_drop_prob": 0.1,
                    "max_attention_batch_size": 2**12,
                    "patch_splitting": {
                        "final_window_size": (2, 2, 2),
                        "out_dim_ratio": 2,
                    },
                },
                {
                    "depth": 2,
                    "num_heads": 8,
                    "window_size": window_sizes[-3],
                    "attn_drop_prob": 0.1,
                    "proj_drop_prob": 0.1,
                    "mlp_drop_prob": 0.1,
                    "max_attention_batch_size": 2**12,
                    "patch_splitting": {
                        "final_window_size": (2, 2, 2),
                        "out_dim_ratio": 2,
                    },
                },
                {
                    "depth": 2,
                    "num_heads": 4,
                    "window_size": window_sizes[-4],
                    "attn_drop_prob": 0.1,
                    "proj_drop_prob": 0.1,
                    "mlp_drop_prob": 0.1,
                    "max_attention_batch_size": 2**12,
                },
            ],
        }
    )
    model_config["unembedding"] = {
        "num_upsamples": 1,
        "upsample_channels": [
            model_config["decoder"].stages[-1].out_dim // 2,
        ],
        "in_channels": model_config["decoder"].stages[-1].out_dim,
        "out_channels": model_config["swin"].in_channels,
    }
    model_config["pathway_drop_prob"] = 0.4

    latent_patch_size = model_config["swin"].patch_size
    for stage in model_config["swin"].stages:
        latent_patch_size = stage.get_out_patch_size(latent_patch_size)
    # latent_patch_size = (64, 64, 64)
    training_latent_grid_size = tuple(size // patch for size, patch in zip(training_image_size, latent_patch_size))
    compression_factor = tuple(training_image_size[i] // training_latent_grid_size[i] for i in range(3))

    batch_size = 60
    num_train_samples_per_datapoint = 5
    num_val_samples_per_datapoint = 20

    transformsd_keys = ["image"]

    clipping_transform = {
        "_target_": "monai.transforms.ScaleIntensityRanged",
        "keys": transformsd_keys,
        "a_min": -1.0,
        "a_max": 1.0,
        "b_min": -1.0,
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
            allowed_shapes=((96, -1), (256, -1), (256, -1)),
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
                        "a_max": 2000,
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
                        "a_max": 2000,
                        "b_min": -1.0,
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
                        "a_min": -1000,
                        "a_max": 2000,
                        "b_min": -1.0,
                        "b_max": 1.0,
                        "clip": True,
                    },
                    {
                        "_target_": "monai.transforms.SpatialPadd",
                        "keys": transformsd_keys,
                        "spatial_size": minimum_input_size,
                        "mode": "constant",
                        "value": -1,
                    },
                    {
                        "_target_": "monai.transforms.DivisiblePadd",
                        "keys": transformsd_keys,
                        "k": latent_patch_size,
                        "mode": "constant",
                        "value": -1,
                    },
                ],
            },
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
                "perceptual_loss": 0.2,
                "ms_ssim_loss": 0.1,
                "kl_loss": 5e-6,
                # "spectral_loss": 1e-6,
            },
            kl_annealing_start_epoch=15,
            kl_annealing_epochs=30,
            # free_bits=1.0,
            #
            residual_connection_epochs=25,
            #
            checkpointing_level=2,
            #
            fast_dev_run=False,
            strategy="ddp",
            #
            accumulate_grad_batches=5,
            gradient_clip_val=1.0,
        )
    )

    patch_sizes = [model_config["swin"].patch_size]
    for i in range(len(model_config["swin"].stages)):
        patch_sizes.append(model_config["swin"].stages[i].get_out_patch_size(patch_sizes[-1]))
    grid_sizes = []
    for i in range(len(model_config["swin"].stages) + 1):
        grid_sizes.append(tuple([size // patch for size, patch in zip(training_image_size, patch_sizes[i])]))
    # Ensure grid size can be divided by window size
    for i in range(len(model_config["swin"].stages)):
        assert all(
            [grid % window == 0 for grid, window in zip(grid_sizes[i], model_config["swin"].stages[i].window_size)]
        ), f"{grid_sizes[i]} is not divisible by {model_config['swin'].stages[i].window_size}"
    clearml_tags = [
        f"Training image size: {training_image_size}",
        f"Patch sizes: {patch_sizes}",
        f"Grid sizes: {grid_sizes}",
        f"Dimensions: {[stage.out_dim for stage in model_config['swin'].stages]}",
        f"Train batch size: {data_config.train_batch_size}",
        f"Compression: {compression_factor}",
        f"Checkpointing level: {training_config.checkpointing_level}",
        #
        "VAE",
        "Added back the residual connection",
        "Reduced training size to 64",
        "Reduced the depth of swin",
        # "Resized input images to (256, 256) before cropping",
    ]

    additional_config = munchify(
        dict(
            task_name="v33__2025_03_11",
            log_on_clearml=True,
            clearml_project="adaptive_autoencoder",
            clearml_tags=clearml_tags,
        )
    )

    distributed_config = munchify(
        dict(
            distributed=False,
            nodes=[
                (node, SERVER_MAPPING[node])
                for node in [
                    "qrnd10.internal.qure.ai",  # First one is master node
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
    from hydra.utils import instantiate

    config = get_config()

    sample_input = torch.zeros((1, 100, 100, 100))
    transforms = instantiate(config.data.toDict()["train_augmentations"])
    print(transforms)
    sample_output = transforms(sample_input)
    print(sample_output.shape)
