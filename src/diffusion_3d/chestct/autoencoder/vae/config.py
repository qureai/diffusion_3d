import torch
from munch import munchify
from vision_architectures.nets.perceiver_3d import Perceiver3DConfig
from vision_architectures.nets.swinv2_3d import SwinV23DConfig, SwinV23DDecoderConfig

from diffusion_3d.constants import SERVER_MAPPING
from diffusion_3d.utils.environment import set_multi_node_environment


def get_config():
    training_image_size = (96, 96, 96)
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
                        "window_size": (4, 4, 4),
                        "attn_drop_prob": 0.1,
                        "proj_drop_prob": 0.1,
                        "mlp_drop_prob": 0.1,
                    },
                    {
                        "patch_merging": {
                            "merge_window_size": (2, 2, 2),
                            "out_dim_ratio": 2,
                        },
                        "depth": 2,
                        "num_heads": 8,
                        "window_size": (4, 4, 4),
                        "attn_drop_prob": 0.1,
                        "proj_drop_prob": 0.1,
                        "mlp_drop_prob": 0.1,
                    },
                    {
                        "patch_merging": {
                            "merge_window_size": (2, 2, 2),
                            "out_dim_ratio": 2,
                        },
                        "depth": 4,
                        "num_heads": 16,
                        "window_size": (4, 4, 4),
                        "attn_drop_prob": 0.1,
                        "proj_drop_prob": 0.1,
                        "mlp_drop_prob": 0.1,
                    },
                    {
                        "patch_merging": {
                            "merge_window_size": (2, 2, 2),
                            "out_dim_ratio": 2,
                        },
                        "depth": 2,
                        "num_heads": 16,
                        "window_size": (6, 6, 6),
                        "attn_drop_prob": 0.1,
                        "proj_drop_prob": 0.1,
                        "mlp_drop_prob": 0.1,
                    },
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
                {
                    "depth": 2,
                    "num_heads": 16,
                    "window_size": (6, 6, 6),
                    "attn_drop_prob": 0.1,
                    "proj_drop_prob": 0.1,
                    "mlp_drop_prob": 0.1,
                    "patch_splitting": {
                        "final_window_size": (2, 2, 2),
                        "out_dim_ratio": 2,
                    },
                },
                {
                    "depth": 4,
                    "num_heads": 16,
                    "window_size": (4, 4, 4),
                    "attn_drop_prob": 0.1,
                    "proj_drop_prob": 0.1,
                    "mlp_drop_prob": 0.1,
                    "patch_splitting": {
                        "final_window_size": (2, 2, 2),
                        "out_dim_ratio": 2,
                    },
                },
                {
                    "depth": 2,
                    "num_heads": 8,
                    "window_size": (4, 4, 4),
                    "attn_drop_prob": 0.1,
                    "proj_drop_prob": 0.1,
                    "mlp_drop_prob": 0.1,
                    "patch_splitting": {
                        "final_window_size": (2, 2, 2),
                        "out_dim_ratio": 2,
                    },
                },
                {
                    "depth": 2,
                    "num_heads": 4,
                    "window_size": (4, 4, 4),
                    "attn_drop_prob": 0.1,
                    "proj_drop_prob": 0.1,
                    "mlp_drop_prob": 0.1,
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

    training_latent_shape = model_config["swin"].patch_size
    for stage in model_config["swin"].stages:
        training_latent_shape = stage.get_out_patch_size(training_latent_shape)
    compression_factor = tuple(training_image_size[i] / training_latent_shape[i] for i in range(3))

    transformsd_keys = ["image"]

    batch_size = 20
    num_train_samples_per_datapoint = 5
    num_val_samples_per_datapoint = batch_size

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
                        "_target_": "monai.transforms.CropForegroundd",
                        "keys": transformsd_keys,
                        "source_key": transformsd_keys[0],
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
                        "_target_": "monai.transforms.RandSpatialCropSamplesd",
                        "keys": transformsd_keys,
                        "roi_size": training_image_size,
                        "max_roi_size": tuple(int(size * 2.0) for size in training_image_size),
                        "random_size": True,
                        "num_samples": num_train_samples_per_datapoint,
                    },
                    {
                        "_target_": "monai.transforms.Resized",
                        "keys": transformsd_keys,
                        "spatial_size": training_image_size,
                        "mode": "trilinear",
                        "anti_aliasing": True,
                    },
                    {
                        "_target_": "monai.transforms.RandomOrder",
                        "transforms": [
                            {
                                "_target_": "monai.transforms.RandFlipd",
                                "keys": transformsd_keys,
                                "prob": 0.5,
                            },
                            {
                                "_target_": "monai.transforms.RandKSpaceSpikeNoised",
                                "keys": transformsd_keys,
                                "intensity_range": (12, 18),
                                "prob": 0.1,
                            },
                            {
                                "_target_": "monai.transforms.OneOf",
                                "transforms": [
                                    {
                                        "_target_": "monai.transforms.RandGaussianNoised",
                                        "keys": transformsd_keys,
                                        "prob": 0.75,
                                    },
                                    {
                                        "_target_": "monai.transforms.RandGaussianSmoothd",
                                        "keys": transformsd_keys,
                                        "prob": 0.75,
                                    },
                                    {
                                        "_target_": "monai.transforms.RandGaussianSharpend",
                                        "keys": transformsd_keys,
                                        "prob": 0.75,
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
            val_augmentations={
                "_target_": "monai.transforms.Compose",
                "transforms": [
                    {
                        "_target_": "monai.transforms.CropForegroundd",
                        "keys": transformsd_keys,
                        "source_key": transformsd_keys[0],
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
                        "spatial_size": training_image_size,
                        "mode": "constant",
                        "value": -1,
                    },
                    {
                        "_target_": "monai.transforms.DivisiblePadd",
                        "keys": transformsd_keys,
                        "k": training_latent_shape,
                        "mode": "constant",
                        "value": -1,
                    },
                    {
                        "_target_": "monai.transforms.RandSpatialCropSamplesd",
                        "keys": transformsd_keys,
                        "roi_size": training_image_size,
                        "num_samples": num_val_samples_per_datapoint,
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
            # start_from_checkpoint=r"/raid3/arjun/checkpoints/adaptive_autoencoder/v1__2025_02_13/version_0/checkpoints/last.ckpt",
            #
            max_epochs=200,
            lr=5e-4,
            seed=42,
            check_val_every_n_epoch=1,
            #
            loss_weights={
                "reconstruction_loss": 1.0,
                "perceptual_loss": 0.1,
                "ms_ssim_loss": 0.1,
                "kl_loss": 1e-4,
                # "spectral_loss": 1e-6,
            },
            kl_annealing_epochs=30,
            # free_bits=1.0,
            #
            checkpointing_level=2,
            #
            fast_dev_run=False,
            strategy="ddp",
            #
            accumulate_grad_batches=10,
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
        "Added adaptor architecture",
        "Updated augmentations",
        "Added activation checkpointing",
    ]

    additional_config = munchify(
        dict(
            task_name="v25__2025_03_06",
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
                    "e2ecloud16.e2e.qure.ai",  # First one is master node
                    "e2ecloud14.e2e.qure.ai",
                    "e2ecloud23.e2e.qure.ai",
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
