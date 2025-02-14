import torch
from munch import munchify
from vision_architectures.nets.swinv2_3d import SwinV23DConfig

from diffusion_3d.constants import SERVER_MAPPING
from diffusion_3d.utils.environment import set_multi_node_environment


def get_config():
    image_size = (256, 256, 256)
    model_config = {
        "swin": SwinV23DConfig.model_validate(
            {
                "patch_size": (4, 4, 4),
                "in_channels": 1,
                "dim": 12,
                "drop_prob": 0.1,
                "stages": [
                    {
                        "depth": 1,
                        "num_heads": 2,
                        "intermediate_ratio": 4,
                        "layer_norm_eps": 1e-6,
                        "window_size": (4, 4, 4),
                        "use_relative_position_bias": True,
                        "attn_drop_prob": 0.1,
                        "proj_drop_prob": 0.1,
                        "mlp_drop_prob": 0.1,
                    },
                    {
                        "patch_merging": {
                            "merge_window_size": (2, 2, 2),
                            "out_dim_ratio": 4,
                        },
                        "depth": 1,
                        "num_heads": 4,
                        "intermediate_ratio": 4,
                        "layer_norm_eps": 1e-6,
                        "window_size": (4, 4, 4),
                        "use_relative_position_bias": True,
                        "attn_drop_prob": 0.1,
                        "proj_drop_prob": 0.1,
                        "mlp_drop_prob": 0.1,
                    },
                    {
                        "patch_merging": {
                            "merge_window_size": (2, 2, 2),
                            "out_dim_ratio": 4,
                        },
                        "depth": 3,
                        "num_heads": 8,
                        "intermediate_ratio": 4,
                        "layer_norm_eps": 1e-6,
                        "window_size": (2, 4, 4),
                        "use_relative_position_bias": True,
                        "attn_drop_prob": 0.1,
                        "proj_drop_prob": 0.1,
                        "mlp_drop_prob": 0.1,
                    },
                    {
                        "patch_merging": {
                            "merge_window_size": (2, 2, 2),
                            "out_dim_ratio": 4,
                        },
                        "depth": 1,
                        "num_heads": 16,
                        "intermediate_ratio": 4,
                        "layer_norm_eps": 1e-6,
                        "window_size": (1, 4, 4),
                        "use_relative_position_bias": True,
                        "attn_drop_prob": 0.1,
                        "proj_drop_prob": 0.1,
                        "mlp_drop_prob": 0.1,
                    },
                ],
            }
        ),
    }
    model_config["adaptor"] = munchify(
        {
            "dim": model_config["swin"].stages[-1]._out_dim,
            "adaptive_queries_size": (24, 8, 8),
            "num_heads": 16,
            "decoder_depth": 2,
            "intermediate_ratio": 4,
            "layer_norm_eps": 1e-6,
            "attn_drop_prob": 0.1,
            "proj_drop_prob": 0.1,
            "mlp_drop_prob": 0.1,
        }
    )
    model_config["decoder"] = SwinV23DConfig.model_validate(
        {
            "patch_size": (4, 4, 4),  # useless
            "in_channels": 1,  # useless
            "drop_prob": 0.1,  # useless
            "dim": model_config["adaptor"].dim,
            "stages": [
                {
                    "depth": 1,
                    "num_heads": 16,
                    "intermediate_ratio": 4,
                    "layer_norm_eps": 1e-6,
                    "window_size": (4, 4, 4),
                    "use_relative_position_bias": True,
                    "attn_drop_prob": 0.1,
                    "proj_drop_prob": 0.1,
                    "mlp_drop_prob": 0.1,
                    "patch_splitting": {
                        "final_window_size": (2, 2, 2),
                        "out_dim_ratio": 4,
                    },
                },
                {
                    "depth": 3,
                    "num_heads": 8,
                    "intermediate_ratio": 4,
                    "layer_norm_eps": 1e-6,
                    "window_size": (4, 4, 4),
                    "use_relative_position_bias": True,
                    "attn_drop_prob": 0.1,
                    "proj_drop_prob": 0.1,
                    "mlp_drop_prob": 0.1,
                    "patch_splitting": {
                        "final_window_size": (2, 2, 2),
                        "out_dim_ratio": 4,
                    },
                },
                {
                    "depth": 1,
                    "num_heads": 4,
                    "intermediate_ratio": 4,
                    "layer_norm_eps": 1e-6,
                    "window_size": (4, 4, 4),
                    "use_relative_position_bias": True,
                    "attn_drop_prob": 0.1,
                    "proj_drop_prob": 0.1,
                    "mlp_drop_prob": 0.1,
                    "patch_splitting": {
                        "final_window_size": (2, 2, 2),
                        "out_dim_ratio": 4,
                    },
                },
                {
                    "depth": 1,
                    "num_heads": 1,
                    "intermediate_ratio": 4,
                    "layer_norm_eps": 1e-6,
                    "window_size": (4, 4, 4),
                    "use_relative_position_bias": True,
                    "attn_drop_prob": 0.1,
                    "proj_drop_prob": 0.1,
                    "mlp_drop_prob": 0.1,
                },
            ],
        }
    )
    model_config["final_layer"] = {
        "in_channels": model_config["decoder"].stages[-1]._out_dim,
        "out_channels": model_config["swin"].in_channels,
        "kernel_size": (4, 4, 4),
    }
    # model_config["decoder"] = munchify(
    #     {
    #         "spatial_dims": 3,
    #         "channels": [12, 48, 192],
    #         "in_channels": model_config["swin"].stages[-1]._out_dim,
    #         "out_channels": model_config["swin"].in_channels,
    #         "num_res_blocks": [2, 2, 2],
    #         "norm_num_groups": 12,
    #         "norm_eps": 1e-6,
    #         "attention_levels": [False, False, False],
    #         "use_convtranspose": True,
    #     }
    # )

    data_config = munchify(
        dict(
            csvpath=r"/raid3/arjun/ct_pretraining/csvs/sources.csv",
            datapath=r"/raid3/arjun/ct_pretraining/scans/",
            checkpointspath=r"/raid3/arjun/checkpoints/ct_pretraining/",
            #
            limited_dataset_size=None,
            #
            allowed_spacings=((0.4, 7), (-1, -1), (-1, -1)),
            allowed_shapes=((225, 256), (256, -1), (256, -1)),
            #
            train_augmentations=[
                {
                    "__fn_name__": "random_resize_2d",
                    "min_shape": (
                        None,
                        image_size[1],
                        image_size[2],
                    ),
                    "max_shape": (
                        None,
                        min(int(image_size[1] * 1.1), 256),
                        min(int(image_size[2] * 1.1), 256),
                    ),
                    "interpolation_mode": "bilinear",
                },
                {
                    "__fn_name__": "random_crop",
                    "target_shape": (-1, 256, 256),
                },
                {
                    "__fn_name__": "pad_to_multiple_of",
                    "min_size": (32, 256, 256),
                    "tile_size": (32, 32, 32),
                },
                {
                    "__fn_name__": "random_rotate",
                    "degrees": 15,
                },
                {
                    "__fn_name__": "random_windowing",
                    "hotspots_and_stds": [
                        ((2000, 500), (10, 5)),  # Bone
                        ((1600, -600), (16, 6)),  # Lung
                        ((400, 40), (4, 1)),  # Abdomen
                        ((350, 50), (4, 1)),  # Soft Tissue
                        ((160, 60), (2, 1)),  # Liver
                        ((500, 50), (5, 1)),  # Mediastinum
                    ],
                    "sampling_probability": [1 / 6] * 6,
                    "normalize_range": (-1.0, 1.0),
                },
                {
                    "__fn_name__": "random_horizontal_flip",
                    "probability": 0.5,
                },
                [
                    [0.4, 0.3, 0.3],
                    [],
                    [
                        {
                            "__fn_name__": "random_gaussian_blurring",
                            "sigma_range": (0, 1),
                        }
                    ],
                    [
                        {
                            "__fn_name__": "random_unsharp_masking",
                            "sigma_range": (0, 1),
                            "alpha_range": (0.5, 2),
                        }
                    ],
                ],
            ],
            val_augmentations=[
                {
                    "__fn_name__": "resize_2d",
                    "target_shape": (None, 256, 256),
                    "interpolation_mode": "bilinear",
                },
                {
                    "__fn_name__": "pad_to_multiple_of",
                    "min_size": (32, 256, 256),
                    "tile_size": (32, 32, 32),
                },
                {
                    "__fn_name__": "random_windowing",
                    "window_choices": [
                        (2000, 500),  # Bone
                        (1600, -600),  # Lung
                        (400, 40),  # Abdomen
                        (350, 50),  # Soft Tissue
                        (160, 60),  # Liver
                        (500, 50),  # Mediastinum
                    ],
                    "normalize_range": (-1.0, 1.0),
                },
            ],
            #
            num_workers=6,
            batch_size=int(torch.cuda.get_device_properties(0).total_memory // 4e9),
            train_sample_size=20_000,
            sample_balance_cols=["Source", "BodyPart"],
        )
    )

    training_config = munchify(
        dict(
            start_from_checkpoint=None,
            # start_from_checkpoint=r"/cache/expdata1/arjun/checkpoints/ct_pretraining/v27__2024_06_30__v26_ct_pretraining/version_0/checkpoints/epoch=38.ckpt",
            #
            max_epochs=1000,
            inital_lr=1e-5,
            seed=42,
            check_val_every_n_epoch=1,
            #
            fast_dev_run=20,
            strategy="ddp",
            #
            # accumulate_grad_batches=5,  implemented directly
            # gradient_clip_val=1.0,  implemented directly
        )
    )

    grid_sizes = []
    for i in range(len(model_config["swin"].stages)):
        grid_sizes.append(
            tuple([size // patch for size, patch in zip(image_size, model_config["swin"].stages[i]._out_patch_size)])
        )
    # Ensure grid size can be divided by window size
    for i in range(len(model_config["swin"].stages)):
        assert all(
            [grid % window == 0 for grid, window in zip(grid_sizes[i], model_config["swin"].stages[i].window_size)]
        ), f"{grid_sizes[i]} is not divisible by {model_config['swin'].stages[i]['window_size']}"
    clearml_tags = [
        f"Training image size: {image_size}",
        f"Patch sizes: {[stage._out_patch_size for stage in model_config['swin'].stages]}",
        f"Grid sizes: {grid_sizes}",
        f"Dimensions: {[stage._out_dim for stage in model_config['swin'].stages]}",
        f"Batch size: {data_config.batch_size}",
        #
        "VAE",
    ]

    additional_config = munchify(
        dict(
            task_name="v1__2025_02_13",
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
        )
    )

    return config
