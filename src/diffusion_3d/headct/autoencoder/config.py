import torch
from munch import munchify
from vision_architectures.nets.swinv2_3d import populate_and_validate_config

from diffusion_3d.constants import SERVER_MAPPING
from diffusion_3d.utils.environment import set_multi_node_environment


def get_config():
    model_config = {
        "encoder": populate_and_validate_config(
            {
                "patch_size": (1, 6, 6),
                "in_channels": 1,
                "use_absolute_position_embeddings": True,
                "learnable_absolute_position_embeddings": False,
                "embed_spacing_info": False,
                "image_size": (64, 384, 384),
                "dim": 24,
                "drop_prob": 0.1,
                "stages": [
                    {
                        "patch_merging": None,
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
                        "num_heads": 16,
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
        ),
    }
    model_config["decoder"] = populate_and_validate_config(
        {
            "patch_size": model_config["encoder"]["stages"][-1]["_out_patch_size"],
            "in_channels": model_config["encoder"]["in_channels"],
            "dim": model_config["encoder"]["stages"][-1]["_out_dim"],
            "drop_prob": 0.1,
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
                    "num_heads": 2,
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
            ],
        }
    )
    model_config = munchify(model_config)

    data_config = munchify(
        dict(
            csvpath=r"/raid3/arjun/ct_pretraining/csvs/sources.csv",
            datapath=r"/raid3/arjun/ct_pretraining/scans/",
            checkpointspath=r"/raid3/arjun/checkpoints/diffusion_3d/",
            #
            limited_dataset_size=None,
            #
            sources=["vrad", "medall", "fts"],
            bodyparts=["head"],
            allowed_spacings=((0.4, 7), (-1, -1), (-1, -1)),
            allowed_shapes=((64, -1), (-1, -1), (-1, -1)),
            #
            train_augmentations=[
                {
                    "__fn_name__": "pad_to_target_shape",
                    "target_shape": (model_config.encoder.image_size[0], 512, 512),
                    "mode": "random",
                },
                {
                    "__fn_name__": "random_crop",
                    "target_shape": (-1, 512, 512),
                },
                {
                    "__fn_name__": "random_rotate",
                    "degrees": 25,
                },
                {
                    "__fn_name__": "random_resize",
                    "min_shape": (
                        model_config.encoder.image_size[0],
                        model_config.encoder.image_size[1],
                        model_config.encoder.image_size[2],
                    ),
                    "max_shape": (
                        int(model_config.encoder.image_size[0] * 1.2),
                        min(int(model_config.encoder.image_size[1] * 1.1), 512),
                        min(int(model_config.encoder.image_size[2] * 1.1), 512),
                    ),
                    "interpolation_mode": "trilinear",
                },
                {
                    "__fn_name__": "random_crop",
                    "target_shape": model_config.encoder.image_size,
                },
                {
                    "__fn_name__": "random_windowing",
                    "hotspots_and_stds": [
                        [(80, 40), (7, 2)],  # Brain window
                        [(37, 37), (4, 2)],  # Stroke window
                        [(3400, 650), (360, 35)],  # Bone window
                        [(8, 32), (0.5, 2)],  # Another stroke window
                        [(210, 75), (10, 4)],  # subdural window
                        [(375, 40), (10, 2)],  # Soft tissue window
                    ],
                    "sampling_probability": [0.4, 0.3, 0.15, 0.05, 0.05, 0.05],
                    "normalize_range": (0, 1),
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
                    "__fn_name__": "pad_to_target_shape",
                    "target_shape": (model_config.encoder.image_size[0], 512, 512),
                    "mode": "random",
                },
                {
                    "__fn_name__": "center_crop",
                    "target_shape": (-1, 512, 512),
                },
                {
                    "__fn_name__": "resize",
                    "target_shape": model_config.encoder.image_size,
                    "interpolation_mode": "trilinear",
                },
                {
                    "__fn_name__": "random_windowing",
                    "window_choices": [
                        (80, 40),  # Brain window
                        # (37, 37),  # Stroke window
                        # (2800, 600),  # Bone window
                        # (4000, 700),  # Bone window
                        # (8, 32),  # Another stroke window
                        # (210, 75),  # subdural window``
                        # (375, 40),  # Soft tissue window
                    ],
                    "normalize_range": (0, 1),
                },
            ],
            #
            num_workers=6,
            # batch_size=int(torch.cuda.get_device_properties(0).total_memory // 1.25e9),  # (32, 384, 384) 100M
            # batch_size=int(torch.cuda.get_device_properties(0).total_memory // 3.2e9),  # (48, 384, 384) 100M
            batch_size=int(torch.cuda.get_device_properties(0).total_memory // 4e9),  # (64, 384, 384) 100M
            # train_sample_size=168_000,
            train_sample_size=70_000,  # (64, 384, 384)
            sample_balance_cols=["Source", "BodyPart"],
        )
    )

    training_config = munchify(
        dict(
            pretraining=True,
            start_from_checkpoint=None,
            # start_from_checkpoint=r".../version_0/checkpoints/epoch=38.ckpt",
            #
            max_epochs=1000,
            inital_lr=1e-5,
            seed=42,
            check_val_every_n_epoch=1,
            #
            dummy=False,
            fast_dev_run=10,
            strategy="ddp",
            #
            accumulate_grad_batches=5,
            gradient_clip_val=1.0,
        )
    )

    grid_sizes = []
    for i in range(len(model_config.encoder.stages)):
        grid_sizes.append(
            tuple(
                [
                    size // patch
                    for size, patch in zip(
                        model_config.encoder.image_size, model_config.encoder.stages[i]["_out_patch_size"]
                    )
                ]
            )
        )
    # Ensure grid size can be divided by window size
    for i in range(len(model_config.encoder.stages)):
        assert all(
            [grid % window == 0 for grid, window in zip(grid_sizes[i], model_config.encoder.stages[i]["window_size"])]
        ), f"{grid_sizes[i]} is not divisible by {model_config.encoder.stages[i]['window_size']}"
    clearml_tags = [
        f"Image size: {model_config.encoder.image_size}",
        f"Patch sizes (encoder): {[stage['_out_patch_size'] for stage in model_config.encoder.stages]}",
        f"Grid sizes (encoder): {grid_sizes}",
        f"Dimensions (encoder): {[stage['_out_dim'] for stage in model_config.encoder.stages]}",
        f"Patch sizes (decoder): {[stage['_out_patch_size'] for stage in model_config.decoder.stages]}",
        f"Dimensions (decoder): {[stage['_out_dim'] for stage in model_config.decoder.stages]}",
        f"Sources: {data_config.sources}",
        f"BodyParts: {data_config.bodyparts}",
        f"Batch size: {data_config.batch_size}",
        #
        "SwinV2 autoencoder",
    ]

    additional_config = munchify(
        dict(
            task_name="v1__2024_12_24",
            log_on_clearml=True,
            clearml_project="diffusion_3d",
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


if __name__ == "__main__":
    from pprint import pprint

    pprint(get_config().toDict())
