import os

triton_cache_dir = r"/raid/arjun/triton_cache"
os.environ["TRITON_CACHE_DIR"] = triton_cache_dir

import torch
from diffusion_3d.constants import SERVER_MAPPING
from diffusion_3d.utils.environment import set_multi_node_environment
from munch import munchify


def get_config(training_input_size=(96, 96, 96)):
    model_config = munchify(
        {
            "in_channels": 1,
            "num_channels": [18, 36, 72, 144],
            "depths": [4, 4, 4, 4],
            "attn_num_heads": [None, None, 8, 8],
            "timesteps": 1000,
            "mid_depth": 8,
            "kernel_size": 3,
            "normalization": "groupnorm",
            "normalization_pre_args": [6],
            "activation": "silu",
            "survival_prob": 1.0,
        }
    )
    model_config.time_channels = model_config.num_channels[0] * 4

    batch_size = 8
    num_train_samples_per_datapoint = 4
    num_val_samples_per_datapoint = batch_size

    transformsd_keys = ["image"]

    ct_min_hu = -1250
    ct_max_hu = 250

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
            checkpointspath=r"/raid3/arjun/checkpoints/diffusion_3d/",
            #
            limited_dataset_size=None,
            #
            allowed_spacings=((0.4, 7), (-1, -1), (-1, -1)),
            allowed_shapes=((training_input_size[0], -1), (256, -1), (256, -1)),
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
                        "a_min": ct_min_hu,
                        "a_max": ct_max_hu,
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
                        "roi_size": tuple(int(size * 0.8) for size in training_input_size),
                        "max_roi_size": tuple(int(size * 1.2) for size in training_input_size),
                        "random_size": True,
                        "num_samples": num_train_samples_per_datapoint,
                    },
                    {
                        "_target_": "vision_architectures.transforms.spatial.ResizedWithCropTrackingd",
                        "keys": transformsd_keys,
                        "original_shape_key": "Shape",
                        "spatial_size": training_input_size,
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
                        "a_min": ct_min_hu,
                        "a_max": ct_max_hu,
                        "b_min": -1.0,
                        "b_max": 1.0,
                        "clip": True,
                    },
                    {
                        "_target_": "vision_architectures.transforms.croppad.RandSpatialCropSamplesWithCropTrackingd",
                        "keys": transformsd_keys,
                        "roi_size": training_input_size,
                        "num_samples": num_val_samples_per_datapoint,
                    },
                    {  # In case image gets cropped to smaller size than required size
                        "_target_": "vision_architectures.transforms.spatial.ResizedWithCropTrackingd",
                        "keys": transformsd_keys,
                        "original_shape_key": "Shape",
                        "spatial_size": training_input_size,
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
                        "a_min": ct_min_hu,
                        "a_max": ct_max_hu,
                        "b_min": -1.0,
                        "b_max": 1.0,
                        "clip": True,
                    },
                    {
                        "_target_": "monai.transforms.CenterSpatialCropd",
                        "keys": transformsd_keys,
                        "roi_size": training_input_size,
                    },
                    {  # In case image gets cropped to smaller size than required size
                        "_target_": "vision_architectures.transforms.spatial.ResizedWithCropTrackingd",
                        "keys": transformsd_keys,
                        "original_shape_key": "Shape",
                        "spatial_size": training_input_size,
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
            start_from_checkpoint=None,
            # start_from_checkpoint=r"/raid3/arjun/checkpoints/adaptive_autoencoder/v69__2025_04_23__epoch155/version_0/checkpoints/last.ckpt",
            #
            max_epochs=1000,
            lr=1e-4,
            seed=42,
            check_val_every_n_epoch=5,
            #
            loss_weights={
                "l1_loss": 0.5,
                "l2_loss": 0.5,
            },
            #
            val_timesteps=200,
            val_ddim_eta=0.0,
            val_ddim_skip_steps=20,
            #
            checkpointing_level=0,
            #
            fast_dev_run=False,
            strategy={
                "_target_": "lightning.pytorch.strategies.fsdp.FSDPStrategy",
            },
            #
            accumulate_grad_batches=5,
            log_every_n_steps=1,
            # gradient_clip_val=5.0,
        )
    )

    clearml_tags = [
        f"Training image size: {training_input_size}",
        f"Dimensions: {model_config.num_channels}",
        f"Train batch size: {data_config.train_batch_size}",
        f"Checkpointing level: {training_config.checkpointing_level}",
        #
        "Base experiment",
        "Cosine noise scheduler",
        "Uniform timestep sampling",
        "Always conditioned on timesteps, spacings",
        "No gradient clipping",
        "Using FSDP",
    ]

    additional_config = munchify(
        dict(
            task_name="v1__2025_05_02",
            log_on_clearml=True,
            clearml_project="diffusion_3d",
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
            input_size=training_input_size,
        )
    )

    return config


# def get_data_config():
#     return {
#         "_target_": "qct_cache.cache_loading.CacheLoader",
#         "_partial_": True,
#         "cache_roots": [
#             "/raid/niraj/storage/qct_data_cache/3d_safetensors_seg_annot",
#             "/raid4/niraj/storage/qct_data_cache/3d_safetensors_seg_annot",
#             "/raid2/niraj/storage/qct_data_cache/3d_safetensors_seg_annot",
#             "/raid12/niraj/storage/qct_data_cache/3d_safetensors_seg_annot",
#             "/cache/fast_data_nas8/niraj/qct_data_cache/3d_safetensors_seg_annot",
#         ],
#         "sampling_heads": "${datamodule.sampling_heads}",
#         "cropping_head": "nodule",
#         "downsample_config": {"shape": [32, 96, 96], "mode": ["padcrop", "padcrop", "padcrop"]},
#         "filter_config": None,
#         "characteristic_legend": {
#             "nodule_texture": {"nan": -100, "none": -100, "solid": 0, "part_solid": 1, "ground_glass_nodule": 2},
#             "nodule_spiculation": {"nan": -100, "none": -100, "no": 0, "yes": 1},
#             "nodule_calcification": {"nan": -100, "none": -100, "no": 0, "yes": 1},
#             "nodule_malignancy": {
#                 "nan": -100,
#                 "none": -100,
#                 "highly unlikely": 0,
#                 "moderately unlikely": 1,
#                 "indeterminate": 2,
#                 "moderately suspicious": 3,
#                 "highly suspicious": 4,
#             },
#             "nodule_internalstructure": {
#                 "nan": -100,
#                 "none": -100,
#                 "internalstructure_soft_tissue": 0,
#                 "internalstructure_fat": 1,
#                 "internalstructure_fat_and_soft_tissue": 2,
#             },
#         },
#     }


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
