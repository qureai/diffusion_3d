config_controlnet_training = dict(
    max_epochs=100,
    inital_lr=1e-5,
    seed=42,
    check_val_every_n_epoch=1,
    #
    dummy=False,
    fast_dev_run=False,
    accumulate_grad_batches=5,
    gradient_clip_val=1.0,
)

clearml_tags = [
    f"Accumulate grad batches: {config_controlnet_training.accumulate_grad_batches}",
]

config_controlnet_training.update(
    dict(
        #
        task_name="v1__2025_01_15__maisi",
        log_on_clearml=True,
        clearml_project="chest_ct_maisi",
        clearml_tags=clearml_tags,
    )
)
