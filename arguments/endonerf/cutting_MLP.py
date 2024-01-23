ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 15
)

OptimizationParams = dict(
    coarse_iterations = 1000,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,
    iterations = 20000,
    percent_dense = 0.01,
    opacity_reset_interval = 4000,
    position_lr_max_steps = 20000,
    prune_interval = 2000
)

ModelHiddenParams = dict(
    defor_depth = 8,
    no_grid = True,
    net_width = 256,
    plane_tv_weight = 0,
    time_smoothness_weight = 0,
    l1_time_planes =  0,
    weight_decay_iteration=0,
)