ModelParams = dict(
    extra_mark = 'scared',
    no_fine=False,
    init_pts=20_000,
    mode='monocular'
)

OptimizationParams = dict(
    coarse_iterations = 3000,
    iterations = 3000,
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 3000,
    
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,
    
    pruning_interval = 2000,
    percent_dense = 0.01,
    opacity_reset_interval = 3000,
)

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 100]
    },
    multires = [1, 2, 4, 8],
    defor_depth = 0,
    net_width = 32,
    plane_tv_weight = 0,
    time_smoothness_weight = 0,
    l1_time_planes =  0,
    weight_decay_iteration=0,
    no_dx=False,
    no_ds=True,
    no_dr=True,
    no_do=False
)