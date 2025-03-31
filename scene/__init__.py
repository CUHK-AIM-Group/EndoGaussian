#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams

class Scene:

    gaussians : GaussianModel
    
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.mode = args.mode

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        
        if os.path.exists(os.path.join(args.source_path, "sparse")) and args.extra_mark is None:
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")) and args.extra_mark == 'endonerf':
            scene_info = sceneLoadTypeCallbacks["endonerf"](args.source_path, mode=args.mode)
            print("Found poses_bounds.py and extra marks with EndoNeRf")
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")) and args.extra_mark == 'hamlyn':
            scene_info = sceneLoadTypeCallbacks["hamlyn"](args.source_path, mode=args.mode)
            print("Found poses_bounds.py and extra marks with EndoNeRf")
        elif os.path.exists(os.path.join(args.source_path, "point_cloud.obj")) or os.path.exists(os.path.join(args.source_path, "left_point_cloud.obj")):
            scene_info = sceneLoadTypeCallbacks["scared"](args.source_path, mode=args.mode, init_pts=args.init_pts)
            print("Found point_cloud.obj, assuming SCARED data!")
        else:
            assert False, "Could not recognize scene type!"
        
        self.maxtime = scene_info.maxtime
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("self.cameras_extent is ", self.cameras_extent)

        print("Loading Training Cameras")
        self.train_camera = scene_info.train_cameras
        print("Loading Test Cameras")
        self.test_camera = scene_info.test_cameras
        print("Loading Video Cameras")
        self.video_camera = scene_info.video_cameras
        
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        self.gaussians._deformation.deformation_net.grid.set_aabb(xyz_max,xyz_min)

        if self.loaded_iter:
            iteration_str = 'iteration_'+str(self.loaded_iter) if not load_coarse else 'coarse_iteration_'+str(self.loaded_iter)
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           iteration_str,
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    iteration_str,
                                                   ))
        else:
            if args.extra_mark == 'endonerf':
                self.gaussians.create_from_pcd(scene_info.point_cloud, args.camera_extent, self.maxtime)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
    
    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera