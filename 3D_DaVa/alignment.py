# -*- coding: utf-8 -*-

### IMPORTS ###
import open3d as o3d
from . import processing as proc
from functools import lru_cache
import numpy as np


def global_fast_alignment(source_pcd, target_pcd, param_list, threshold):
    '''
        FPFH-feature-based FGR global registration, with optimizable FPFH-processing.
    Args:
        source_pcd (PointCloud obj): point cloud to be aligned with target
        target_pcd (PointCloud obj): target point cloud
        param_list (list): ordered parameter list containing:
            - iter (int): number of iterations
            - normal_radius (float): radius for finding neighbours in normal estimation
            - feature_radius (float): radius for finding neighbours in FPFH calculation
            - max_nn_normal (int): maximum number of neighbours to be searched
            - max_nn_feature (int) :  maximum number of neighbours to be searched
        threshold (float): max correspondence points-pair distance
        
     Returns:
         result (RegistrationResult): result of registration process
    '''
    # Fetch actual parameters from list:
    iter = param_list[0]
    normal_radius = param_list[1]
    feature_radius = param_list[2]
    max_nn_normal = param_list[3]
    max_nn_feature = param_list[4]
    # Calculate FPFH features and downsampling:
    source_fpfh = proc.get_FPFH_features(source_pcd, normal_radius = normal_radius, feature_radius=feature_radius, max_nn_normal = max_nn_normal, max_nn_feature=max_nn_feature)
    target_fpfh = proc.get_FPFH_features(target_pcd, normal_radius = normal_radius, feature_radius=feature_radius, max_nn_normal = max_nn_normal, max_nn_feature=max_nn_feature)
    # Actual registration:
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(source_pcd, target_pcd, source_fpfh, target_fpfh, 
                                                                                   o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=threshold, iteration_number=iter))
    # trans = result.transformation
    return result


def evaluate(source_pcd, target_pcd, max_correspondence_distance, transformation):
    '''
        Evaluation function. Calculates and outputs two metrics: fitness, measures the overlapping area (higher is better) and inlier_rmse: measures RMSE of inlier correspondences (lower is better).
    Args:
        source_pcd (PointCloud obj): point cloud to be aligned with target
        target_pcd (PointCloud obj): target point cloud
        max_correspondence_distance (float): maximum correspondence points-pair distance
        transformation (np.ndarray): transformation matrix to be evaluated

    Returns:
        evaluation () : TODO
    '''
    eva = o3d.pipelines.registration.evaluate_registration(source_pcd, target_pcd, max_correspondence_distance, transformation)
    return eva

def icp_P2P_registration(source_pcd, target_pcd, distance_threshold, transformation, max_iter = 2000):
    '''
        Point-to-point ICP (Iterative Closest Point) algorithm. Returns transformation needed for alignment.
    Args:
        source_pcd (PointCloud obj): point cloud to be aligned with target
        target_pcd (PointCloud obj): target point cloud
        distance_threshold (float): maximum correspondence points-pair distance
        transformation (np.ndarray): initial transformation (e.g. obtained by global registration)
        max_iter (int): maximum number of iterations, if not converged, defaults to 2000

    Returns:
        trans (np.ndarray) : transformation matrix
    '''
    reg_p2p = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, distance_threshold, transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    trans = reg_p2p.transformation
    return trans

def icp_P2L_registration(source_pcd, target_pcd, distance_threshold, transformation):
    '''
        Point-to-plane ICP (Iterative Closest Point) algorithm. Returns transformation needed for alignment.
    Args:
        source_pcd (PointCloud obj): point cloud to be aligned with target
        target_pcd (PointCloud obj): target point cloud
        distance_threshold (float): maximum correspondence points-pair distance
        transformation (np.ndarray): initial transformation (e.g. obtained by global registration)

    Returns:
        trans (np.ndarray) : transformation matrix
    '''
    reg_p2l = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, distance_threshold, transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
    trans = reg_p2l.transformation
    return trans

#_____________________________________________________________________________________
def global_RANSAC_alignment(source_pcd, target_pcd, voxel_size = 0.05, ransac_n=3, ransac_iter = 100000, confidence = 0.999, edge_len = 0.9):
    '''
        RANSAC alignment. In each iteration, n random points are picked from the source point cloud. Their corresponding points in the target point cloud are detected by querying the nearest neighbor in the 33-dimensional FPFH feature space.
    Args:
        source_pcd (PointCloud): point cloud to be aligned with target
        target_pcd (PointCloud): target point cloud
        voxel_size (float): downsampling size (performance boost), defaults to 0.05
        ransac_iter (int): number of RANSAC iterations, defaults to 100000
        confidence (float): confidence probability
        edge_len (float): check if the point clouds build the polygons with similar edge lengths: 0 (loose) to 1 (strict)

    Returns:
        trans (np.ndarray): transformation matrix
    '''
    # Calculate FPFH features and downsampling:
    source_down, source_fpfh = proc.get_FPFH_features(source_pcd, voxel_size)
    target_down, target_fpfh = proc.get_FPFH_features(target_pcd, voxel_size)
    # Max correspondence points-pair distance (liberal, because of downsampling)
    distance_threshold = voxel_size
    # Actual registration:
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                edge_len),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iter, confidence))
    trans = result.transformation

    return trans


def global_fast_alignment_with_correspondence(source_pcd, target_pcd, correspondence_set):
    '''
        Fast alignment using corrrespondence set of points.
    Args:
        source_pcd (PointCloud obj): point cloud to be aligned with target
        target_pcd (PointCloud obj): target point cloud
        correspondence_set (Eigen::Vector2i): corresponding points
     Returns:
         result (RegistrationResult): result of registration process
    '''
    result = o3d.pipelines.registration.registration_fgr_based_on_correspondence(
        source_pcd, target_pcd, correspondence_set,
        o3d.pipelines.registration.FastGlobalRegistrationOption())
    return result
