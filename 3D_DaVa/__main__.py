# -*- coding: utf-8 -*-
'''This file should be invoked as python -m 3D_DaVa'''

### IMPORTS ###
import os
import glob
from . import processing as proc
from . import alignment as ali
from . import inout
from functools import lru_cache
import numpy as np
import sys
import argparse
import pickle
import open3d as o3d
import copy
import time
import matplotlib.pyplot as plt
import datetime
import random
import re
import concurrent.futures
from itertools import combinations
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

# Global variables - used in No-Reference and Reference Processing:
nrpcqa_downsized = None
nrpcqa_links = None
nrpcqa_clean = None
nrpcqa_voxel_size = None
nrpcqa_probs = None
nrpcqa_tree = None
rpcqa_completeness_ratio = 0.0
original = None
large = False
saving = False  #i.e Save denoised point cloud
snapshotting = False # i.e Save intermediary outputs
visualization = False # i.e Visualize intermediary outputs

@lru_cache
def process_with_reference(input_file:str, reference_file:str, *args):
    '''
        Point Cloud Data Validation - with reference
    Args:
        input_file (str) : path of point cloud to analyse
        reference_file (str) : path of reference point cloud
        *args (str) : name of output file - optional
    '''
    # TODO: For intermediary metric: turn point cloud (not downsized) to mesh using voxel modelling
    # TODO: Try other R-PCQA approaches (see literature-Graph Similarity) ...
    # TODO: Clean functions and envelop
    # TODO: Try other registration methods (ISS/Curvature/Graph)

   # Global variables to reuse:
    global nrpcqa_downsized
    global nrpcqa_links
    global nrpcqa_clean
    global nrpcqa_voxel_size
    global nrpcqa_probs
    # Keep boolean check for large detected cloud:
    global large
    global original
    global rpcqa_completeness_ratio
    # Keep boolean check for visualization/snapshots:
    global snapshotting
    global visualization


    print("Action in progress: process cloud without reference...")
    process_without_reference(input_file)
    denoised_pcd = nrpcqa_clean # Pre-denoised cloud
    NUM_RAW_POINTS_PCD = len(original.points)
    # Cloud to be worked on:
    pcd = copy.deepcopy(original)

    # Keep track of processing time:
    start_time = time.time()

    print("Action in progress: read and sample reference...")
    # Read reference:
    original_ref = inout.read_mesh(reference_file)
    # If large, use as many points as original pcd, else triple the number:
    if large:
        factor = 1
    else:
        factor = 3
    sampling_rate = NUM_RAW_POINTS_PCD * factor
    # Sample CAD, turning mesh into point cloud:
    ref_pcd = proc.create_uniform_sampled_cloud_from_mesh(original_ref, nr_points = sampling_rate)
    # Reference to be worked on:
    ref = copy.deepcopy(ref_pcd)
    NUM_RAW_POINTS_REF = len(ref.points)
    if snapshotting:
        inout.snapshot(ref, name = "rpcqa_sampled_ref")
    if visualization:
        inout.visualize(ref)
    

    print("Action in progress: scaling point clouds to same size...")
    # Rescale pcd to mesh size (or the other way around). Use denoised cloud for this for better results.
    pcd_max_bound = denoised_pcd.get_max_bound()
    pcd_min_bound = denoised_pcd.get_min_bound()
    pcd_dims = pcd_max_bound - pcd_min_bound
    ref_max_bound = ref.get_max_bound()
    ref_min_bound = ref.get_min_bound()
    ref_dims = ref_max_bound - ref_min_bound
    # Check which boundary box is bigger by volume:
    vol1 = np.prod(pcd_dims)
    vol2 = np.prod(ref_dims)
    if vol1 > vol2:
        scaling_factor = max(pcd_dims) / max(ref_dims)
        ref.scale(scaling_factor, center=ref.get_center())
        original_ref.scale(scaling_factor, center=ref.get_center())
    else:
        scaling_factor = max(ref_dims) / max(pcd_dims)
        # Scale point cloud and denoised point cloud:
        pcd.scale(scaling_factor, center=pcd.get_center())
        denoised_pcd.scale(scaling_factor, center=denoised_pcd.get_center())
    if visualization:
        inout.visualize_differences(pcd, ref)


    print("Action in progress: eventual mirror transformation for symmetric point clouds ...")
    # In case orientation is wrong, one getst he choice of flipping over the first two axis.
    mirror_check = input("Mirror PCD? (y/n) or ENTER to continue:")
    mirror_check = mirror_check.lower()
    yes_choices = ["y", "yes"]
    no_choices = ["n", "no"]
    possible_choices = [""] + yes_choices + no_choices
    while mirror_check not in possible_choices:
        mirror_check = input("Type error! Mirror PCD? (y/n) or ENTER to continue:")
        mirror_check = mirror_check.lower()
    if mirror_check in yes_choices:
        # Compute the principal component analysis (PCA) - for original pcd:
        mean, covariance = pcd.compute_mean_and_covariance()
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        largest_eigenvector_index = np.argmax(eigenvalues)
        second_largest_eigenvector_index = np.argsort(eigenvalues)[-2]
        mirrored_points = np.array(pcd.points)
        mirrored_points[:, [largest_eigenvector_index, second_largest_eigenvector_index]] *= -1
        mirrored_point_cloud = o3d.geometry.PointCloud()
        mirrored_point_cloud.points = o3d.utility.Vector3dVector(mirrored_points)
        pcd = mirrored_point_cloud
        if visualization:
            inout.visualize_differences(pcd, ref)
        # Compute the principal component analysis (PCA) - for denoised pcd:
        mean, covariance = denoised_pcd.compute_mean_and_covariance()
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        largest_eigenvector_index = np.argmax(eigenvalues)
        second_largest_eigenvector_index = np.argsort(eigenvalues)[-2]
        mirrored_points = np.array(denoised_pcd.points)
        mirrored_points[:, [largest_eigenvector_index, second_largest_eigenvector_index]] *= -1
        mirrored_point_cloud = o3d.geometry.PointCloud()
        mirrored_point_cloud.points = o3d.utility.Vector3dVector(mirrored_points)
        denoised_pcd = mirrored_point_cloud
        if visualization:
            inout.visualize_differences(denoised_pcd, ref)


    print("Action in progress: downsize/voxelize denoised point cloud and reference ...")
    # Original downsized variables:
    down_pcd = nrpcqa_downsized 
    pcd_links = nrpcqa_links
    voxel_size_pcd = nrpcqa_voxel_size
    # Get ref and denoised downsized variables:
    if large:
        constraint = 0.0025
    else:
        constraint = 0.025
    # Downsize given boundary box constraint - ref:
    voxel_size_ref = round(max(ref.get_max_bound() - ref.get_min_bound()) * constraint, 8)
    down_ref, corr_inds_ref = proc.downsample_and_trace_cloud(ref, voxel_size_ref)
    voxel_size_denoised_pcd = round(max(denoised_pcd.get_max_bound() - denoised_pcd.get_min_bound()) * constraint, 8)
    down_denoised_pcd, corr_inds_denoised_pcd = proc.downsample_and_trace_cloud(denoised_pcd, voxel_size_denoised_pcd)
    if snapshotting:
        inout.snapshot(down_ref, name = "rpcqa_downsized_ref")
    if visualization:
        inout.visualize(down_ref)

     # Use downsized and denoised versions if the clouds are too large:
    if large:
        alignment_ref = down_ref
        alignment_pcd = down_denoised_pcd
        alignment_voxel_size = voxel_size_denoised_pcd
    else:
        alignment_pcd = denoised_pcd
        alignment_ref = ref
        alignment_voxel_size = voxel_size_pcd


    print("Action in progress: fast global alignment using feature matching...")
    # TODO: Non-blocking visualization (http://www.open3d.org/docs/latest/tutorial/visualization/non_blocking_visualization.html)
    iter = 50  # The larger the better -> but more time use
    max_nn = 50
    large_factor = 5
    max_bound = alignment_pcd.get_max_bound()
    min_bound = alignment_pcd.get_min_bound()
    max_dist = (proc.get_l2_distance(min_bound, max_bound)*0.1)/2
    normal_radius = alignment_voxel_size * large_factor if large else max_dist 
    feature_radius = alignment_voxel_size * large_factor if large else max_dist
    max_nn_normal = max_nn * large_factor if large else max_nn
    max_nn_feature = max_nn * large_factor if large else max_nn
    solution = [iter, normal_radius, feature_radius, max_nn_normal, max_nn_feature]
    distances = alignment_pcd.compute_point_cloud_distance(alignment_ref)
    lowest_distance = np.min(distances)
    bounding_box_diagonal = proc.get_l2_distance(min_bound, max_bound)
    overlap_rate = bounding_box_diagonal/lowest_distance
    # Define custom tolerances
    relative_tolerance = 0.1  # 10%
    absolute_tolerance = 0.5  # 0.5 units
    # If overlapping clouds, choose thresholds according to lowest distance between closest points
    if overlap_rate >= 1:
        if np.isclose(bounding_box_diagonal, overlap_rate, rtol=relative_tolerance, atol=absolute_tolerance):
            threshold = lowest_distance
        else:
            threshold = lowest_distance/2
    else:
        threshold = (bounding_box_diagonal * overlap_rate)/2
    global_result = ali.global_fast_alignment(alignment_pcd, alignment_ref, solution, threshold)
    global_trans = global_result.transformation
    vis_alignment_pcd = copy.deepcopy(alignment_pcd)
    vis_alignment_pcd.transform(global_trans)
    if visualization:
        inout.visualize_differences(vis_alignment_pcd, alignment_ref)


    print("Action in progress: registration refinement using ICP local alignment (P2P)...")
    # Refine until sufficient registration (here 1 represents a perfect registration): 
    good_registration_check = False
    good_fitness_lower_threshold = 0.9
    good_fitness_upper_threshold = 1.0
    max_iterations = 3 # Holds maximum number of iterations before picking last achieved transformation
    curr_iterations = 0 # Holds current iteration value
    while not good_registration_check:
        icp_trans = ali.icp_P2P_registration(alignment_pcd, alignment_ref, threshold, global_trans)
        eva = ali.evaluate(alignment_pcd, alignment_ref, threshold, icp_trans)
        fitness = eva.fitness
        print("Point cloud registration fitness: ", fitness)
        if fitness <= good_fitness_upper_threshold and fitness > good_fitness_lower_threshold:
            good_registration_check = True
        elif curr_iterations == max_iterations:
            good_registration_check = True
        else:
            global_trans = icp_trans
            curr_iterations += 1

    pcd.transform(icp_trans) # Transform actual (original) point cloud 
    if visualization:
        inout.visualize_differences(pcd, ref)


    print("Action in progress: distortion detection based on euclidean distance (P2Point)...")
    # Calculate distances between downsized digitized scan and reference
    distances = pcd.compute_point_cloud_distance(ref)
    distances = np.asarray(distances)
    # Update scores (the lowest the distance, the better --> lower score):
    mapped_probs = proc.map_to_probabilities(distances, inverse=False)
    for i in range(len(nrpcqa_probs)):
        nrpcqa_probs[i] += mapped_probs[i]

    print("Action in progress: completeness estimated from reference distances (P2Point)...")
    # Calculate distances between downsized digitized scan and reference
    distances = ref.compute_point_cloud_distance(pcd)
    distances = np.asarray(distances)
    Q1 = np.percentile(distances, 25) # First quartile
    Q3 = np.percentile(distances, 75) # Third quartile
    IQR = Q3 - Q1
    missing_values_threshold = Q3 + 1.5 * IQR
    missing_values_indices = np.where(distances > missing_values_threshold)[0]
    missing_values_ref_pcd = ref.select_by_index(missing_values_indices)
    ok_ref_pcd =  ref.select_by_index(missing_values_indices, invert=True)
    if snapshotting:
        inout.snapshot(missing_values_ref_pcd, name="rpcqa_missing_values")
    if visualization:
        inout.visualize_differences(ok_ref_pcd, missing_values_ref_pcd)
    NUM_MISSING_VALUES = len(missing_values_ref_pcd.points)
    REF_COMPLETENESS = 1.0 - (NUM_MISSING_VALUES/NUM_RAW_POINTS_REF)


    print("Action in progress: distortion detection based on point-to-plane euclidean distances (P2Plane)...")
    # Point2plane:
    scene = o3d.t.geometry.RaycastingScene()
    mesh_ids = {}
    ref_mesh = copy.deepcopy(original_ref)
    reference = o3d.t.geometry.TriangleMesh.from_legacy(ref_mesh)
    mesh_ids[scene.add_triangles(reference)] = 'ref'
    query_points = np.array(pcd.points).astype(np.float32)
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points['points'].numpy(),
                              axis=-1)
    rays = np.concatenate([query_points, np.ones_like(query_points)], axis=-1)
    intersection_counts = scene.count_intersections(rays).numpy()
    is_inside = intersection_counts % 2 == 1
    distance[is_inside] *= -1
    signed_distances = distance # Negative values mean points within a mesh.
    # Update scores (the lowest the distance, the better --> lower score):
    mapped_probs = proc.map_to_probabilities(signed_distances, inverse=False)
    for i in range(len(nrpcqa_probs)):
        nrpcqa_probs[i] += mapped_probs[i]
    # TODO: Sofia's approach has several metrics that could be used.
    # Check triangle occurence:
    # closest_triangles = closest_points['geometry_ids'].numpy()
    # counts = Counter(closest_triangles)
    # METRIC_MEAN_NR_OF_POINTS_IN_TRIANGLE = np.mean(list(counts.values()))

    '''
    print("Action in progress: completeness estimation based on surface areas...")
    # Turn scaled point clouds into mesh and compute surface area:
    pcd_mesh = proc.point_to_mesh(denoised_pcd, voxel_size_denoised_pcd)
    pcd_mesh_surface_area = pcd_mesh.get_surface_area()
    ref_mesh = proc.point_to_mesh(ref, voxel_size_ref)
    reference_surface_area = ref_mesh.get_surface_area()
    if reference_surface_area >= pcd_mesh_surface_area:
        SURFACE_COMPLETENESS = pcd_mesh_surface_area/reference_surface_area
    else:
        SURFACE_COMPLETENESS = reference_surface_area/pcd_mesh_surface_area
    # Surface completeness closer to 1 is better, close to 0 is worst.
    if snapshotting:
        inout.snapshot(pcd_mesh, name="rpcqa_meshed_pcd")
    if visualization:
        inout.visualize(pcd_mesh)


    print("Action in progress: completeness estimated from PCA similarity...")
    components_pcd, exp_var_ratio_pcd, exp_var_pcd, transformed_pcd = proc.principal_component_analysis(np.array(pcd.points))
    components_ref, exp_var_ratio_ref, exp_var_ref, transformed_ref = proc.principal_component_analysis(np.array(ref.points))
    # Get metrics:
    METRIC_TRUENESS_PCA_1 = abs(proc.cosine_similarity(components_pcd[0], components_ref[0]))
    METRIC_TRUENESS_PCA_2 = abs(proc.cosine_similarity(components_pcd[1], components_ref[1]))
    METRIC_TRUENESS_PCA_3 = abs(proc.cosine_similarity(components_pcd[2], components_ref[2]))
    PCA_COMPLETENESS = (METRIC_TRUENESS_PCA_1 + METRIC_TRUENESS_PCA_2 + METRIC_TRUENESS_PCA_3)/3
    '''

    # Compute completeness score:
    # rpcqa_completeness_ratio = (SURFACE_COMPLETENESS + PCA_COMPLETENESS + REF_COMPLETENESS) / 3
    rpcqa_completeness_ratio = REF_COMPLETENESS

    print("Action in progress: denoising original point cloud...") # Must be last
    cleaning(snapshotting = snapshotting, visualization = visualization)

    # End processing time
    end_time = time.time()
    rpcqa_time = datetime.timedelta(seconds=end_time - start_time)
    print("Processing time (RPCQA):" +  str(rpcqa_time))



#____________________________________NR-PCQA____________________________________
def nrpcqa_downsize(pcd, snapshotting = False, visualization = False):
    '''
        Downsize big models to allow faster processing. Boundary box constraints are applied for adaptive voxel sizing.
    Args:
        pcd (type: PointCloud object) : point cloud to downsample
        snapshotting (bool) : allow (True) taking snapshots along the process, or default (False)
        visualization (bool) : allow (True) visualization along the process, or default (False)
    '''

    global nrpcqa_downsized
    global nrpcqa_voxel_size
    global nrpcqa_links
    global large

    if large:
        constraint = 0.0025
    else:
        constraint = 0.025
    # Downsize given boundary box-based constraint:
    voxel_size = round(max(pcd.get_max_bound() - pcd.get_min_bound()) * constraint, 8)
    down, corr_inds = proc.downsample_and_trace_cloud(pcd, voxel_size)
    # Save for reference-processing:
    pcd_voxel_inds = range(len(corr_inds))
    pcd_actual_inds = [list(corr_inds[i]) for i in pcd_voxel_inds]
    pcd_links = dict(zip(pcd_voxel_inds, pcd_actual_inds))
    # Visual outputs:
    if snapshotting:
        inout.snapshot(down, name = "nrpcqa_downsized")
    if visualization:
        inout.visualize(down)
    # Update global:
    nrpcqa_downsized = down
    nrpcqa_voxel_size = voxel_size
    nrpcqa_links = pcd_links


def nrpcqa_modelling(pcd, snapshotting = False, visualization = False):
    '''
        Turns a point cloud to a mesh by 3D-modelling voxels and clusters connected areas / finds low-connection areas to calculate metrics related to density.
    Args:
        pcd (type: PointCloud object) : point cloud to model
        snapshotting (bool) : allow (True) taking snapshots along the process, or default (False)
        visualization (bool) : allow (True) visualization along the process, or default (False)
    '''
    global nrpcqa_voxel_size
    global nrpcqa_links
    global nrpcqa_downsized
    global nrpcqa_probs
    # Get mesh and triangles (indices of vertices):
    v_mesh = proc.point_to_mesh(pcd, nrpcqa_voxel_size)
    v_mesh_triangles =  np.asarray(v_mesh.triangles)
    # Clustering connected triangles ...
    triangle_cluster_ids, cluster_n_triangles, cluster_area = v_mesh.cluster_connected_triangles()
    triangle_cluster_ids = np.asarray(triangle_cluster_ids)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    # Remove small clusters (small = 90% reduction in number of triangles compared to biggest cluster):
    biggest_cluster_ind = np.argmax(cluster_n_triangles)
    constraint = int(cluster_n_triangles[biggest_cluster_ind] * 0.1)
    # big_cluster_triangles = cluster_n_triangles[triangle_cluster_ids] > constraint
    small_cluster_triangles = cluster_n_triangles[triangle_cluster_ids] <= constraint
    small_cluster_num_triangles = np.unique(cluster_n_triangles[cluster_n_triangles <= constraint])
    if len(small_cluster_num_triangles) != 0:
        max_small_cluster_num_triangles = np.max(small_cluster_num_triangles)
        # Associate each small cluster to a color:
        colors = proc.generate_N_unique_colors(len(small_cluster_num_triangles))
        num_color_dict = dict(zip(small_cluster_num_triangles, colors))
        color_num_dict = dict(zip(colors, small_cluster_num_triangles))
        # Fetch vertice indices for each small cluster:
        num_tr_vs_vertices = {}
        for num_tr in small_cluster_num_triangles:
            num_tr_mask = cluster_n_triangles[triangle_cluster_ids] == num_tr
            num_tr_vs_vertices[num_tr] = np.unique(np.ravel(v_mesh_triangles[num_tr_mask]))
        # Update with new colors for each small cluster:
        low_density_mesh = copy.deepcopy(v_mesh)
        small_clusters = []
        for num_tr in small_cluster_num_triangles:
            cluster_color = num_color_dict[num_tr]
            part_low_density_mesh = low_density_mesh.select_by_index(num_tr_vs_vertices[num_tr])
            part_low_density_mesh.paint_uniform_color(list(cluster_color))
            small_clusters.append(part_low_density_mesh)
        # Aggregate points to construct small cluster mesh:
        small_cluster_mesh = small_clusters[0]
        for i in range(1,len(small_clusters)):
            small_cluster_mesh += small_clusters[i]
        # Add high-density mesh as well and paint it uniformily:
        high_density_mesh = copy.deepcopy(v_mesh)
        high_density_mesh.remove_triangles_by_mask(small_cluster_triangles)
        high_density_mesh.remove_unreferenced_vertices()
        high_density_mesh.paint_uniform_color([0, 0.5, 0]) # Green
        full_mesh  = high_density_mesh + small_cluster_mesh
        # Turn mesh into point cloud:
        full_pcd = o3d.geometry.PointCloud()
        full_pcd.points = full_mesh.vertices
        full_pcd.colors = full_mesh.vertex_colors
        # Score distortion by number of triangles in the small cluster it belongs to:
        tree = o3d.geometry.KDTreeFlann(full_pcd)
        np_down = np.array(nrpcqa_downsized.points)
        values = np.empty(nrpcqa_probs.shape[0])
        for i in range(len(np_down)):
            # For each voxel, get closest point in full_pcd:
            [k, idx, _] = tree.search_knn_vector_3d(np_down[i], 1)
            # Get color of the point in full_pcd:
            point_color = list(np.asarray(full_pcd.colors)[idx[0]])
            # Get actual point cloud indices:
            actual_inds = nrpcqa_links[i]
            if point_color == [0, 0.5, 0]:
                for ind in actual_inds:
                    values[ind] = max_small_cluster_num_triangles + 1 # Set a bigger number than the biggest small cluster
            else:
                for ind in actual_inds:
                    values[ind] = color_num_dict[tuple(point_color)]
        # Update scores:
        mapped_probs = proc.map_to_probabilities(values)
        for k,v in nrpcqa_links.items():
            for ind in v:
                nrpcqa_probs[ind] += mapped_probs[k]
        if snapshotting:
            inout.snapshot(full_mesh, name = "nrpcqa_modelling_clusters")
        if visualization:
            inout.visualize(full_mesh)
    else:
        if snapshotting:
            inout.snapshot(v_mesh, name = "nrpcqa_modelling")
        if visualization:
            inout.visualize(v_mesh)


def nrpcqa_voxelization(pcd, snapshotting = False, visualization = False, shades = 25):
    '''
        Uses voxelized model statistics to calculate metrics related to completeness.
    Args:
        pcd (type: PointCloud object) : point cloud to voxelize
        snapshotting (bool) : allow (True) taking snapshots along the process, or default (False)
        visualization (bool) : allow (True) visualization along the process, or default (False)
        shades (int) : equal to bar number in plot, if plotting is allowed, defaults to 25
    '''
    global nrpcqa_links
    global nrpcqa_downsized
    global nrpcqa_probs

    # Get number of points per voxel (with and without index):
    values = [len(v) for v in list(nrpcqa_links.values())]
    # Color voxelized cloud according to number of neighbours:
    colored_down, color_range = proc.color_cloud_greyscale(nrpcqa_downsized, values, shades = shades)
    if snapshotting:
        inout.snapshot(colored_down, name = "nrpcqa_voxelized")
        inout.plot_values_by_color(values, color_range, x_label="Number contained points per voxel", y_label="Number of voxels", save = True, name="nrpcqa_plot_voxelization")
    if visualization:
        inout.visualize(colored_down)
        inout.plot_values_by_color(values, color_range, x_label="Number contained points per voxel", y_label="Number of voxels", show=True, name="nrpcqa_plot_voxelization")
    # Update scores:
    mapped_probs = proc.map_to_probabilities(values)
    for k,v in nrpcqa_links.items():
        for ind in v:
            nrpcqa_probs[ind] += mapped_probs[k]
    

def nrpcqa_radius_nb(pcd, snapshotting = False, visualization = False, k_points = 5, n_nb= 10, shades = 10):
    '''
        Uses radius-based neighbourhood statistics to calculate metrics related to completeness/validity.
    Args:
        pcd (type: PointCloud object) : point cloud to analyze
        snapshotting (bool) : allow (True) taking snapshots along the process, or default (False)
        visualization (bool) : allow (True) visualization along the process, or default (False)
        k_points (int) : radius is estimated by the mean distance to n_nb neighbours from a random set of k_points points, k_points defaults to 5
        n_nb (int) : radius is estimated by the mean distance to n_nb neighbours from a random small set of k_points points, n_nb defaults to 10
        shades (int) : equal to bar number in plot, if plotting is allowed, defaults to 10

    Returns:
        (list) : metric list of tuples with format (METRIC_NAME, METRIC_SCORE)
    '''
    global nrpcqa_downsized
    global nrpcqa_links
    global nrpcqa_probs
    global large
    global nrpcqa_tree

    # If small data set, use original scan, else, keep using voxelized model:
    original_pcd = pcd
    NUM_RAW_POINTS = len(pcd.points)
    if large:
        pcd =  nrpcqa_downsized
    if nrpcqa_tree:
        tree = nrpcqa_tree
    else:
        tree = o3d.geometry.KDTreeFlann(pcd)
        nrpcqa_tree = tree
    # Choose radius based on mean distance to n neighbours by a small set of k points from KDTree
    mean_dist = find_mean_distance(pcd, k_points, n_nb)
    radius = 2 * mean_dist
    nb_within_radius = [tree.search_radius_vector_3d(point, radius) for point in pcd.points]
    # Color according to number of neighbours:
    values = [len(nb_list[1]) for nb_list in nb_within_radius]
    if large:
        # Reattribute colors to full point cloud:
        down_values = np.empty(NUM_RAW_POINTS).astype(np.int32)
        for k,v in nrpcqa_links.items():
            down_values[v] = values[k]
        values = down_values
    values = list(values)
    colored_pcd, color_range = proc.color_cloud_greyscale(original_pcd, values, shades = shades)
    if snapshotting:
        inout.snapshot(colored_pcd, name = "nrpcqa_neighbourhood")
        inout.plot_values_by_color(values, color_range,  x_label="Number of neighbours", y_label="Frequency", save = True, name="nrpcqa_plot_neighbourhood")
    if visualization:
        inout.visualize(colored_pcd)
        inout.plot_values_by_color(values, color_range,  x_label="Number of neighbours", y_label="Frequency", show = True, name="nrpcqa_plot_neighbourhood")
    # Update scores:
    mapped_probs = proc.map_to_probabilities(values)
    for k,v in nrpcqa_links.items():
        for ind in v:
            nrpcqa_probs[ind] += mapped_probs[k]


def find_mean_distance(pcd, k_points = 5, n_nb = 10):
    '''
        Calculates mean distance between the n_nb neighbours of k_points random points in the point cloud (pcd).
    Args:
        pcd (type: PointCloud object) : point cloud
        k_points (int) : number of points to consider, defaults to 5
        n_nb (int) : number of neighbours to consider for each point, defaults to 10

    Returns:
        mean_dist (float) : mean distance of all neighbours to the chosen point and between them
    '''
    global nrpcqa_tree

    ks = random.sample(range(0, len(pcd.points)-1), k = k_points)
    dists = []
    # For k random points:
    for k in ks:
        # Find their n closest neighbours:
        nb, inds, _ = nrpcqa_tree.search_knn_vector_3d(pcd.points[k], n_nb)
        nb_points = [pcd.points[i] for i in inds[1:]]
        # Get all unique pairs of points:
        combs = combinations(nb_points, 2)
        # Calculate the distances within pairs:
        dists += [proc.get_l2_distance(c[0], c[1]) for c in combs]
    # Set radius to the double of the mean over all found distances:
    mean_dist =  np.mean(dists)
    return mean_dist


def nrpcqa_lof(pcd, snapshotting = False, visualization = False, k = 10, k_points =5, n_nb = 10):
    '''
        Uses LOF outlier detection to calculate metrics related to accuracy/validity. A radius search is used to determine mean number of neighbours for the LOF algorithm, which is factored for large clouds.
    Args:
        pcd (type: PointCloud object) : point cloud to analyze
        snapshotting (bool) : allow (True) taking snapshots along the process, or default (False)
        visualization (bool) : allow (True) visualization along the process, or default (False)
        k (int) : mean of number of neighbours is found for random k points, defaults to 10.
        k_points (int) : radius is estimated by the mean distance to n_nb neighbours from a random set of k_points points, k_points defaults to 5
        n_nb (int) : radius is estimated by the mean distance to n_nb neighbours from a random small set of k_points points, n_nb defaults to 10
    '''

    global nrpcqa_downsized
    global nrpcqa_links
    global nrpcqa_probs
    global large
    global nrpcqa_tree

    original_pcd = pcd
    NUM_RAW_POINTS = len(pcd.points)
    if large:
        pcd = nrpcqa_downsized
    if nrpcqa_tree:
        tree = nrpcqa_tree
    else:
        tree = o3d.geometry.KDTreeFlann(pcd)
        nrpcqa_tree = tree
    mean_dist = find_mean_distance(pcd, k_points, n_nb)
    # Given k points, get mean of number of neighbours within radius:
    random_point_inds = random.sample(range(0, len(pcd.points)-1), k)
    mean_num_nb = int(np.mean([len(tree.search_radius_vector_3d(pcd.points[i], mean_dist)[1]) for i in random_point_inds]))
    numpy_pcd = np.array(pcd.points)
    # User-allowed contamination (user has a priori knowledge about outlier proportion)
    user_contamination = input("Insert LOF Contamination. Allowed range (0.0-0.5] or ENTER for unknown:")
    if user_contamination:
        pattern = r'(0\.[0-4]+\d*)|(0\.50*$)'
        matches = re.findall(pattern, user_contamination)
        while len(matches) != 1 :
            user_contamination = input("Type error!0.0 Insert LOF Contamination. Allowed range (0.0-0.5] or ENTER for unknown:")
            if user_contamination == "":
                break
            matches = re.findall(pattern, user_contamination)
        user_contamination = float(user_contamination) if user_contamination != "" else "auto"
    else:
        # If no proportion given --> use standard LOF setting.
        user_contamination = "auto"
    if large:
        factor = 2
    else:
        factor = 1
    # Local Outlier Factor - get all points scored:
    clf = LocalOutlierFactor(n_neighbors = mean_num_nb*factor, n_jobs = -1, contamination = user_contamination)
    preds = clf.fit_predict(numpy_pcd)
    inds_outliers = np.where(preds == -1)[0] # Store outlier (i.e -1 values) indices
    if len(np.unique(preds)) != 1:
        # For large point cloud - use downsized version:
        if large:
            # Remove all point belonging to outlier voxels from original cloud:
            new_inds_full = [nrpcqa_links[i] for i in inds_outliers]
            new_inds = [j for sub in new_inds_full for j in sub] # Flatten
            noise_pcd = original_pcd.select_by_index(new_inds)
            clean_pcd = original_pcd.select_by_index(new_inds, invert=True)
        else:
            noise_pcd = pcd.select_by_index(inds_outliers)
            clean_pcd = pcd.select_by_index(inds_outliers, invert=True)
    if snapshotting:
        inout.snapshot(noise_pcd, name = "nrpcqa_lof_noise")
        inout.snapshot(clean_pcd, name = "nrpcqa_lof_clean" )
    if visualization:
        inout.visualize_differences(clean_pcd, noise_pcd)
    # Update scores:
    true_noise_ind = new_inds if large else inds_outliers
    index_array = np.zeros(nrpcqa_probs.shape[0], dtype=bool)
    index_array[true_noise_ind] = True
    nrpcqa_probs[index_array] += 1


def nrpcqa_statistical(pcd, snapshotting = False, visualization = False, k_points= 5, n_nb = 10, k = 10, std_ratio = 2.0):
    '''
        Uses statistical outlier detection to calculate metrics related to accuracy/validity. A radius search is used to determine mean number of neighbours for the detection algorithm.
    Args:
        pcd (type: PointCloud object) : point cloud to analyze
        snapshotting (bool) : allow (True) taking snapshots along the process, or default (False)
        visualization (bool) : allow (True) visualization along the process, or default (False)
        k_points (int) : radius is estimated by the mean distance to n_nb neighbours from a random set of k_points points, k_points defaults to 5
        n_nb (int) : radius is estimated by the mean distance to n_nb neighbours from a random small set of k_points points, n_nb defaults to 10
        k (int) : mean of number of neighbours is found for random k points, defaults to 10
        std_ratio (float) : standard deviation ratio used in statistical removal (lower is more aggressive), defaults to 2.0
    '''

    global nrpcqa_downsized
    global large
    global nrpcqa_tree
    global nrpcqa_links
    global nrpcqa_probs

    original_pcd = pcd
    NUM_RAW_POINTS = len(pcd.points)
    if large:
        pcd = nrpcqa_downsized
    if nrpcqa_tree:
        tree = nrpcqa_tree
    else:
        tree = o3d.geometry.KDTreeFlann(pcd)
        nrpcqa_tree = tree
    mean_dist = find_mean_distance(pcd, k_points, n_nb)
    # Given k points, get mean of number of neighbours within radius:
    random_point_inds = random.sample(range(0, len(pcd.points)-1), k)
    # Use mean number of neighbours as number of neighbors around the target point.
    mean_num_nb = int(np.mean([len(tree.search_radius_vector_3d(pcd.points[i], mean_dist)[1]) for i in random_point_inds]))
    # Remove points distant from their neighbors in average, standard deviation ratio (lower -> aggressive)
    cl, ind = proc.remove_outliers(pcd, num_nb=mean_num_nb, std_ratio=std_ratio)
    if large:
        # Remove all point belonging to outlier voxels from original cloud:
        actual_inds_full = [nrpcqa_links[i] for i in ind]
        actual_inds = [j for sub in actual_inds_full for j in sub] # Flatten
        clean_pcd = original_pcd.select_by_index(actual_inds)
        noise_pcd = original_pcd.select_by_index(actual_inds, invert=True)
    else:
        clean_pcd = pcd.select_by_index(ind)
        noise_pcd = pcd.select_by_index(ind, invert=True)
    if snapshotting:
        inout.snapshot(noise_pcd, name = "nrpcqa_stat_noise")
        inout.snapshot(clean_pcd, name = "nrpcqa_stat_clean" )
    if visualization:
        inout.visualize_differences(clean_pcd, noise_pcd)
    # Update scores:
    true_clean_ind = actual_inds if large else ind
    index_array = np.ones(nrpcqa_probs.shape[0], dtype=bool)
    index_array[true_clean_ind] = False
    nrpcqa_probs[index_array] += 1
    # Construct metric:
    #*NUM_STATISTICAL_OUTLIERS = len(noise_pcd.points)
    #*return [("METRIC_RATIO_STAT_V",NUM_STATISTICAL_OUTLIERS/NUM_RAW_POINTS)]


def cleaning(snapshotting = False, visualization = False, output_name = "denoised"):
    '''
        Cleans point cloud of noise and outliers given global probabilities that have been updated in previous nrpcqa steps. Saves model.
    Args:
        snapshotting (bool) : allow (True) taking snapshots along the process, or default (False)
        visualization (bool) : allow (True) visualization along the process, or default (False)
        output_name (str) : name of denoised point cloud
    '''
    global nrpcqa_clean
    global nrpcqa_probs
    global original
    global saving

    # Scale probabilities to [0,1]:
    scaled_probs = proc.minmax_scale(nrpcqa_probs)
    # Use IQR rule as threshold:
    Q1 = np.percentile(scaled_probs, 25) # First quartile
    Q3 = np.percentile(scaled_probs, 75) # Third quartile
    IQR = Q3 - Q1
    noise_threshold = Q3 + 1.5 * IQR
    aggresive_noise_threshold = Q3 + IQR
    clean_indices = np.where(scaled_probs <= aggresive_noise_threshold)[0]
    clean = proc.get_cloud_by_index(original, clean_indices)
    # Paint uniformly:
    clean.paint_uniform_color([0, 0.5, 0])
    if snapshotting:
        inout.snapshot(clean, name="nrpcqa_denoised")
    if visualization:
        inout.visualize(clean)
    if saving:
        # Save denoised cloud locally with proper extension:
        full_name = output_name+".pcd"
        inout.write_cloud(full_name, nrpcqa_clean)
    # Save globally too:
    nrpcqa_clean = clean


@lru_cache
def process_without_reference(input_file: str):
    '''
        Point Cloud Data Validation - without reference
    Args:
        input_file (str) : path of point cloud to analyse
    '''
    # TODO: Add more NR-PCQA approaches  ...

    # Global variables to update:
    global nrpcqa_downsized
    global nrpcqa_links
    global nrpcqa_clean
    global nrpcqa_voxel_size
    global nrpcqa_probs
    # Keep boolean check for large detected cloud:
    global large
    # Keep boolean check for visualization/snapshots:
    global snapshotting
    global visualization
    global original

    # Keep track of processing time:
    start_time = time.time()

    print("Action in progress: reading point cloud...") # Order matters
    # Read point cloud from path:
    original = inout.read_cloud(input_file)
    NUM_RAW_POINTS = len(original.points)
    # Cloud to be worked on:
    pcd = copy.deepcopy(original)
    if snapshotting:
        inout.snapshot(original, name = "nrpcqa_original")
    if visualization:
        inout.visualize(original)

    # Keep boolean check for large scan (over 0.5 mil):
    if len(original.points) > 500_000:
        large = True

    print("Action in progress: instantiate score tracking...")
    # Keep point-wise distortion score for outlier detection:
    point_probs = np.zeros(NUM_RAW_POINTS)
    nrpcqa_probs = point_probs

    print("Action in progress: downsize/voxelize point cloud...")
    nrpcqa_downsize(pcd, snapshotting = snapshotting, visualization = visualization) # Must be first

    print("Action in progress: density assessment based on voxelization...")
    nrpcqa_voxelization(pcd, snapshotting = snapshotting, visualization = visualization)

    print("Action in progress: density assessment based on voxel modelling...")
    nrpcqa_modelling(pcd, snapshotting = snapshotting, visualization = visualization)

    print("Action in progress: density assessment followed by statistical noise removal based on radius neighbourhood,...")
    nrpcqa_radius_nb(pcd, snapshotting = snapshotting, visualization = visualization)

    print("Action in progress: density-based noise removal based on Local Outlier Factor...")
    nrpcqa_lof(pcd, snapshotting = snapshotting, visualization = visualization)

    print("Action in progress: statistical removal of noise based on distance to neighbours...")
    nrpcqa_statistical(pcd, snapshotting = snapshotting, visualization = visualization)

    print("Action in progress: denoising original point cloud...") # Must be last
    cleaning(snapshotting = snapshotting, visualization = visualization)

    # End processing time
    end_time = time.time()
    nrpcqa_time = datetime.timedelta(seconds=end_time - start_time)
    print("Processing time (NRPCQA):" +  str(nrpcqa_time))
    

def weight_quality(metrics, weights):
    '''
        Maps weighted metrics to actual quality.
    Args:
        metrics (list) : metric scores in following order ACCURACY/VALIDITY/(optional)COMPLETENESS
        weights (list) : list of weights for  ACCURACY/VALIDITY/(optional)COMPLETENESS respectively, defaults to equally weighted
    Returns:
        quality (str) : either "Bad Quality", "Mixed Quality", "Good Quality", depending on value
    '''
    # Get the boundary values for each quality score:
    best_score = sum(weights)
    worst_score = 0
    range_boundaries = np.linspace(worst_score, best_score, 4)
    ranges = [(range_boundaries[i], range_boundaries[i+1]) for i in range(len(range_boundaries)-1)]
    # Return quality according to the range of the score:
    actual_score = (metrics[0] * weights[0]) + (metrics[1] * weights[1]) + (metrics[2] * weights[2])
    for i in range (len(ranges)):
        if actual_score <= ranges[i][1] and actual_score >= ranges[i][0]:
            if i == 0:
                return "Bad Quality"
            elif i == 1:
                return "Mixed Quality"
            else:
                return "Good Quality"


def quality_calculation(weights, reference = False, output = "metrics"):
    '''
        Calculates accuracy, validity, completeness based on global scores that have been updated in previous nrpcqa steps. Saves results to JSON. 
    Args:
        weights (list) : list of weights for ACCURACY/COMPLETENESS/VALIDITY respectively, defaults to equally weighted
        reference (bool): boolean to check if NRPCQA or RPCQA.
    '''
    global nrpcqa_probs
    global original

    # TODO: Find rule for differencing between noise and outliers in literature or explain choice.
    scaled_probs = proc.minmax_scale(nrpcqa_probs)
    # Use IQR rule as threshold:
    Q1 = np.percentile(scaled_probs, 25) # First quartile
    Q3 = np.percentile(scaled_probs, 75) # Third quartile
    IQR = Q3 - Q1
    noise_threshold = Q3 + 1.5 * IQR
    # Construct metrics (both outliers and noise):
    unclean_values = [x for x in scaled_probs if x > noise_threshold]
    unclean_Q1 = np.percentile(unclean_values, 25) # First quartile
    unclean_Q3 = np.percentile(unclean_values, 75) # Third quartile
    unclean_IQR = unclean_Q3 - unclean_Q1
    outlier_threshold = unclean_Q3 + 1.5 * unclean_IQR
    # Fetch clouds by indices:
    noise_indices = np.where((scaled_probs > noise_threshold) & (scaled_probs <= outlier_threshold))[0]
    noise = proc.get_cloud_by_index(original, noise_indices)
    outlier_indices = np.where(scaled_probs > outlier_threshold)[0]
    outlier = proc.get_cloud_by_index(original, outlier_indices)
    # Get number of highly possible noise/outliers:
    NUM_NOISE = len(noise.points)
    NUM_OUTLIERS = len(outlier.points)
    NUM_RAW_POINTS = len(original.points)
    if visualization:
        inout.visualize_differences(noise, outlier)
    if reference:
        # Calculate both accuracy, validity, completeness:
        # TODO : Check notes:
        ACCURACY = 1.0 - (NUM_NOISE/NUM_RAW_POINTS)
        VALIDITY = 1.0 - (NUM_OUTLIERS/NUM_RAW_POINTS)
        COMPLETENESS = rpcqa_completeness_ratio
    else:
        ACCURACY = 1.0 - (NUM_NOISE/NUM_RAW_POINTS)
        VALIDITY = 1.0 - (NUM_OUTLIERS/NUM_RAW_POINTS)
        COMPLETENESS = 1.0 # We do not know - so we assume perfect (no missing values)
    metrics = [ACCURACY, VALIDITY, COMPLETENESS]
    # Weight results and fetch overall quality metric:
    QUALITY = weight_quality(metrics, weights)
    # Save to JSON:
    json_dict = {}
    json_dict["ACCURACY"] = ACCURACY
    json_dict["VALIDITY"] = VALIDITY
    json_dict["COMPLETENESS"] = COMPLETENESS
    json_dict["QUALITY"] = QUALITY
    inout.write_to_json(json_dict, output)


def stitch(dir_path:str, *args):
    '''
        Aggregates a directory of .ply files into single PointCloud object.
        Outputs result to same directory.
        Assumes same coordinates/scale between point clouds.
    Args:
        dir_path (str) : path of directory containing .ply files
        *args (str) : name of output file - optional
    '''
    # TODO: Multiway registration

    # Get all paths of .ply files from given directory:
    ply_files = []
    os.chdir(dir_path)
    for file in glob.glob("*.ply"):
        ply_files.append(file)

    # Check that .ply files are found in directory:
    num_plys = len(ply_files)
    if num_plys == 0:
        raise FileNotFoundError("No .ply files found in directory!")

    # Stitch :
    stitched_pcd = proc.stitch_clouds(ply_files)

    # If name specified, place in same directory with new name:
    if len(args) != 0:
        inout.write_cloud(dir_path + "/" + args[0]+ ".ply", stitched_pcd)
    # Else, place in same directory with default name:
    else:
        name = "/3ddava_Stitched_"+str(num_plys)+"_Clouds.ply"
        inout.write_cloud(dir_path + name, stitched_pcd)


def main(argv = None):
    ''' Parsing command line arguments
    Args:
        argv (list): list of arguments
    '''
    global visualization
    global snapshotting
    global saving

    # Top-level parser:
    parser = argparse.ArgumentParser(add_help=False)

    # No arguments:
    if argv is None:
        argv = sys.argv[1:]
        if not len(argv):
            parser.error('Insufficient arguments provided.')

    # Subparsers for each functionality (processing and stitching):
    subparsers = parser.add_subparsers(help='Help', dest='which')

    # Processing subparser:
    processing_input = subparsers.add_parser('processing', help='Processing Point Cloud Help')
    # Obligatory argument:
    processing_input.add_argument("cloud_file", help = "Input point cloud file to be analyzed (format: .ply).")
    # Optional arguments:
    processing_input.add_argument("-r", "--reference",  help = "The reference file (format: .stl) - optional.")
    processing_input.add_argument("-o", "--output",  help = "Filename of output file. No file extension needed.")
    # Add visualization / snapshotting as argument:
    processing_input.add_argument("-vis", "--visualize",  help = "Allow visualization of process steps.",  action="store_true")
    processing_input.add_argument("-snap", "--snapshot",  help = "Allow snapshotting of process steps." ,  action="store_true")
    # Add saving denoised point cloud as argument
    processing_input.add_argument("-save", "--save",  help = "Save denoised point cloud as (.pcd)." ,  action="store_true")

    # Stitching subparser:
    stitching = subparsers.add_parser('stitching', help='Stitching Files Help')
    # Obligatory argument:
    stitching.add_argument("directory_path", type=proc.is_dir, help = "Absolute path of directory containing .ply clouds.")
    # Optional arguments:
    stitching.add_argument("-o", "--output",  help = "Filename of output file. No file extension needed.")
    # TODO: Add saving denoised point cloud as an option under processing.

    # TODO: Validate output filename (e.g. pathvalidate package, regex)
    args = parser.parse_args(argv)
    # Parse stitching:
    if args.which == "stitching":
        try:
            print("Stitching .ply files from directory ... ")
            directory_path = args.directory_path
            if args.output:
                output_file = args.output
                stitch(directory_path, output_file)
            else:
                stitch(directory_path)
        except Exception as e:
            print(str(e) + " Action dropped: stitching clouds.  Use -h, --h or --help for more information.")
    # Parse processing:
    else:
        try:
            while True:
                try:
                    weight_accuracy = float(input("Accuracy weight/importance: "))
                    weight_validity = float(input("Validity weight/importance: "))
                    weight_completeness = float(input("Completeness weight/importance: "))
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid float or integer.")
            weights = [weight_accuracy,  weight_validity,  weight_completeness]
            # Validate format:
            input_file = args.cloud_file
            if not proc.is_path(input_file):
                raise FileNotFoundError("Specified input cloud file was not found on given path.")
            if not input_file.endswith(".ply"):
                raise TypeError("Input file must be in .ply format.")
            # If present, process with reference:
            visualization = args.visualize
            snapshotting = args.snapshot
            saving = args.save
            if args.reference:
                print("Processing with reference ... ")
                reference_file = args.reference
                if not proc.is_path(reference_file):
                    raise FileNotFoundError("Specified reference/CAD file was not found on given path.")
                # Validate format:
                if not reference_file.endswith(".stl"):
                    raise TypeError("Reference/CAD file must be in .stl format.")
                process_with_reference(input_file, reference_file)
                # If output filename is given, use as output file name:
                if args.output:
                    output_file = args.output
                    quality_calculation(weights, reference=True, output = output_file)
                else:
                    quality_calculation(weights, reference=True)
            else:
                print("Processing without reference ... ")
                process_without_reference(input_file)
                # If output filename is given, use as output file name:
                if args.output:
                    output_file = args.output
                    quality_calculation(weights, output = output_file)
                else:
                    quality_calculation(weights)
        except Exception as e:
            print(str(e) + " Action dropped: processing clouds.  Use -h, --h or --help for more information.")

if __name__ == '__main__':
    main()
