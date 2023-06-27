# -*- coding: utf-8 -*-
'''This file should be invoked as python -m 3D_DaVa'''

### IMPORTS ###
import os
import glob
from . import processing as proc
from . import alignment as ali
from . import io
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
from itertools import combinations
from sklearn.neighbors import LocalOutlierFactor
import time


# Global variables - used in No-Reference and Reference Processing:
nrpcqa_downsized = None
nrpcqa_links = None
nrpcqa_clean = None
nrpcqa_voxel_size = None
nrpcqa_probs = None
nrpcqa_tree = None
large = False
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
    # TODO: Gatekeep snapshotting/visualization with one boolean
    # TODO: Make possible changing between io.visualize and io.snapshot with one boolean
    # TODO: Try other R-PCQA approaches (see literature-Graph Similarity, total area covered, completeness) ...
    # TODO: Clean functions -> move to .py files
    # TODO: Register using correspondence sets (ISS/Curvature) or graph
    global nr_process_downsized_pcd
    global nr_process_links
    global nr_process_results
    global nr_process_cleaned_pcd
    global nr_process_voxel_size

    # Keep track of processing time:
    start_time = time.time()

    # Read original point cloud:
    original_pcd = io.read_cloud(input_file)
    NUM_RAW_POINTS_PCD = len(original_pcd.points)

    print("Action in progress: process cloud without reference and denoise cloud...")
    # Cloud to be worked on (cleaned cloud):
    # TODO: User can choose whether or not cleaning happens
    process_without_reference(input_file)
    pcd = nr_process_cleaned_pcd
    # pcd = copy.deepcopy(original_pcd)
    # io.visualize(pcd)

    # Keep downsizing check for large point clouds (over 0.5mil):
    large = False
    if NUM_RAW_POINTS_PCD > 500_000:
        large = True


    print("Action in progress: reading and sampling reference...")
    # Read reference:
    original_ref = io.read_mesh(reference_file)
    # TODO: Intelligent sampling rate
    if large:
        factor = 1
    else:
        factor = 5
    sampling_rate = NUM_RAW_POINTS_PCD * factor
    # Sample CAD, turning mesh into point cloud:
    ref_pcd = proc.create_uniform_sampled_cloud_from_mesh(original_ref, nr_points = sampling_rate)
    # Reference to be worked on:
    ref = copy.deepcopy(ref_pcd)
    NUM_RAW_POINTS_REF = len(ref.points)
    # io.snapshot(ref, name="sampling", ref = True)


    print("Action in progress: voxelize reference...")
    if large:
        # Downsize given boundary box constraint - ref:
        voxel_size_ref = round(max(ref.get_max_bound() - ref.get_min_bound()) * 0.005, 8)
        down_ref, corr_inds_ref = proc.downsample_and_trace_cloud(ref, voxel_size_ref)
        NUM_VOXELS_REF = len(down_ref.points)
        voxel_size_pcd = round(max(pcd.get_max_bound() - pcd.get_min_bound()) * 0.005, 8)
        down_pcd, corr_inds_pcd = proc.downsample_and_trace_cloud(pcd, voxel_size_pcd)
        NUM_VOXELS_PCD = len(down_pcd.points)
        # Save links for ref:
        ref_voxel_inds = range(len(corr_inds_ref))
        ref_actual_inds = [list(corr_inds_ref[i]) for i in ref_voxel_inds]
        ref_links = dict(zip(ref_voxel_inds, ref_actual_inds))
        # Save links for pcd:
        pcd_voxel_inds = range(len(corr_inds_pcd))
        pcd_actual_inds = [list(corr_inds_pcd[i]) for i in pcd_voxel_inds]
        pcd_links = dict(zip(pcd_voxel_inds, pcd_actual_inds))
        # Snapshot:
        io.snapshot(down_pcd, name ="downsized_pcd", ref = True)
        io.snapshot(down_ref, name ="downsized_ref", ref = True)


    print("Action in progress: scaling to same size...")
    # Use downsized versions if the clouds are too large:
    if large:
        pcd = down_pcd
        ref = down_ref
    # Rescale pcd to mesh size (or the other way around):
    pcd_max_bound = pcd.get_max_bound()
    pcd_min_bound = pcd.get_min_bound()
    pcd_dims = pcd_max_bound - pcd_min_bound
    ref_max_bound = ref.get_max_bound()
    ref_min_bound = ref.get_min_bound()
    ref_dims = ref_max_bound - ref_min_bound
    # Check which boudnaring box is bigger by volume:
    vol1 = np.prod(pcd_dims)
    vol2 = np.prod(ref_dims)
    if vol1 > vol2:
        scaling_factor = max(pcd_dims) / max(ref_dims)
        ref.scale(scaling_factor, center=ref.get_center())
    else:
        scaling_factor = max(ref_dims) / max(pcd_dims)
        pcd.scale(scaling_factor, center=pcd.get_center())
    # io.visualize_differences(pcd, ref)


    print("Action in progress: fast global alignment using feature matching...")
    # TODO: Genetic optimization of solutions - or similar (remove manual)
    # TODO: Non-blocking visualization (http://www.open3d.org/docs/latest/tutorial/visualization/non_blocking_visualization.html)
    if large:
        voxel_size = voxel_size_pcd
    distance = pcd.compute_point_cloud_distance(ref)
    mean_distance = np.mean(distance)
    threshold = 0.1 * mean_distance
    trans_path = "trans.pickle"
    if os.path.isfile(trans_path):
        with open('trans.pickle', 'rb') as handle:
            global_trans = pickle.load(handle) # Deserialize
    else:
        max_dist = proc.get_l2_distance(pcd.get_min_bound(), pcd.get_max_bound())/2
        iter = 200                           # The larger the better -> but more time use
        distance_factor = 0.8                # ONLY LARGE: Try: [0.005, 0.05, 0.5, 1.0, 2.0]
        normal_factor = 5 if large else 0.1   # Try [3, 5, 8] (bigger than 1) and [0.005, 0.05, 0.5, 1.0, 5]
        feature_factor = 5 if large else 0.5  # Try [3, 5, 8] (bigger than 1) and [0.005, 0.05, 0.5, 1.0, 5]
        max_nn_n = 50                         # Try [5, 10, 50, 100, 150]
        max_nn_f = 50                      # Try [5, 10, 50, 100, 150]
        normal_radius = voxel_size * normal_factor if large else max_dist * normal_factor
        feature_radius = voxel_size * feature_factor if large else max_dist * feature_factor
        # threshold = voxel_size * distance_factor if large else max_dist/len(pcd.points)
        solution = [iter, normal_radius, feature_radius, max_nn_n, max_nn_f]
        global_result = ali.global_fast_alignment(pcd, ref, solution, threshold)
        global_trans = global_result.transformation
        eva = ali.evaluate(pcd, ref, threshold, global_trans)
        fitness = eva.fitness
        inlier_rmse = eva.inlier_rmse
        # pcd.transform(global_trans)
        # io.visualize_differences(pcd, ref)

    print("Action in progress: ICP local alignment P2P...")
    icp_trans = ali.icp_P2P_registration(pcd, ref, threshold, global_trans)
    pcd.transform(icp_trans)
    # io.visualize_differences(pcd, ref)

    if large:
        pcd = original_pcd.transform(icp_trans)
    # io.visualize_differences(pcd, ref)


    print("Action in progress: removing outliers (background elements)...")
    # TODO: For large, use links:
     # Calculate distances between downsized digitized scan and reference
    distances = pcd.compute_point_cloud_distance(ref)
    # fig = plt.figure(figsize =(10, 7))
    plt.hist(distances,  bins='auto')
    plt.title("Distances between digitized model and reference - Histogram")
    plt.show()
    # Use statistical outlier detection:
    distances = np.asarray(distances)
    q25, q75 = np.percentile(distances, [25 ,75]) # Get 1st and 3rd quartile
    iqr = q75 - q25                               # Calculate interquartile range
    background_threshold = q75 + 1.5 * iqr        # Higher bound
    ind = np.where(distances >= background_threshold)[0]
    background_pcd = pcd.select_by_index(ind)
    no_background_pcd = pcd.select_by_index(ind, invert=True)
    io.visualize_differences(no_background_pcd, background_pcd)
    # Get metric:
    NUM_BACKGROUND = len(background_pcd.points)
    METRIC_RATIO_OUTLIER_BACKGROUND = NUM_BACKGROUND/NUM_RAW_POINTS_PCD
    # Point2point:
    # METRIC_P2Point_MEAN = np.mean(distances)
    # METRIC_P2Point_STD = np.std(distances)
    # METRIC_P2Point_VAR = np.var(distances)

    # TODO: Difference meshes voxel + ref
    # TODO: Go back to original pcd:
    pcd = copy.deepcopy(original_pcd)
    pcd.scale(scaling_factor, center=pcd.get_center())
    pcd.transform(icp_trans)

    print("Action in progress: calculate PCA properties and metrics...")
    components_pcd, exp_var_ratio_pcd, exp_var_pcd, transformed_pcd = proc.principal_component_analysis(np.array(pcd.points))
    components_ref, exp_var_ratio_ref, exp_var_ref, transformed_ref = proc.principal_component_analysis(np.array(ref.points))
    # Get metrics:
    METRIC_TRUENESS_PCA_1 = proc.cosine_similarity(components_pcd[0], components_ref[0])
    METRIC_TRUENESS_PCA_2 = proc.cosine_similarity(components_pcd[1], components_ref[1])
    METRIC_TRUENESS_PCA_3 = proc.cosine_similarity(components_pcd[2], components_ref[2])


    print("Action in progress: calculate distance-based metrics...")
    # N-noisy:
    user_N = 0.05 # TODO: Ask user for input
    N_noisy = True if len(distances[distances <= user_N]) == 0 else False
    # Get metric:
    if N_noisy:
        METRIC_N_NOISY = 1
    else:
        METRIC_N_NOISY = 0

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
    signed_distances = distance
    METRIC_P2Plane_MEAN = np.mean(signed_distances)
    METRIC_P2Plane_STD = np.std(signed_distances)
    METRIC_P2Plane_VAR = np.var(signed_distances)

    # Check triangle occurence:
    # closest_triangles = closest_points['geometry_ids'].numpy()
    # counts = Counter(closest_triangles)
    # METRIC_MEAN_NR_OF_POINTS_IN_TRIANGLE = np.mean(list(counts.values()))

    # End processing time
    end_time = time.time()
    METRIC_PROCESS_TIME = datetime.timedelta(seconds=end_time - start_time)

    # TODO: Update the no-reference dict instead:
    TRUENESS = METRIC_TRUENESS_PCA_1 + METRIC_TRUENESS_PCA_2 + METRIC_TRUENESS_PCA_3
    NOISY = (1-METRIC_RATIO_OUTLIER_BACKGROUND) + METRIC_N_NOISY
    TOTAL = NOISY + TRUENESS
    QUALITY = calculate_quality(TOTAL, 5)


    # Write metrics to JSON file:
    json_dict = {"NOISE - Outliers: Ratio background-classified points": str(round(METRIC_RATIO_OUTLIER_BACKGROUND,2)),
                 "NOISE - Outliers: N-noisiness": str(METRIC_N_NOISY),
                 "TRUENESS - Cosine similarity PCA between SCAN vs REF (1)": str(round(METRIC_TRUENESS_PCA_1,2)),
                 "TRUENESS - Cosine similarity PCA between SCAN vs REF (2)": str(round(METRIC_TRUENESS_PCA_2,2)),
                 "TRUENESS - Cosine similarity PCA between SCAN vs REF (3)": str(round(METRIC_TRUENESS_PCA_3,2)),
                 "NOISY SCORE": str(round(NOISY, 2)),
                 "TRUENESS SCORE" : str(round(TRUENESS, 2)),
                 "TOTAL SCORE": str(round(TOTAL, 2)),
                 "QUALITY": QUALITY,
                 "PROFILING (processing time)" : str(METRIC_PROCESS_TIME)
    }
    print("Action in progress: writting to .json file ...")
    if len(args) != 0:
        io.write_to_json(json_dict, args[0], ref = True)
    else:
        io.write_to_json(json_dict, ref = True)


#____________________________________NR-PCQA____________________________________
def nrpcqa_downsize(pcd, snapshotting = False, visualization = False):
    """
    TODO
    """
    global nrpcqa_downsized
    global nrpcqa_voxel_size
    global nrpcqa_links
    global large
    # We downsize big models (over 0.5mil) to allow faster processing
    if large:
        constraint = 0.0025
    else:
        constraint = 0.025
    # Downsize given boundary box-based constraint:
    voxel_size = round(max(pcd.get_max_bound() - pcd.get_min_bound()) * constraint, 8)
    down, corr_inds = proc.downsample_and_trace_cloud(pcd, voxel_size)
    # REMOVE: Save links between voxels and within-voxel points:
    # linked_inds = [[i, list(corr_inds[i])] for i in range(len(corr_inds))]
    # linked_inds_dict = dict([(i, list(corr_inds[i])) for i in range(len(corr_inds))])
    # Save for reference-processing:
    pcd_voxel_inds = range(len(corr_inds))
    pcd_actual_inds = [list(corr_inds[i]) for i in pcd_voxel_inds]
    pcd_links = dict(zip(pcd_voxel_inds, pcd_actual_inds))
    # Visual outputs:
    if snapshotting:
        io.snapshot(down, name = "downsized")
    if visualization:
        io.visualize(down)
    # Update global:
    nrpcqa_downsized = down
    nrpcqa_voxel_size = voxel_size
    nrpcqa_links = pcd_links
    # NUM_VOXELS = len(down.points)

def nrpcqa_modelling(pcd, snapshotting = False, visualization = False):
    """
    TODO
    """
    global nrpcqa_voxel_size
    global nrpcqa_links
    global nrpcqa_downsized
    global nrpcqa_probs

    # Reference: https://towardsdatascience.com/how-to-automate-voxel-modelling-of-3d-point-cloud-with-python-459f4d43a227
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size = nrpcqa_voxel_size)
    voxels = voxel_grid.get_voxels()
    # Create mesh by turning voxels into 3D cubes:
    v_mesh = o3d.geometry.TriangleMesh()
    for i in range(len(voxels)):
        cube=o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        cube.paint_uniform_color(voxels[i].color)
        voxel_coord = voxels[i].grid_index
        cube.translate(voxel_coord, relative=False)
        v_mesh += cube
    v_mesh.translate([0.5, 0.5, 0.5], relative=True)
    v_mesh.scale(nrpcqa_voxel_size, [0, 0, 0])
    v_mesh.translate(voxel_grid.origin, relative=True)
    v_mesh.merge_close_vertices(0.0000001)
    v_mesh.compute_vertex_normals()
    # Clustering connected triangles ...
    triangle_cluster_ids, cluster_n_triangles, cluster_area = v_mesh.cluster_connected_triangles()
    triangle_cluster_ids = np.asarray(triangle_cluster_ids)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    # Remove small clusters (small = less than half the number from the biggest cluster):
    biggest_cluster_ind = np.argmax(cluster_n_triangles)
    constraint = int(cluster_n_triangles[biggest_cluster_ind] * 0.5)
    small_cluster_triangles = cluster_n_triangles[triangle_cluster_ids] <= constraint
    big_cluster_triangles = cluster_n_triangles[triangle_cluster_ids] > constraint
    high_density_mesh = copy.deepcopy(v_mesh)
    high_density_mesh.remove_triangles_by_mask(small_cluster_triangles)
    high_density_mesh.remove_unreferenced_vertices()
    high_density_mesh.paint_uniform_color([0, 0.5, 0]) # Green
    low_density_mesh = copy.deepcopy(v_mesh)
    low_density_mesh.remove_triangles_by_mask(big_cluster_triangles)
    low_density_mesh.remove_unreferenced_vertices()
    low_density_mesh.paint_uniform_color([1, 0, 0]) # Red
    full_mesh = high_density_mesh + low_density_mesh
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = full_mesh.vertices
    full_pcd.colors = full_mesh.vertex_colors
    # Outlier Detection: +1 for points in low-connected areas
    tree = o3d.geometry.KDTreeFlann(full_pcd)
    np_down = np.array(nrpcqa_downsized.points)
    noise_inds = []
    for i in range(len(np_down)):
        # For each voxel, get closest point in full_pcd
        [k, idx, _] = tree.search_knn_vector_3d(np_down[i], 1)
        # Get color of the point in full_pcd:
        point_color = list(np.asarray(full_pcd.colors)[idx[0]])
        # If point is red, set probability higher for actual points:
        if point_color == [1., 0., 0.]:
            # Get actual point cloud indices belonging to red-paired voxels:
            actual_inds = nrpcqa_links[i]
            for ind in actual_inds:
                noise_inds.append(ind)
    index_array = np.zeros(nrpcqa_probs.shape[0], dtype=bool)
    index_array[noise_inds] = True
    nrpcqa_probs[index_array] += 1
    # Print the index of the closest triangle for each point in the point cloud
    if snapshotting:
        io.snapshot(full_mesh, name = "modelling")
    if visualization:
        io.visualize(full_mesh)
    # Metrics:
    NUM_RAW_TRIANGLES = len(v_mesh.triangles)
    NUM_HIGH_CONNECTION_TRIANGLES = len(high_density_mesh.triangles)
    NUM_LOW_CONNECTION_TRIANGLES =  len(low_density_mesh.triangles)
    # EXTRA: Mesh surface area (sum of all triangle areas) - voxelized:
    # MESH_AREA = v_mesh.get_surface_area()
    return [("METRIC_RATIO_TRIANGLE_DENSITY", NUM_LOW_CONNECTION_TRIANGLES/NUM_RAW_TRIANGLES)]

def nrpcqa_voxelization(pcd, snapshotting = False, visualization = False, shades = 25):
    """
    TODO
    """
    global nrpcqa_links
    global nrpcqa_downsized
    global nrpcqa_probs
    # Get number of points per voxel (with and without index):
    values = [len(v) for v in list(nrpcqa_links.values())]
    # Color voxelized cloud according to number of neighbours:
    colored_down, color_range = proc.color_cloud_greyscale(nrpcqa_downsized, values, shades = shades)
    if snapshotting:
        io.snapshot(colored_down, name = "voxelized")
        io.plot_values_by_color(values, color_range, x_label="Number contained points per voxel", y_label="Number of voxels", save = True, name="plot_voxelization")
    if visualization:
        io.visualize(colored_down)
        io.plot_values_by_color(values, color_range, x_label="Number contained points per voxel", y_label="Number of voxels", show=True, name="plot_voxelization")
    # Get all indexes of red voxels:
    RED = np.where(np.all(np.array(colored_down.colors)==np.array([1.0,0.0,0.0]),axis=1))[0]
    # Get number of red voxels:
    NUM_RED_VOXELS = RED.shape[0]
    NUM_VOXELS = len(nrpcqa_downsized.points)
    # Get number of points belonging to red voxels:
    NUM_RED_POINTS = np.sum([len(nrpcqa_links[i]) for i in list(RED)])
    NUM_RAW_POINTS = len(pcd.points)
    # Outlier Detection: +1 for points belonging to red voxel:
    actual_inds_red = [nrpcqa_links[i] for i in list(RED)]
    actual_inds_flat = [j for sub in actual_inds_red for j in sub]
    index_array = np.zeros(nrpcqa_probs.shape[0], dtype=bool)
    index_array[actual_inds_flat] = True
    nrpcqa_probs[index_array] += 1
    # Outlier Detection: increase probability given lower values:
    mapped_probs = map_to_probabilities(values)
    for k,v in nrpcqa_links.items():
        for ind in v:
            nrpcqa_probs[ind] += mapped_probs[k]
    return [("METRIC_RATIO_RED_VOXELS",NUM_RED_VOXELS/NUM_VOXELS),("METRIC_RATIO_RED_POINTS", NUM_RED_POINTS/NUM_RAW_POINTS)]

def nrpcqa_radius_nb(pcd, snapshotting = False, visualization = False, k_points = 5, n_nb= 10, shades = 10):
    """
    TODO
    """
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
    # MEAN_NEIGHBOURHOOD_SIZE = np.mean(values)
    values = list(values)
    colored_pcd, color_range = proc.color_cloud_greyscale(original_pcd, values, shades = shades)
    if snapshotting:
        io.snapshot(colored_pcd, name = "neighbourhood")
        io.plot_values_by_color(values, color_range,  x_label="Number of neighbours", y_label="Frequency", save = True, name="plot_neighbourhood")
    if visualization:
        io.visualize(colored_pcd)
        io.plot_values_by_color(values, color_range,  x_label="Number of neighbours", y_label="Frequency", show = True, name="plot_neighbourhood")
    # Outlier Detection: increase probability given lower values:
    mapped_probs = map_to_probabilities(values)
    for k,v in nrpcqa_links.items():
        for ind in v:
            nrpcqa_probs[ind] += mapped_probs[k]
    # Can eventually set a stop here if no removal is necessary.
    # Note: Normal/Gaussian distribution -> Can be used for removing outliers on the lower bound
    Q1 = np.percentile(values, 25) # First quartile
    Q3 = np.percentile(values, 75) # Third quartile
    IQR = Q3 - Q1
    low_threshold = Q1 - 1.5 * IQR
    # Remove all noise from pcd:
    ind = np.where(values <= low_threshold)[0]
    noise_pcd = colored_pcd.select_by_index(ind)
    clean_pcd = colored_pcd.select_by_index(ind, invert=True)
    if snapshotting:
        io.snapshot(noise_pcd, name = "nb_noise")
        io.snapshot(clean_pcd, name = "nb_clean")
    if visualization:
        io.visualize_differences(clean_pcd, noise_pcd)
    # Outlier Detection: +1 for points statistically classified as noise:
    index_array = np.zeros(nrpcqa_probs.shape[0], dtype=bool)
    index_array[ind] = True
    nrpcqa_probs[index_array] += 1
    # Construct metric:
    NUM_NB_OUTLIERS = len(noise_pcd.points)
    return  [("METRIC_RATIO_RADIUS", NUM_NB_OUTLIERS/NUM_RAW_POINTS)]

def find_mean_distance(pcd, k_points, n_nb):
    """
    TODO
    """
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
    """
    TODO
    """
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
        NUM_LOF_OUTLIERS = len(noise_pcd.points)
    else:
        NUM_LOF_OUTLIERS = 0
    if snapshotting:
        io.snapshot(noise_pcd, name = "lof_noise")
        io.snapshot(clean_pcd, name = "lof_clean" )
    if visualization:
        io.visualize_differences(clean_pcd, noise_pcd)
    # Outlier Detection: +1 for points classified as noise:
    true_noise_ind = new_inds if large else inds_outliers
    index_array = np.zeros(nrpcqa_probs.shape[0], dtype=bool)
    index_array[true_noise_ind] = True
    nrpcqa_probs[index_array] += 1
    return [("METRIC_RATIO_LOF",NUM_LOF_OUTLIERS/NUM_RAW_POINTS)]


def nrpcqa_statistical(pcd, snapshotting = False, visualization = False, k_points= 5, n_nb = 10, k = 10, std_ratio = 2.0):
    """
    TODO
    """
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
    # Remove points distant from their neighbors in average, standard deviation ratio (lower -> aggresive)
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
        io.snapshot(noise_pcd, name = "stat_noise")
        io.snapshot(clean_pcd, name = "stat_clean" )
    if visualization:
        io.visualize_differences(clean_pcd, noise_pcd)
    # Outlier Detection: +1 for points classified as noise:
    true_clean_ind = actual_inds if large else ind
    index_array = np.ones(nrpcqa_probs.shape[0], dtype=bool)
    index_array[true_clean_ind] = False
    nrpcqa_probs[index_array] += 1
    # Construct metric:
    NUM_STATISTICAL_OUTLIERS = len(noise_pcd.points)
    return [("METRIC_RATIO_STAT",NUM_STATISTICAL_OUTLIERS/NUM_RAW_POINTS)]


def nrpcqa_denoising(pcd, snapshotting = False, visualization = False):
    """
    TODO
    """
    global nrpcqa_clean
    global nrpcqa_probs
    # Add user-defined denoising agggresiveness:
    # - Alt 1: remove only points with highest probability - soft removal
    # - Alt 2: E.g keep points with zero probability - aggresive removal
    # - Alt 3: No removal
    outlier_removal_type = input("Insert outlier removal type aggresive (a/A) or soft (s/S) or none (ENTER):")
    outlier_removal_type = outlier_removal_type.lower()
    soft_choices = ["s", "soft"]
    aggresive_choices = ["a", "aggresive"]
    possible_choices = [""] + soft_choices + aggresive_choices
    while outlier_removal_type not in possible_choices:
        outlier_removal_type = input("Type error! Insert outlier removal type aggresive (a/A) or soft (s/S) or none (ENTER):")
        outlier_removal_type = outlier_removal_type.lower()
    # *TODO: Add a GUI bar for aggresiveness.
    # TODO: For now: use quartile Q1 (soft) and Q3 (aggresive) as thresholds.
    if outlier_removal_type in soft_choices:
        clean_indices = np.where(nrpcqa_probs != np.max(np.unique(nrpcqa_probs)))[0]
    elif outlier_removal_type in aggresive_choices:
        clean_indices = np.where(nrpcqa_probs == 0)[0]
    else:
        # All are clean:
        clean_indices = list(range(0, len(nrpcqa_probs)))
    nrpcqa_clean = proc.get_cloud_by_index(pcd, clean_indices)
    # Paint uniformly:
    # nrpcqa_clean.paint_uniform_color([1, 0, 0])
    if snapshotting:
        io.snapshot(nrpcqa_clean, name="nrpcqa_denoised")
    if visualization:
        io.visualize(nrpcqa_clean)


@lru_cache
def process_without_reference(input_file: str, *args):
    '''
        Point Cloud Data Validation - without reference
    Args:
        input_file (str) : path of point cloud to analyse
        *args (str) : name of output file - optional
    '''
    # TODO: Parallelize/schedule work
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

    # Keep track of processing time:
    start_time = time.time()

    print("Action in progress: reading point cloud...") # Order matters
    # Read point cloud from path:
    original_pcd = io.read_cloud(input_file)
    NUM_RAW_POINTS = len(original_pcd.points)
    # Cloud to be worked on:
    pcd = copy.deepcopy(original_pcd)
    if snapshotting:
        io.snapshot(original_pcd, name = "original")
    if visualization:
        io.visualize(original_pcd)

    # Keep boolean check for large scan (over 0.5 mil):
    if len(pcd.points) > 500_000:
        large = True

    print("Action in progress: instantiate probability tracking...") # Order matters
    # Keep point-wise probability for outlier detection:
    point_probs = np.zeros(NUM_RAW_POINTS)
    # Update global:
    nrpcqa_probs = point_probs

    print("Action in progress: downsize/voxelize point cloud...")
    nrpcqa_downsize(pcd, snapshotting = snapshotting, visualization = visualization) # Order matters

    print("Action in progress: density assessment based on voxel modelling...")
    nrpcqa_modelling_metrics = nrpcqa_modelling(pcd, snapshotting = snapshotting, visualization = visualization)

    print("Action in progress: density assessment based on voxelization...")
    nrpcqa_voxelization_metrics = nrpcqa_voxelization(pcd, snapshotting = snapshotting, visualization = visualization)

    print("Action in progress: density assessment followed by statistical noise removal based on radius neighbourhood,...")
    nrpcqa_radius_metrics = nrpcqa_radius_nb(pcd, snapshotting = snapshotting, visualization = visualization)

    print("Action in progress: density-based noise removal based on Local Outlier Factor...")
    nrpcqa_lof_metrics = nrpcqa_lof(pcd, snapshotting = snapshotting, visualization = visualization)

    print("Action in progress: statistical removal of noise based on distance to neighbours...")
    nrpcqa_stat_metrics = nrpcqa_statistical(pcd, snapshotting = snapshotting, visualization = visualization)

    print("Action in progress: denoising original point cloud...") # Must be last
    nrpcqa_denoising(pcd, snapshotting = snapshotting, visualization = visualization)

    print("Action in progress: calculating quality...")
    # Fetch metrics:
    metrics = [nrpcqa_modelling_metrics, nrpcqa_voxelization_metrics, nrpcqa_radius_metrics, nrpcqa_lof_metrics, nrpcqa_stat_metrics]
    metrics_dict = dict(metrics[0])
    for i in range(1, len(metrics)):
        metrics_dict.update(metrics[i])
    # Calculate scores:
    number_metrics = len(list(metrics_dict.values())) - 1
    TOTAL = 0.0
    for k,v in metrics_dict.items():
        TOTAL += 1 - v
    QUALITY = calculate_quality(TOTAL, number_metrics)

    print("Action in progress: writting to .json file...")
    # Write metrics to JSON file:
    json_dict = metrics_dict
    json_dict["QUALITY"] = QUALITY
    # End processing time
    end_time = time.time()
    nrpcqa_time = datetime.timedelta(seconds=end_time - start_time)
    json_dict["PROCESSING_TIME"] = str(nrpcqa_time)
    if len(args) != 0:
        # If output file name given, use it as filename:
        io.write_to_json(json_dict, args[0])
    else:
        io.write_to_json(json_dict)


def map_to_probabilities(values):
    '''
        Maps value to relative probability values using inverse of min-max scaling.
    Args:
        values (list) : list of integers/floats
    Returns:
        mapped_probs (list) : list of floats by mapping the values to the relative probability
    '''
    min_value = min(values)
    max_value = max(values)
    mapped_probs = [1.0 - ((value - min_value)/(max_value - min_value)) for value in values]
    return mapped_probs

def calculate_quality(value, num_metrics):
    '''
        Maps value to actual quality (lower values mapped to worse quality). At the time being all measures are weighted equally.
    Args:
        value (float) : total score computed by aggregating metrics
        num_metrics (int) : number of metrics used to calculate score
    Returns:
        quality (str) : either "Bad Quality", "Mixed Quality", "Good Quality", depending on value
    '''
    # Get the boundary values for each quality score:
    range_boundaries = np.linspace(0, num_metrics + 1, 4)
    ranges = [(range_boundaries[i], range_boundaries[i+1]) for i in range(len(range_boundaries)-1)]

    # Return quality according to the range of the score:
    for i in range (len(ranges)):
        if value <= ranges[i][1] and value >= ranges[i][0]:
            if i == 0:
                return "Bad Quality"
            elif i == 1:
                return "Mixed Quality"
            else:
                return "Good Quality"


def stitch(dir_path:str, *args):
    '''
        Aggregates a directory of .ply files into single PointCloud object.
        Outputs result to same directory.
        Assumes same coordinates/scale between point clouds.
    Args:
        dir_path (str) : path of directory containing .ply files
        *args (str) : name of output file - optional
    '''
    # TODO: Add progress bar (e.g. tqdm)
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
        io.write_cloud(dir_path + "/" + args[0]+ ".ply", stitched_pcd)
    # Else, place in same directory with default name:
    else:
        name = "/3ddava_Stitched_"+str(num_plys)+"_Clouds.ply"
        io.write_cloud(dir_path + name, stitched_pcd)


def main(argv = None):
    ''' Parsing command line arguments
    Args:
        argv (list): list of arguments
    '''
    global visualization
    global snapshotting

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

    # Stitching subparser:
    stitching = subparsers.add_parser('stitching', help='Stitching Files Help')
    # Obligatory argument:
    stitching.add_argument("directory_path", type=proc.is_dir, help = "Absolute path of directory containing .ply clouds.")
    # Optional arguments:
    stitching.add_argument("-o", "--output",  help = "Filename of output file. No file extension needed.")

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
            # Validate format:
            input_file = args.cloud_file
            if not proc.is_path(input_file):
                raise FileNotFoundError("Specified input cloud file was not found on given path.")
            if not input_file.endswith(".ply"):
                raise TypeError("Input file must be in .ply format.")
            # If present, process with reference:
            visualization = args.visualize
            snapshotting = args.snapshot
            print(visualization)
            print(snapshotting)
            if args.reference:
                print("Processing with reference ... ")
                reference_file = args.reference
                if not proc.is_path(reference_file):
                    raise FileNotFoundError("Specified reference/CAD file was not found on given path.")
                # Validate format:
                if not reference_file.endswith(".stl"):
                    raise TypeError("Reference/CAD file must be in .stl format.")
                # If output filename is given, use as output file name:
                if args.output:
                    output_file = args.output
                    process_with_reference(input_file, reference_file, output_file)
                else:
                    process_with_reference(input_file, reference_file)
            else:
                print("Processing without reference ... ")
                # If output filename is given, use as output file name:
                if args.output:
                    output_file = args.output
                    process_without_reference(input_file, output_file)
                else:
                    process_without_reference(input_file)
        except Exception as e:
            print(str(e) + " Action dropped: processing clouds.  Use -h, --h or --help for more information.")

if __name__ == '__main__':
    main()
