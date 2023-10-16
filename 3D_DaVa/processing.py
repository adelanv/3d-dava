# -*- coding: utf-8 -*-

### IMPORTS ###
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree as kdt
from . import inout
from sklearn.decomposition import PCA
import os.path
from pathlib import Path
from numpy import linalg
from numpy import dot
from numpy.linalg import norm


def map_to_probabilities(values, inverse = True):
    '''
        Maps value to relative probability values. Inverse of min-max scaling is also possible.
    Args:
        values (list) : list of integers/floats
    Returns:
        mapped_probs (list) : list of floats by mapping the values to the relative probability
    '''
    min_value = min(values)
    max_value = max(values)
    if inverse:
        if (max_value - min_value) == 0:
            mapped_probs = [1.0 for value in values]
        else:
            mapped_probs = [1.0 - ((value - min_value)/(max_value - min_value)) for value in values]
    else:
        if (max_value - min_value) == 0:
            mapped_probs = [0.0 for value in values]
        else:
            mapped_probs = [((value - min_value)/(max_value - min_value)) for value in values]
    return mapped_probs

def generate_N_unique_colors(N):
    '''
        Color generator. Generates N unique colors in format [r,g,b], where RGB values are in range [0-1].
    Args:
        N (int) : number of colors

    Returns:
        colors (list) : list of lists with unique colors
    '''
    colors = []
    while len(colors) < N:
        color = np.random.choice(range(256), size=3).astype(np.float64) / 255.0
        color = tuple(color)
        if color not in colors:
            colors.append(color)
    return colors

def minmax_scale(X):
    '''
        Min-max scaling of a list of floats/integers. Maps all values to range [0,1].
    Args:
        X (list) : list of ints/floats

    Returns:
        scaled (list) : scaled list of ints/floats
    '''
    X_max = max(X)
    X_min = min(X)
    scaled = [((x - X_min) / (X_max - X_min)) for x in X]
    return scaled

def sigmoid(x):
    '''
    Turns x value into a [0,1] value using sigmoid:
    Args:
        x (float) : mapped value

    Returns:
        y (float) : x mapped to [0,1]
    '''
    return 1 / (1 + np.exp(-x))


def point_to_mesh(pcd, voxel_size):
    '''
        Turns a point cloud to a mesh by turning into voxel grid and 3D-modelling voxels.
    Args:
        pcd (type: PointCloud object) : point cloud to model
        voxel_size (float) : voxel size

    Returns:
        v_mesh (type: TriangleMesh object)
    '''
    # Reference: https://towardsdatascience.com/how-to-automate-voxel-modelling-of-3d-point-cloud-with-python-459f4d43a227
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size = voxel_size)
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
    v_mesh.scale(voxel_size, [0, 0, 0])
    v_mesh.translate(voxel_grid.origin, relative=True)
    v_mesh.merge_close_vertices(0.0000001)
    v_mesh.compute_vertex_normals()
    return v_mesh


def downsample_and_trace_cloud(pcd, voxel_size = 0.5):
    """
        Downsample point cloud to voxels with given voxel size, and return trace.
    Args:
        pcd (PointCloud obj) : point cloud to downsample
        voxel_size (float) : voxel size

    Returns:
        pcd_down (PointCloud obj) : downsampled PointCloud
        voxel_vecs (List[open3d.utility.IntVector]): list of int vectors representing original point indexes
    """
    pcd_down, _, voxel_vecs = pcd.voxel_down_sample_and_trace(voxel_size = voxel_size,
                                                              min_bound = pcd.get_min_bound(),
                                                              max_bound=pcd.get_max_bound())
    return pcd_down, voxel_vecs


def create_uniform_sampled_cloud_from_mesh(mesh, nr_points = 500,  poisson = False,  factor = 5):
    '''
        Takes a mesh, returns sampled model of nr_points points. If Poisson is allowed, evenly distribute the points by removing samples from the initial sampled cloud, with nr_points/factor as new sampling criterion.
    Args:
        mesh (TriangleMesh obj) : mesh object to be sampled
        poisson (bool) : allow poisson sampling: each point has approx. same distance to the neighbouring points, defaults to False
        nr_points (int) : total nr of points of sampling
        factor (int):  used for sample elimination (nr_points/factor as new sampling criterion), defaults to 5

    Returns:
        pcd (PointCloud obj) : uniformly sampled point cloud
    '''
    pcd = mesh.sample_points_uniformly(number_of_points=nr_points) # Sample uniformly
    if poisson:
        pcd = mesh.sample_points_poisson_disk(number_of_points=int(nr_points/factor), pcl=pcd)
    return pcd


def color_cloud_greyscale(pcd, values, shades = 10):
    """
        Color the point cloud according to values, that is, darker for low values and lighter for high values.
    Args:
        pcd (PointCloud obj) : point cloud object
        values (np.array) : values for each point in point cloud
        shades (int) : number of shades or gradient power, defaults to 10

    Returns:
        pcd (PointCloud obj) : point cloud with greyscale coloring
        color_range (dict) : dictionary used for plotting
    """
    # TODO: Improve computation time -> change to numpy arrays / numba
    # Create ranges:
    color_blocks = np.linspace(min(values), max(values), num=shades)
    ranges = [(color_blocks[i], color_blocks[i+1]) for i in range(len(color_blocks)-1)]
    lowest_value = ranges[0][0]
    # Lightness increasing rate:
    increase_rate = 1 / (len(ranges) + 1)
    # We begin with 0 - as (0,0,0) is the value for black:
    val_color = 0
    # Assign colors to ranges and increase lightness:
    color_range = {}
    for i in range (len(ranges)):
        # Color smallest range in red:
        if i == 0:
            color_range[ranges[i]] = [0,0,0]
        else:
            color_range[ranges[i]] = [val_color, val_color, val_color]
            val_color += increase_rate
    # Assign values to colors given range :
    colors = []
    for v in values:
        for r in ranges:
            if v <= r[1] and v >= r[0]:
                colors.append(color_range[r])
                break
    # Paint point cloud in new greyscale color:
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors).astype(np.float16))
    return pcd, color_range


def get_l2_distance(first_point, second_point):
    '''
        Calculate Euclidean (L2-norm) distance between two points
    Args:
        first_point, second_point (np.array): array with x,y,z coordinates of shape (,3)

    Returns:
        dist (float) : Euclidean (L2-norm) distance between points
    '''
    dist = np.linalg.norm(first_point-second_point)
    return dist


def color_cloud_rainbow(pcd, values, shades = 10):
    """
        Color the point cloud according to values, that is, similar ranges in same color
    Args:
        pcd (PointCloud obj) : point cloud object
        values (np.array) : values for each point in point cloud
        shades (int) : number of shades or gradient power, defaults to 10

    Returns:
        pcd (PointCloud obj) : point cloud with greyscale coloring
        color_range (dict) : dictionary used for plotting
    """
    # TODO: Improve computation time -> turn to numpy arrays / numba
    # Create ranges:
    color_blocks = np.linspace(min(values), max(values), num=shades)
    ranges = [(color_blocks[i], color_blocks[i+1]) for i in range(len(color_blocks)-1)]

    # Assign colors:
    color_range = {}
    for ran in ranges:
        color = np.random.choice(range(256), size=3).astype(np.float64) / 255.0
        color = list(color)
        color_range[ran] = color
    # Assign values to colors given range :
    colors = []
    for v in values:
        for r in ranges:
            if v <= r[1] and v >= r[0]:
                colors.append(color_range[r])
                break
    # Paint point cloud in new color:
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors).astype(np.float16))
    return pcd, color_range


def remove_outliers(pcd, num_nb, std_ratio):
    '''
        Removes points far from their neighbors in comparation to the point cloud average.
    Args:
        pcd (PointCloud obj) : point cloud
        num_nb (int) : number of neighbors used to calculate the average distance
        std_ratio (float) : standard deviation-based threshold level. Lower means aggresive filter.
    Return:
        clean (open3d.geometry.PointCloud) : cleaned point cloud
        ind (List[int]) : indexes of points remaining in clean cloud
    '''
    clean, ind = pcd.remove_statistical_outlier(nb_neighbors=num_nb, std_ratio=std_ratio)
    return clean, ind


def get_FPFH_features(pcd, normal_radius = 1, feature_radius = 1,  max_nn_normal = 30, max_nn_feature = 30):
    '''
        Compute a FPFH feature for each point = 33D vectors that represent the local geometric property.
    Args:
        pcd (PointCloud obj) : point cloud to get features from
        normal_radius : find neighbours within radius in normal estimation, defaults to 1
        max_nn_normal: maximum number of neighbours in neighbourhood, defaults to 30
        feature_radius : find neighbours within radius in FPFH calculation, defaults to 1
        max_nn_feature :  maximum number of neighbours allowed in neighbourhood, defaults to 30
    Returns:
        pcd_fpfh (open3d.cpu.pybind.pipelines.registration.Feature) : FPFH features for each point
    '''
    # KDTree search for normal neighbours using hybrid parameters:
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn_normal))
    # Compute the feature vectors:
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn= max_nn_feature))
    return pcd_fpfh


def get_cloud_by_index(pcd, index, invert = False):
    '''
        Creates cloud file on the given path, using the given point cloud object.
    Args:
        pcd (PointCloud) : point cloud object
        index (list) : array/list of integers for indexes of points that we want to keep
        invert (bool) : True to keep all other indexes except those in 'index'

    Returns:
        new_pcd (PointCloud) : point cloud made out of the points belonging to index list
    '''
    return pcd.select_by_index(index, invert = invert)


def stitch_clouds(pcdlist):
    '''
        Turns a list of .ply filepaths into single PointCloud object and return. Assumption: no rigid transformation needed. Same coordinates and scale.
    Args:
        pcdlist (list) : list of point cloud paths

    Returns:
        pcd (PointCloud object) : aggregated point cloud from the paths '''
    pcd_np_arrs = [np.asarray(inout.read_cloud(file).points) for file in pcdlist]
    pcd_unique_points = np.unique(np.vstack(pcd_np_arrs), axis=0)
    # Pass points to Open3D.o3d.geometry.PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_unique_points)
    return pcd


def is_path(path_str:str):
    '''
        Check that path is exists.
    Args:
        path_str (str): specified path
    Returns:
        exists (bool): returns True if path exists, else False
    '''
    f = Path(path_str)
    exists = f.exists()
    return exists


def principal_component_analysis(data, keep = None):
    '''
        Principal Component Analysis: information about dispersion
    Args:
        data (np.array): (N,M) dimensional array of N instances and M features
        keep (int or 'mle') : number of components to keep (number guessed by setting it to 'mle'), defaults to all

    Returns:
        components (np.array): array of shape (data.shape[0], keep) representing principal components or directions of maximum variance
        exp_var_ratio (np.array): array of shape (data.shape[0],) representing percentages of variance explained by the components, cumulates to 1.0
        exp_var (np.array): array of shape (data.shape[0],) representing variance explanation power
        new_data (np.array): newly transformed data
    '''
    if keep is not None and keep > data.shape[1]:
        raise ValueError("Number of components must be lower or equal to number of feature dimensions.")
    pca = PCA(n_components = keep)
    # Run PCA. Keep newly transformed data (centered)
    pca.fit(data)
    # Get metrics and directions in order of descending variance:
    components = pca.components_
    exp_var_ratio = pca.explained_variance_ratio_
    exp_var = pca.explained_variance_
    # Return:
    new_data = pca.transform(data)
    return components, exp_var_ratio, exp_var, new_data


def cosine_similarity(a,b):
    '''
        Calculates cosine similarity between two vectors (1D)
    Args:
        a,b (list) : vectors

    Returns:
        cos_sim (float) : cosime similarity
    '''
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

#______________________________________________________________________________
def get_closest_distance(pcd_arr, ind, *args):
    '''
        Calculates distance to closest neighbour in point cloud. Returns smallest distance to closest neighbour.
    Args:
        pcd_arr (np.array) : point cloud numpy array
        ind (int): point index
        pcd_tree (KDTree) : KDTree representation (optional) - used to speed up process

    Returns:
        distance (int) : smallest distance to closest point in point cloud
    '''
    if len(args) != 0:
        pcd_tree = args[0]
    else:
       pcd = o3d.geometry.PointCloud()
       pcd.points = o3d.utility.Vector3dVector(pcd_arr)
       pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    distance = pcd_tree.search_knn_vector_3d(pcd_arr[ind], 2)[2][1]
    return distance


def get_ISS_keypoints(pcd,
                      salient_radius: float = 0.0,
                      non_max_radius : float = 0.0,
                      gamma_21 : float = 0.975,
                      gamma_32 : float = 0.975,
                      min_n : int = 5):
    '''
        Computes the ISS keypoints given point cloud. Proposed by Yu Zhong, 'Intrinsic Shape Signatures: A Shape Descriptor for 3D Object Recognition', 2009.
    Args:
        pcd (PointCloud object) : input point cloud
        salient_radius (float): neighborhood radius, defaults to 0.0
        non_max_radius (float): non maxima supression radius, defaults to 0.0
        gamma_21 (float): Upper bound on the ratio between the second and the first eigenvalue, defaults to 0.975
        gamma_32 (float): Upper bound on the ratio between the third and the second eigenvalue, defaults to 0.975
        min_n (int): minimal number of neighbors to include as keypoint

    Returns:
        keypoints (PointCloud object) : ISS points'''
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd,
                                                            salient_radius=salient_radius,
                                                            non_max_radius=non_max_radius,
                                                            gamma_21=gamma_21,
                                                            gamma_32=gamma_32,
                                                            min_neighbors = min_n)
    return keypoints


def create_voxelized_cloud_from_cloud(pcd, voxel_size = 0.5):
    '''
        Takes a point cloud and voxel size, returns voxelizated model.
    Args:
        pcd (PointCloud obj) : input point cloud
        voxel_size (float) : voxel size

    Returns:
        down_pcd (PointCloud object) : voxelized model according to chosen voxel size
    '''
    down_pcd = pcd.voxel_down_sample(voxel_size = voxel_size)
    return down_pcd


def create_voxelized_cloud_from_mesh(mesh, voxel_size = 0.5):
    '''
        Takes a mesh object and voxel size, returns voxelizated model.
    Args:
        mesh (TriangleMesh obj) : mesh file
        voxel_size (float) : voxel size

    Returns:
        pcd (PointCloud object) : voxelized model according to chosen voxel size
    '''
    mesh.scale(1/np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
    pc_np = np.asarray([voxel_grid.origin + pt.grid_index*voxel_grid.voxel_size for pt in voxel_grid.get_voxels()])
    # Turn voxel grid to point cloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np)
    return pcd


def downsample_cloud(pcd, voxel_size = 0.5):
    """
        Downsample point cloud to voxels with given voxel size.
    Args:
        pcd (PointCloud obj) : point cloud to downsample
        voxel_size (float) : voxel size

    Returns:
        pcd_down (PointCloud obj) : downsampled PointCloud
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down


def get_FPFH_features_with_downsizing(pcd, voxel_size = 0.5, max_nn_normal = 30, normal_factor = 2, max_nn_feature = 100, feature_factor = 5):
    '''
        Compute a FPFH feature for each point = 33D vectors that represent the local geometric property.
    Args:
        pcd (PointCloud obj) : point cloud to get features from
        voxel_size (float) : downsampling size, defaults to 0.5
        max_nn_normal: maximum number of neighbours to be searched, defaults to 30
        normal_factor : used in calculating the search radius, multiplied by voxel_size, defaults to 2
        max_nn_feature :  maximum number of neighbours to be searched, defaults to 100
        feature_factor : used in calculating the search radius, multiplied by voxel_size, defaults to 5

    Returns:
        pcd_down (PointCloud obj) : downsampled point cloud
        pcd_fpfh (open3d.cpu.pybind.pipelines.registration.Feature) : FPFH features for each point

    '''
    pcd_down = downsample_cloud(pcd, voxel_size) # Downsample to reduce time
    radius_normal = voxel_size * normal_factor   # Search radius for normals
    # KDTree search for normal neighbours using hybrid parameters:
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_normal))
    radius_feature = voxel_size * feature_factor
    # Compute the feature vectors:
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn= max_nn_feature))
    return pcd_down, pcd_fpfh


def dbscan_clustering_cloud(pcd, eps = 0.5, min_points = 100):
    '''
        Performs clustering of local point groups and returns the colored clustered version.
    Args:
        pcd (PointCloud obj): point cloud to cluster
        eps (float): points are neighbors in cluster if their distance is lower than epsillon, defaults to 0.5
        min_points (int): minimum number of points required to form a dense cluster, defaults to 100

    Returns:
        pcd (PointCloud obj) : colored point cloud according to dbscan result
        labels (np.array) : array with labels for clusters
    '''
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()
    print("Number of clusters: " + str(max_label + 1))
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd, labels


def estimate_point_covariances(pcd, k = 30):
    '''
        Estimate covariance matrix for each point in point cloud given its neighbourhood size
    Args:
        pcd (PointCloud obj) : point cloud to estimate covariance matrixes for
        k (int) : neighbourhood for kNN search done for each point

    Returns:
        covs (np.array) : covariance matrixes for each point
    '''
    covs = o3d.geometry.PointCloud.estimate_point_covariances(pcd, search_param = o3d.geometry.KDTreeSearchParamKNN(knn = k))
    return np.asarray(covs)


def calculate_normals(pcd, knn = 30):
    """
        Calculate normals for all points, based on local surface described by k closest neighbours.
    Args:
        pcd (PointCloud obj) : point cloud to calculaet normals on.
        knn (int) : number of neighbours to search for

    Returns:
        pcd (PointCloud obj) : point cloud object with normals stored
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return pcd


def orient_cloud_normals(pcd, camera_orientation = False, ref = np.array([0., 0., 1.])):
    """
        Reorient normals for all points, based on either camera orientation or given direction.
    Args:
        pcd (PointCloud obj) : point cloud with previously estimated normals
        camera_orientation (bool) : check for whether or not we want camera_orientation to be orientation criteria
        ref (np.array) : orientation reference by 3D vector
    Returns:
        pcd (PointCloud obj) : point cloud object with normals reoriented
    """
    if camera_orientation:
        pcd.orient_normals_towards_camera_location()
    else:
        pcd.orient_normals_to_align_with_direction(orientation_reference=ref)
    return pcd


def report_properties(obj):
    '''
        Returns report/properties of a PointCloud or TriangleMesh:
            - Center: center of geometry coordinates
            - Max/Min bounds: max/min bounds for geometry coordinates
            - Euler Pointcare Characteristic: Vertices + Triangles - Edges
            - Surface area: the sum of the individual triangle surfaces
            - Watertight: true if edge manifold, vertex manifold and not self intersecting
            - Self-intersecting: true if there exists a triangle in the mesh that is intersecting another mesh
            - Edge manifold: true if each edge is bounding either one or two triangles
            - Vertex manifold: true if two or more faces connected only by a vertex and not by an edge
            - Volume: computes volume of the mesh, under the condition that it is watertight and orientable
    Args:
        obj (PointCloud/TriangleMesh) : point cloud or triangle mesh to report

    Returns:
        properties (dict) : properties of object
    '''
    geometryType = str(obj.get_geometry_type()).split(".",1)[1]
    print("Action in progress: intrinsic report on "+ str(obj.dimension())+"D "+ geometryType)
    if geometryType == "PointCloud":
        mean, cov = obj.compute_mean_and_covariance()
        eig_val, eig_vec = linalg.eig(cov)
        properties = {"Number raw points" : str(len(obj.points)),
            "Center point/centroid" : str(obj.get_center()),
            "Max bound" : str(obj.get_max_bound()),
            "Min bound" : str(obj.get_min_bound()),
            "Mean of all points" : str(mean),
            "Covariance matrix" : str(cov),
            "Eigen values" : str(eig_val),
            "Eigen vectors" : str(eig_vec)}
        return properties
    elif geometryType == "TriangleMesh":
        properties = {"Number triangles" : str(np.asarray(obj.triangles).shape[0]),
              "Number vertices" :  str(np.asarray(obj.vertices).shape[0]),
              "Center/centroid" : str(obj.get_center()),
              "Max bound": str(obj.get_max_bound()),
              "Min bound": str(obj.get_min_bound()),
              "Euler Poincare Characteristic":str(obj.euler_poincare_characteristic()),
              "Surface area": str(obj.get_surface_area()),
              "Watertight": str(obj.is_watertight()),
              "Self-intersecting" : str(obj.is_self_intersecting()),
              "Edge manifold" :str(obj.is_edge_manifold()),
              "Vertex manifold": str(obj.is_vertex_manifold())}
        if obj.is_watertight():
            properties["Volume"] = str(obj.get_volume())
        return properties
    else:
        print("\nAction dropped: report object.")
        raise TypeError("Geometry type not recognized.")


def is_dir(path_str:str):
    '''
        Check that path to directory is valid/exists.
    Args:
        path_str (str): specified path

    Returns:
        path_str (str): specified path if directory, else raise Error
    '''
    # Check that path exists:
    if is_path(path_str):
        if os.path.isdir(path_str):
            return path_str
    raise NotADirectoryError(path_str)
