# -*- coding: utf-8 -*-

### IMPORTS ###
import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
import json
import os.path


def read_cloud(filepath):
    '''
        Reads PointCloud (e.g. .ply) file path, returns PointCloud object.
    Args:
        filepath (str) : file path

    Returns:
        pcd (type: PointCloud object)

    '''
    pcd = o3d.io.read_point_cloud(filepath)
    return pcd


def read_mesh(filepath, normals = True):
    '''
        Reads PointCloud (e.g. .ply) file path, returns TriangleMesh object. Calculates normals unless normals is set to False.
    Args:
        filepath (str) : file path
        normals (bool) : calculate normals, defaults to True

    Returns:
        mesh (type: TriangleMesh object)
    '''
    mesh = o3d.io.read_triangle_mesh(filepath)
    if normals:
        mesh.compute_vertex_normals()
    return mesh


def write_cloud(filepath, pcd):
    '''
        Creates cloud file on the given path, using the given point cloud object.
    Args:
        filepath (str) : file path
        pcd (PointCloud) : point cloud object
    '''
    o3d.io.write_point_cloud(filepath, pcd)


def write_to_json(item_dict, name = "output", ref = False):
    '''
        Takes a dictionary and writes the elements to JSON output file.
    Args:
        item_dict (dict) : dictionary with metrics mapped to their string value
        name (str) : filename for json output file
        ref (bool): True if processing with reference, defaults to False
    '''
    name = set_name(name, ref, "json")
    json_obj = json.dumps(item_dict, indent = 4)
    with open(name, "w") as output_file:
        output_file.write(json_obj)


def set_name(name, ref, ext):
    '''
        Create a name that does not exist on current path given specifications.
    Args:
        name (str): name of the file
        ref (bool): True if processing with reference, defaults to False
        ext (str): file extension
    '''
    # Add prefix (ref = reference provided|idv = individual processing)
    if ref:
        prefix = "ref"
    else:
        prefix = "idv"
    i = 0
    while os.path.exists("3ddava_%s_%s (%s).%s" % (prefix, name , i, ext)):
        i += 1
    name = "3ddava_%s_%s (%s).%s" % (prefix, name, i, ext)
    return name


def snapshot(pcd, name = "snapshot", ref = False):
    '''
        Snap picture of Open3D window.
    Args:
        pcd (PointCloud obj): point cloud object
        name (str): name of picture, defaults to "3ddava_snapshot.jpg"
        ref (bool): True if processing with reference, defaults to False
    '''
    # TODO: Make camera adjustment possible (https://github.com/isl-org/Open3D/issues/1110)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    vis.get_render_option().point_size = 5.0
    vis.add_geometry(pcd)
    name = set_name(name, ref, "png")
    vis.capture_screen_image(name, do_render=True)
    # Close:
    vis.destroy_window()


def visualize_differences(*args, rotate = False, VERTICAL_ROT = -0.5, HORIZONTAL_ROT = 0.5):
    '''
        Visualize PointClouds or TriangleMesh objects in 3D space with different coloring. No rotation unless rotate is set to True, and VERTICAL_ROT, HORIZONTAL_ROT not 0.0.
    Args:
        *args (PointCloud/TriangleMesh): point cloud or mesh (1+ of same type)
        rotate (bool) : allow rotation, defaults to False
        VERTICAL_ROT (float) :  vertical rotation factor
        HORIZONTAL_ROT (float) : horizontal rotation factor
    '''
    def rotate_view(vis):
        """ Animation rotation controls """
        ctr = vis.get_view_control()
        ctr.rotate(HORIZONTAL_ROT, VERTICAL_ROT)
        return False
    args = list(args)
    # We color copies of the point clouds (red and blue for two clouds, >=3 random):
    if len(args) == 2:
        red = [1,0,0]
        blue = [0,0,1]
        args[0] = copy.deepcopy(args[0])
        args[0].paint_uniform_color(red)
        args[1] = copy.deepcopy(args[1])
        args[1].paint_uniform_color(blue)
    else:
        for i in range(len(args)):
            args[i] = copy.deepcopy(args[i])
            color = np.random.choice(range(256), size=3).astype(np.float) / 255.0
            color = list(color)
            args[i].paint_uniform_color(color)
    if rotate:
        o3d.visualization.draw_geometries_with_animation_callback(args, rotate_view)
    else:
        o3d.visualization.draw_geometries(args)


def visualize(*args, normal_view = False, rotate = False, VERTICAL_ROT = -0.5, HORIZONTAL_ROT = 0.5):
    '''
        Visualize PointCloud or TriangleMesh objects in 3D space. No rotation unless rotate is set to True, and VERTICAL_ROT, HORIZONTAL_ROT not 0.0.
    Args:
        *args (PointCloud/TriangleMesh): point cloud or mesh (1+ of same type)
        rotate (bool) : allow rotation, defaults to False
        VERTICAL_ROT (float) :  vertical rotation factor
        HORIZONTAL_ROT (float) : horizontal rotation factor
    '''
    def rotate_view(vis):
        """ Animation rotation controls """
        ctr = vis.get_view_control()
        ctr.rotate(HORIZONTAL_ROT, VERTICAL_ROT)
        return False

    args = list(args)

    if rotate:
        if normal_view:
            o3d.visualization.draw_geometries_with_animation_callback(args, rotate_view, point_show_normal=True)
        else:
            o3d.visualization.draw_geometries_with_animation_callback(args, rotate_view)
    else:
        if normal_view:
            o3d.visualization.draw_geometries(args, point_show_normal=True)
        else:
            o3d.visualization.draw_geometries(args)


def plot_values_by_color(values, color_range, x_label = "X", y_label = "Y", fig_length = 10, fig_height = 5, save=False, show=False, name="plot",  ref = False):
    '''
        Plot a histogram of values using provided color for bins and other specifications.
    Args:
        values (int/float list) : list of values to plot
        color_range (dict) : dictionary of value ranges mapped to colors
        x_label (str) :  label on x-axis of plot, defaults to "X"
        y_label (str) :  label on y-axis of plot, defaults to "Y"
        fig_length (int) : length of figure, defaults to 10
        fig_height (int) : height of figure, defaults to 5
        save (bool) : True if we want to save to .png file, defaults to False
        show (bool) : True if we want to output plot while running, defaults to False
        name (str) : name of the output file if Save is set to True, defaults to 'plot'
        ref (bool): True if processing with reference, defaults to False
    '''
    ranges = []
    colors = []
    for r,c in color_range.items():
        ranges.append(r)
        colors.append(c)
    bins = [r[0] for r in ranges]
    fig, ax = plt.subplots(figsize=(10,5), facecolor='w')
    cnts, values, bars = ax.hist(values, edgecolor='black', bins=bins)
    for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
        bar.set_facecolor(colors[i % len(colors)])
    # Set x,y labels:
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # Saving the plot as a .png file:
    if save:
        name = set_name(name, ref, "png")
        plt.savefig(name)
    if show:
        plt.show()
#_______________________________________________________________________________
