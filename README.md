<h1> 3D-DaVa Package </h1>

3D-DaVa is a Python-based package that offers point cloud (3D) data validation for digitized products, allowing no-reference and full-reference point cloud quality assessments. Compared to other quality assessment tools, which often output use-case specific metrics, 3D-DaVa can offer a more user-friendly and natural explanation to justify the quality scores found for a given digitized cloud. It has also a few other functionalities, such as stitching point clouds together and denoising point clouds. The pipeline allows customization of metrics and user-defined thresholds that weigh particular properties of the point clouds and give them less/more importance in the overall assessment.

Implementation backbone: Open3D


<h2> Installation Guide </h2>

To use 3D-DaVa, first follow the installation steps:

Anaconda and Git are required prior to package installation !

Installation steps:

1. Via Git: clone repository locally using either the HTTPS or SSH-key using the command:

    git clone <ADD HTTPS/SSH-KEY>

2. You should now have a ProjectClouds folder containing all necessary files. Change location to the ProjectClouds folder through your Anaconda Prompt terminal.

3. In the Anaconda Prompt, create the conda environment from the environment.yml file by entering:

    conda env create -f environment.yml

4. Activate environment by entering:

    conda activate 3ddava

5. Verify environment creation by entering (and checking if 3d-dava is listed):

    conda env list

6. In the same directory, run the following script to install 3D-DaVa package:

    python3 -m pip install .

7. The 3D-DaVa package should now be locally installed. You can verify this by checking if 3d-dava is present in the activated environment by entering:

    conda list


<h2> Usage </h2>

You can use the package directly from the Anaconda command line.

<h3> No Reference Data Validation </h3>
    
Data validation using only the intrinsic properties of the digitized point cloud. Returns a .json file containing the quality scores based on these.

    python -m 3D_DaVa processing [-h] [-o OUTPUT] cloud_file

Arguments:

**cloud_file**: point cloud file to be asessed (recommended format: .ply, see notes).

**-h, –help**: (Optional) show arguments

**-o OUTPUT, –output OUTPUT**: (Optional) filename of metric-filled output file. No file extension needed (automatically turned to .json)

<h3> Full Reference Data Validation </h3>

Data validation using both the no-reference metrics of the digitized point cloud and metrics resulting from the alignment with a reference model. Returns a .json file containing the quality scores based on these.

    python -m 3D_DaVa processing [-h]  [-o OUTPUT] cloud_file -r REFERENCE

Arguments:

**cloud_file**: point cloud file to be asessed (recommended format: .ply, see notes).

**-h, –help**: (Optional) show arguments

**-r REFERENCE**, –reference REFERENCE: reference file (recommended format: .stl, see notes)

**-o OUTPUT, –output OUTPUT**: (Optional) filename of metric-filled output file. No file extension needed (automatically turned to .json)

<h3> Stitching </h3>

Fuses together several point clouds into a single point cloud, if necessary. This is a typical step when we have more digitized point clouds of the same object. For now we assume that the scale and orientation are the same for all point clouds involved.

    python -m 3D_DaVa stitching [-h] [-o OUTPUT] directory_path

Arguments:

**directory_path**: absolute path of directory containing point clouds (recommended format: .ply, see notes).

**-h, –help**: (Optional) show arguments

**-o OUTPUT, –output OUTPUT**: (Optional) name of stitched point cloud file
