import os
import numpy as np
import trimesh
import open3d as o3d

CAD_POINTS_PATH = "data/cad_points.ply"


def cad_to_points(input_path, n_points=10000):
    """
    Load CAD file, sample points from surface and save as PLY

    CAD units must be the same as the input data

    Args:
        input_path (str): Path to input CAD file (obj, stl etc)
        output_path (str): Path to save output PLY file
        n_points (int): Number of points to sample
    """
    # Load mesh using trimesh
    mesh = trimesh.load(input_path)

    if any(mesh.bounds[1] - mesh.bounds[0] > 1.0):
        print("Your cad is in mm or cm, make sure the input data is in the same units")

    # Scale so the mesh is between -1 and 1.0
    scale = 1 / np.max(mesh.bounds[1] - mesh.bounds[0])
    mesh.apply_scale(scale)

    # Center about zero
    mesh.vertices -= np.mean(mesh.vertices, axis=0)

    # Sample points from surface
    points, _ = trimesh.sample.sample_surface(mesh, n_points)

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Save as PLY
    o3d.io.write_point_cloud(CAD_POINTS_PATH, pcd)

    return scale


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to input CAD file")
    parser.add_argument("data_ply", help="Path to input PLY file")
    parser.add_argument(
        "--n_points", type=int, default=10000, help="Number of points to sample"
    )
    args = parser.parse_args()

    scale = cad_to_points(args.input_path, args.n_points)

    # Load the cad_points file, load the ply file and scale that as well. move it to the origin
    # Then save both to the txt file format

    # Load the sampled CAD points
    cad_pcd = o3d.io.read_point_cloud(CAD_POINTS_PATH)
    cad_points = np.asarray(cad_pcd.points)

    # Save CAD points to txt
    np.savetxt("data/cad_model_points.txt", cad_points)

    # Load the sampled CAD points
    data_pcd = o3d.io.read_point_cloud(args.data_ply)
    data_points = np.asarray(data_pcd.points)

    # Scale points
    data_points *= scale

    # Put at origin
    data_points -= np.mean(data_points, axis=0)

    # Save data points to txt
    np.savetxt("data/data_points.txt", data_points)

    print("done...")
