import os
import numpy as np
import trimesh
import open3d as o3d

from utils import save_transform

CAD_POINTS_PATH = "data/cad_points.ply"


def cad_to_points(input_path, n_points=10000):
    """
    Load CAD file, sample points from surface and save as PLY

    CAD units must be the same as the input data

    Rescales and centers the output file
    """

    # Load mesh using trimesh
    mesh = trimesh.load(input_path)

    if any(mesh.bounds[1] - mesh.bounds[0] > 1.0):
        print("Your cad is in mm or cm, make sure the input data is in the same units")

    # Center about zero, save transform
    shift = np.mean(mesh.vertices, axis=0)
    mesh.vertices -= shift

    # Create homogeneous transform matrix
    transform = np.eye(4)
    transform[:3, 3] = -shift
    save_transform(transform, "data/C_to_Chat.txt")

    # Scale so the mesh is between -1 and 1.0
    scale = 1 / np.max(mesh.bounds[1] - mesh.bounds[0])
    mesh.apply_scale(scale)
    np.savetxt("data/scale.txt", np.array([scale]))

    # Sample points from surface
    points, _ = trimesh.sample.sample_surface(mesh, n_points)

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Save as PLY
    o3d.io.write_point_cloud(CAD_POINTS_PATH, pcd)

    return scale


def prepare_points(data_ply_path, scale):
    # Load the sampled CAD points
    cad_pcd = o3d.io.read_point_cloud(CAD_POINTS_PATH)
    cad_points = np.asarray(cad_pcd.points)

    # Save CAD points to txt
    np.savetxt("data/cad_model_points.txt", cad_points)

    # Load the sensor data points
    data_pcd = o3d.io.read_point_cloud(data_ply_path)
    data_points = np.asarray(data_pcd.points)

    # Calculate shift
    shift = np.mean(data_points, axis=0)
    data_points -= shift

    transform = np.eye(4)
    transform[:3, 3] = -shift
    save_transform(transform, "data/O_to_Ohat.txt")

    # Scale points
    data_points *= scale

    # Save data points to txt
    np.savetxt("data/data_points.txt", data_points)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to input CAD file")
    parser.add_argument("data_ply", help="Path to input PLY file")
    parser.add_argument("--n_points", type=int, default=10000, help="Number of points to sample")
    args = parser.parse_args()

    scale = cad_to_points(args.input_path, args.n_points)
    prepare_points(args.data_ply, scale)

    print("done...")
