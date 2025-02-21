"""
Script to run goicp as a pose estimation module in the workr perception pipeline
"""

import trimesh
import numpy as np
import open3d as o3d
import cv2
import json
import time

from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE


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
    C_to_Chat = np.eye(4)
    C_to_Chat[:3, 3] = -shift

    # Scale so the mesh is between -1 and 1.0
    scale = 1 / np.max(mesh.bounds[1] - mesh.bounds[0])
    mesh.apply_scale(scale)

    # Sample points from surface
    points, _ = trimesh.sample.sample_surface(mesh, n_points)

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = np.asarray(pcd.points)

    return scale, C_to_Chat, pcd


def prepare_points(data_pcd, scale):
    data_points = np.asarray(data_pcd.points)

    # Calculate shift
    shift = np.mean(data_points, axis=0)
    data_points -= shift

    O_to_Ohat = np.eye(4)
    O_to_Ohat[:3, 3] = -shift

    # Scale points
    data_points *= scale

    return O_to_Ohat, data_points


def rgbd_to_pointclouds(rgb_path, depth_path, camera_path, mask_path):
    """
    Load RGBD image and camera intrinsics, return point clouds
    """

    pcd_list = []

    # Load RGB and depth images
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # Load camera intrinsics
    with open(camera_path, "r") as f:
        camera_data = json.load(f)
        camera_data = camera_data['cam_K']

    # Create Open3D intrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=rgb.shape[1],
        height=rgb.shape[0],
        fx=camera_data[0],
        fy=camera_data[4],
        cx=camera_data[2],
        cy=camera_data[5],
    )

    # Get unique mask values
    unique_masks = np.unique(mask)
    unique_masks = unique_masks[unique_masks != 0]  # exclude background

    for mask_value in unique_masks:
        # Create copy of depth for this mask
        depth_masked = depth.copy()
        depth_masked[mask != mask_value] = 0.0

        # Convert RGB to Open3D format
        rgb_o3d = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_o3d = o3d.geometry.Image(rgb_o3d)

        # Convert depth to float and handle invalid values
        depth_masked = depth_masked.astype(np.float32)
        depth_masked[depth_masked <= 0.01] = np.nan
        depth_o3d = o3d.geometry.Image(depth_masked)

        # Create RGBD image and convert to point cloud
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        pcd_list.append(pcd)

    return pcd_list


def loadPointCloud(pcloud: np.ndarray):
    plist = list(pcloud)
    p3dlist = []
    for x, y, z in plist:
        pt = POINT3D(x, y, z)
        p3dlist.append(pt)
    return pcloud.shape[0], p3dlist, pcloud


def init_go_icp():
    goicp = GoICP()
    rNode = ROTNODE()
    tNode = TRANSNODE()

    # This is the search space, -pi over two pi means we test all possible rotations
    rNode.a = -3.1416
    rNode.b = -3.1416
    rNode.c = -3.1416
    rNode.w = 6.2832

    # Same for translations, because all the points are in the range [-0.5, 0.5]
    tNode.x = -0.5
    tNode.y = -0.5
    tNode.z = -0.5
    tNode.w = 1.0

    goicp.MSEThresh = 0.001
    goicp.trimFraction = 0.0

    if goicp.trimFraction < 0.001:
        goicp.doTrim = False

    return goicp, rNode, tNode


def get_points_data(goicp, rNode, tNode, cad_model_points, data_points):
    Nm, a_points, _ = loadPointCloud(cad_model_points)
    Nd, b_points, np_b_points = loadPointCloud(data_points)

    goicp.loadModelAndData(Nm, a_points, Nd, b_points)

    # LESS DT Size = LESS TIME CONSUMPTION = HIGHER ERROR
    goicp.setDTSizeAndFactor(100, 2.0)
    goicp.setInitNodeRot(rNode)
    goicp.setInitNodeTrans(tNode)

    return Nd, np_b_points, goicp


def run_icp(goicp, Nd, np_b_points):
    start = time.time()
    goicp.BuildDT()
    print("REGISTERING....")
    goicp.Register()
    end = time.time()
    print("ICP TIME : ", (end - start))
    optR = np.array(goicp.optimalRotation())
    optT = goicp.optimalTranslation()
    optT.append(1.0)
    optT = np.array(optT)

    Ohat_to_Chat_scaled = np.empty((4, 4))
    Ohat_to_Chat_scaled[:3, :3] = optR
    Ohat_to_Chat_scaled[:, 3] = optT

    return Ohat_to_Chat_scaled


def goicp_pem(cad_path, rgb_path, depth_path, camera_path, mask_path, base_dir, use_rerun):
    O_to_Ohats = []
    data_points_list = []
    final_transforms = []

    # Prep data
    scale, C_to_Chat, cad_pcd = cad_to_points(cad_path)
    data_pointclouds = rgbd_to_pointclouds(rgb_path, depth_path, camera_path, mask_path)
    for data_pointcloud in data_pointclouds:
        O_to_Ohat, data_points = prepare_points(data_pointcloud, scale)
        O_to_Ohats.append(O_to_Ohat)
        data_points_list.append(data_points)

    # Ready to run ICP
    goicp, rNode, tNode = init_go_icp()
    for idx, data_points in enumerate(data_points_list):
        Nd, np_b_points, goicp = get_points_data(goicp, rNode, tNode, cad_pcd, data_points)
        Ohat_to_Chat_scaled = run_icp(goicp, Nd, np_b_points)
        Ohat_to_Chat = Ohat_to_Chat_scaled.copy()
        Ohat_to_Chat[:3, 3] = Ohat_to_Chat_scaled[:3, 3] / scale

        # Apply the transform chain
        Chat_to_Ohat = np.linalg.inv(Ohat_to_Chat)
        C_to_O = np.linalg.inv(O_to_Ohats[idx]) @ Chat_to_Ohat @ C_to_Chat
        final_transforms.append(C_to_O)


if __name__ == "__main__":
    goicp_pem(
        cad_path="/home/jasper/tilde/dev_tools/bin_picking/pem_goicp_pem/cad.stl",
        rgb_path="/home/jasper/tilde/dev_tools/bin_picking/pem_goicp_pem/rgb.png",
        depth_path="/home/jasper/tilde/dev_tools/bin_picking/pem_goicp_pem/depth.png",
        camera_path="/home/jasper/tilde/dev_tools/bin_picking/pem_goicp_pem/camera.json",
        mask_path="/home/jasper/tilde/dev_tools/bin_picking/pem_goicp_pem/seg_mask.png",
        base_dir="/home/jasper/tilde/dev_tools/bin_picking/pem_goicp_pem",
        use_rerun=True,
    )
