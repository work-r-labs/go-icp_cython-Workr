from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE
import numpy as np
import time


def loadPointCloud(filename):
    pcloud = np.loadtxt(filename, skiprows=1)
    plist = pcloud.tolist()
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


goicp, rNode, tNode = init_go_icp()

Nm, a_points, _ = loadPointCloud("./data/cad_model_points.txt")
Nd, b_points, np_b_points = loadPointCloud("./data/data_points.txt")

goicp.loadModelAndData(Nm, a_points, Nd, b_points)

# LESS DT Size = LESS TIME CONSUMPTION = HIGHER ERROR
goicp.setDTSizeAndFactor(300, 2.0)
goicp.setInitNodeRot(rNode)
goicp.setInitNodeTrans(tNode)

start = time.time()
print("Building Distance Transform...")
goicp.BuildDT()
print("REGISTERING....")
goicp.Register()
end = time.time()
print("TOTAL TIME : ", (end - start))
optR = np.array(goicp.optimalRotation())
optT = goicp.optimalTranslation()
optT.append(1.0)
optT = np.array(optT)

transform = np.empty((4, 4))
transform[:3, :3] = optR
transform[:, 3] = optT

print(np_b_points.shape, np.ones((Nd, 1)).shape)

# Now transform the data mesh to fit the model mesh
transform_model_points = (transform.dot(np.hstack((np_b_points, np.ones((Nd, 1)))).T)).T
transform_model_points = transform_model_points[:, :3]

PLY_FILE_HEADER = (
    "ply\nformat ascii 1.0\ncomment PYTHON generated\nelement vertex %s\nproperty float x\nproperty float y\nproperty float z\nend_header"
    % (Nd)
)
np.savetxt(
    "./data/data_points_aligned.ply",
    transform_model_points,
    header=PLY_FILE_HEADER,
    comments="",
)

## DO COMPARISON - Broken shit
# import open3d as o3d
# # Convert numpy arrays to Open3D point clouds
# source = o3d.geometry.PointCloud()
# target = o3d.geometry.PointCloud()
# source = o3d.io.read_point_cloud("./data/data_points.txt")
# target = o3d.io.read_point_cloud("./data/cad_model_points.txt")

# # Apply ICP
# reg_p2p = o3d.pipelines.registration.registration_icp(
#     source, target, 0.02,
#     np.eye(4),
#     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
# )

# transformed_source = source.transform(reg_p2p.transformation)

# # Save the transformed point cloud
# o3d.io.write_point_cloud("./data/transformed_source.ply", transformed_source)


## END DO COMPARISON

print(optR)
print(optT)
print(transform)
