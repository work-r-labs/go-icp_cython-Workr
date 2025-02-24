from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE
import numpy as np
import time

from utils import save_transform


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


def get_points_data(goicp, rNode, tNode):
    Nm, a_points, _ = loadPointCloud("./data/cad_model_points.txt")
    Nd, b_points, np_b_points = loadPointCloud("./data/data_points.txt")

    goicp.loadModelAndData(Nm, a_points, Nd, b_points)

    # LESS DT Size = LESS TIME CONSUMPTION = HIGHER ERROR
    goicp.setDTSizeAndFactor(100, 2.0)
    goicp.setInitNodeRot(rNode)
    goicp.setInitNodeTrans(tNode)

    return Nd, np_b_points, goicp


def run_icp(goicp, Nd, np_b_points):
    start = time.time()
    print("Building Distance Transform...")
    goicp.BuildDT()
    print("REGISTERING....")
    goicp.Register(3.0)
    end = time.time()
    print("TOTAL TIME : ", (end - start))
    optR = np.array(goicp.optimalRotation())
    optT = goicp.optimalTranslation()
    optT.append(1.0)
    optT = np.array(optT)

    transform = np.empty((4, 4))
    transform[:3, :3] = optR
    transform[:, 3] = optT

    save_transform(transform, "data/Ohat_to_Chat_scaled.txt")

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

    print(optR)
    print(optT)
    print(transform)

    return transform


if __name__ == "__main__":
    goicp, rNode, tNode = init_go_icp()
    Nd, np_b_points, goicp = get_points_data(goicp, rNode, tNode)
    transform = run_icp(goicp, Nd, np_b_points)
    print(transform)
