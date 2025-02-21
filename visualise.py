import rerun as rr
import trimesh
import numpy as np


from utils import load_transform

rr.init("icp_visualisation", strict=True)
rr.spawn(memory_limit="16GB", hide_welcome_screen=True)

# Log axes at the origin
rr.log(
    "world",
    rr.Arrows3D(
        origins=[[0, 0, 0]],
        vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    ),
)

# Visualise the CAD model
cad_mesh = trimesh.load("data/cad.stl")

# Sample points from the CAD model
cad_points = cad_mesh.sample(10000)
rr.log("world/cad_points", rr.Points3D(cad_points, colors=[0, 255, 0]))

# Visualise the original data points
data_points = trimesh.load("data/flat_table.ply")
rr.log("world/data_points", rr.Points3D(data_points.vertices, colors=[255, 0, 0]))

# Apply the C_to_Chat tf to the CAD points
C_to_Chat = load_transform("data/C_to_Chat.txt")
cad_points_transformed = np.dot(cad_points, C_to_Chat[:3, :3].T) + C_to_Chat[:3, 3]
rr.log("world/Chat", rr.Points3D(cad_points_transformed, colors=[0, 255, 0]))

# Apply the O_to_Ohat tf to the data points
O_to_Ohat = load_transform("data/O_to_Ohat.txt")
data_points_transformed = np.dot(data_points.vertices, O_to_Ohat[:3, :3].T) + O_to_Ohat[:3, 3]
rr.log("world/Ohat", rr.Points3D(data_points_transformed, colors=[0, 0, 255]))

# Apply Chat_to_Ohat
Chat_to_Ohat = load_transform("data/Chat_to_Ohat.txt")
aligned_icp_points = np.dot(cad_points_transformed, Chat_to_Ohat[:3, :3].T) + Chat_to_Ohat[:3, 3]
rr.log("world/aligned_icp_points", rr.Points3D(aligned_icp_points, colors=[255, 255, 0]))

# Apply Ohat_to_O
Ohat_to_O = load_transform("data/Ohat_to_O.txt")
final_object_points = np.dot(aligned_icp_points, Ohat_to_O[:3, :3].T) + Ohat_to_O[:3, 3]
rr.log("world/final_object_points", rr.Points3D(final_object_points, colors=[255, 255, 0]))

# Show the final object pose
final_object_pose = load_transform("data/final_object_pose.txt")
cad_points_final = np.dot(cad_points, final_object_pose[:3, :3].T) + final_object_pose[:3, 3]
rr.log("world/final_object_pose", rr.Points3D(cad_points_final, colors=[255, 255, 0]))
