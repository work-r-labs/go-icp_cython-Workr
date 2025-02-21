from rerun_sdk import rerun as rr
import trimesh

rr.init("icp_visualisation", spawn=True)

# Visualise the CAD model
cad_mesh = trimesh.load("data/cad.stl")

# Sample points from the CAD model
cad_points = cad_mesh.sample(10000)
rr.log("cad_points", cad_points, colors=(0, 255, 0))

# Visualise the original data points

