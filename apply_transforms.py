"""
This script converts the generated transforms into a C->O transform which takes the original cad model
points and transforms them to align with the original object data points. 

C->O = C->C^ * C^->O^ * O^->O

where 

C->C^ is the tf applied to the original cad file to center it

O^->C^ scaled, is the GO-ICP estimate required to align the two centered and scaled point clouds 

O->O^ is the tf applied to the original object data point cloud to center it


"""

import numpy as np

from utils import load_transform, save_transform

C_to_Chat = load_transform("data/C_to_Chat.txt")
Ohat_to_Chat_scaled = load_transform("data/Ohat_to_Chat_scaled.txt")
O_to_Ohat = load_transform("data/O_to_Ohat.txt")
scale = np.loadtxt("data/scale.txt")

# Unscale Ohat_to_Chat_scaled
Ohat_to_Chat = Ohat_to_Chat_scaled.copy()
Ohat_to_Chat[:3,3] = Ohat_to_Chat_scaled[:3,3] / scale

# invert stuff
Chat_to_Ohat = np.linalg.inv(Ohat_to_Chat)
Ohat_to_O = np.linalg.inv(O_to_Ohat)

# The transform which places the CAD model pointcloud at the
# location of the object points in the original frame
C_to_O = C_to_Chat @ Chat_to_Ohat @ Ohat_to_O

save_transform(C_to_O, "data/final_object_pose.txt")
print(f"Transform saved to data/final_object_pose.txt")