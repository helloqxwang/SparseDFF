import numpy as np
from optimize.alignment import read_hand_arm

hand_gt_pose = read_hand_arm(name='monkey0') 
hand_gt_pose[:, 9:] = np.array(
[0.64253635, 1.57270415, 0.49236225,-0.08214985,  0.68294667 , 1.6002892,
0.53108901, -0.03578039,  0.0619254,  1.64880056,  0.25137844 , 0.32642732,
0.08721518,  0.64840628 , 1.53626404 , 0.70697043 , 0.01392171 , 0.89621698,
1.62880352 ,0.06177642 , -0.82736548 , 0.28054595 ])[None, :]
hand_gt_pose[:, 2] += 0.04  # probably because of the wrest
hand_gt_pose[:, 1] -= 0.16
hand_gt_pose[:, 0] -= 0.18
hand_gt_pose[:, 3:9] = np.array([0, 0, 1, 0, 1, 0])[None, :]
hand_gt_pose[:, -5] -= 0.5
np.save('./camera/hand_arm/drill.npy', hand_gt_pose)