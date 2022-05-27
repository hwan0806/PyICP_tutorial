import os
import time
import math

import numpy as np
import matplotlib.pyplot as plt

import minisam

# def getConstdigitsNumber()

# def getUnixTime()

def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))     # Euler angle에서 dot연산 순서는 꼭 zyx 순서여야 함?  xyz 순서로 하면 어떻게 될까?
 
    return R

def yawdeg2so3(yaw_deg):
    yaw_rad = np.deg2rad(yaw_deg)
    return eulerAnglesToRotationMatrix([0, 0, yaw_rad])

def yawdeg2se3(yaw_deg):
    se3 = np.eye(4)
    se3[:3, :3] = yawdeg2so3(yaw_deg)
    return se3 

def getGraphNodePose(graph,idx):
    pose = graph.at(minisam.key('x',idx))
    pose_trans = pose.translation()
    pose_rot = pose.so3().matrix()
    return pose_trans, pose_rot

# def saveOptimizedGraphPose

# def vizCurrentTrajectory