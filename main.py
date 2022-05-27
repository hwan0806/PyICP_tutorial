
import os
import copy
import argparse

import numpy as np
np.set_printoptions(precision=4)    # 부동소숫점 출력 자리수 결정

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import collections

#from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

#from minisam import *

import utils.UtilsDataset as DtUtils
import utils.UtilsPointcloud as PtUtils
import utils.UtilsVisualization as VsUtils
from utils.UtilsMisc import *

from utils.PoseGraphManager import *
from utils.ScanContextManager import *
import utils.ICP2 as ICP2
import utils.ICP as ICP



# argparser 
parser = argparse.ArgumentParser(description='Python ICP SLAM tutorial')

parser.add_argument('--down_num_points',type=int,default=5000)      # downsample된 point 수 
parser.add_argument('--num_rings',type=int,default=20)              # SC의 행 수 
parser.add_argument('--num_sectors',type=int,default=60)            # SC의 열 수 
parser.add_argument('--num_candidates',type=int,default=10)         # loop detection 후보군 개수 
parser.add_argument('--try_gap_loop_detection',type=int,default=10) # loop detection 과정을 몇 frame 마다 진행할 것인가
parser.add_argument('--loop_threshold',type=float,default=0.11)     # loop detection 결정짓는 경계값
parser.add_argument('--data_base_dir',type=str,                     # base data directory
                    default='/run/user/1000/gvfs/smb-share:server=synology-nas.local,share=slam_dataset/KITTI Dataset/Obometry/data_odometry_velodyne/dataset/sequences')
parser.add_argument('--sequence_idx',type=str,default='00')         # dataset sequence
parser.add_argument('--LiDAR_name', type=str,default='velodyne')
parser.add_argument('--save_gap',type=int,default=300)              # ?????
args = parser.parse_args()

# scan dataset 
scan_manager = DtUtils.ScanDirManager(args.data_base_dir, args.sequence_idx, args.LiDAR_name)
scan_paths = scan_manager.scan_fullpaths       # scan bin file들의 list 
num_frames = len(scan_paths)

# Pose Graph Manager 
PGM = PoseGraphManager()                       # Pose Graph Optimization을 위한 클래스 생성
PGM.addPriorFactor()                           # 초기 노드 prior factor 생성 

SCM = ScanContextManager(shape=[args.num_rings,args.num_sectors],
                         num_candidates=args.num_candidates,
                         threshold=args.loop_threshold)

src_node = []
dst_node = []

for for_idx, scan_path in tqdm(enumerate(scan_paths),total=num_frames,mininterval=5.0):
    
    # point cloud data 처리
    curr_scan_pts = PtUtils.readScan(scan_path)
    curr_scan_down_pts =PtUtils.random_sampling(curr_scan_pts,args.down_num_points)
    
    
    PGM.curr_node_idx = for_idx
    SCM.addNode(node_idx=PGM.curr_node_idx, ptcloud=curr_scan_down_pts)
    if(PGM.curr_node_idx == 0):
        PGM.prev_node_idx = PGM.curr_node_idx
        prev_scan_pts = copy.deepcopy(curr_scan_pts)
        icp_initial = np.eye(4)
        continue
    ## =======================================================================================================

    # calc odometry 
    prev_scan_down_pts = PtUtils.random_sampling(prev_scan_pts,num_points=args.down_num_points)
    odom_transform,_,_ = ICP2.icp(prev_scan_down_pts,curr_scan_down_pts, init_pose=icp_initial, max_iterations=20)


    PGM.curr_se3 = np.matmul(PGM.curr_se3, odom_transform)      # 이전 frame의 se3에서 update된 se3로 변환
    icp_initial = odom_transform

    PGM.addOdometryFactor(odom_transform)                       # odom 정보를 factor로 update

    PGM.prev_node_idx = PGM.curr_node_idx
    prev_scan_pts = copy.deepcopy(curr_scan_pts)


    ## ========================================================================================================
    # loop detection and optimize the graph 
    if(PGM.curr_node_idx > 1 and PGM.curr_node_idx % args.try_gap_loop_detection == 0): 
        # 1/ loop detection 
        loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
        if(loop_idx == None): # NOT FOUND
            pass
        else:
            print("Loop event detected: ", PGM.curr_node_idx, loop_idx, loop_dist)
            src_node.append(for_idx)
            dst_node.append(loop_idx)

            loop_scan_down_pts = SCM.getPtcloud(loop_idx)
            loop_transfom,_,_ = ICP2.icp(prev_scan_down_pts,curr_scan_down_pts,init_pose=yawdeg2se3(yaw_diff_deg),max_iterations=20)
            PGM.addLoopFactor(loop_transfom, loop_idx)          # loop factor 추가 : 실질적 loop closing 시작!
            
            PGM.optimizePoseGraph()                             # loop factor를 활용하여 전체 map optimization
            
            
            
        

print(src_node)
print(dst_node)

# src_node=[1610,3320,3340,3360,3380,3450,3470,3480,3500,3570,3590,3600,3610,3640,3730,3740,3760,3790,4510,4520]
# dst_node=[166,2376,2396,2416,2432,447,468,480,507,612,635,647,660,695,719,794,806,830,871,66,81]

# pose dataset
pose_manager = DtUtils.PoseDirManager()
x,y,z = pose_manager.getPose()


# visualization
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(x,y,z)
ax.plot(x,y,0)


for i in range(len(src_node)):
    line_x = [x[src_node[i]],x[dst_node[i]]]
    line_y = [y[src_node[i]],y[dst_node[i]]]
    line_z = [7,0]
    ax.plot(line_x,line_y,line_z,c='black')
    

plt.show()

