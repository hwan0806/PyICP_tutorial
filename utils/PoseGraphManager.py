import numpy as np
np.set_printoptions(precision=4)

import minisam
from utils.UtilsMisc import getGraphNodePose

class PoseGraphManager:
    def __init__(self):
        
        # covariance 값 초기화
        self.prior_cov = minisam.DiagonalLoss.Sigmas(np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]))  # 왜 이렇게 초기화??
        self.const_cov = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])                                     # 왜 이렇게 초기화??  
        self.odom_cov = minisam.DiagonalLoss.Sigmas(self.const_cov)
        self.loop_cov = minisam.DiagonalLoss.Sigmas(self.const_cov)

        self.graph_factors = minisam.FactorGraph()      # factor = edge : node간 연결 = tranformation 정보 
        self.graph_initials = minisam.Variables()       # Variable = node : pose 정보 

        self.opt_param = minisam.LevenbergMarquardtOptimizerParams()
        self.opt = minisam.LevenbergMarquardtOptimizer(self.opt_param)

        # 변수 생성
        self.curr_se3 = None
        self.curr_node_idx = None
        self.prev_node_idx = None
        self.graph_optimized = None

    # 첫번째 pose를 위한 prior factor 생성 : identity
    def addPriorFactor(self):
        self.curr_node_idx = 0
        self.prev_node_idx = 0

        self.curr_se3 = np.eye(4)

        self.graph_initials.add(minisam.key('x', self.curr_node_idx), minisam.SE3(self.curr_se3))   # initial 값 선언 - node idx, se3
        self.graph_factors.add(minisam.PriorFactor(                                                 # prior factor 생성 - node idx, se3, cov 필요
                                                minisam.key('x', self.curr_node_idx), 
                                                minisam.SE3(self.curr_se3), 
                                                self.prior_cov))

    # graph node 간 odom factor 생성
    def addOdometryFactor(self, odom_transform):    # odom transform 정보가 들어감!
        self.graph_initials.add(minisam.key('x', self.curr_node_idx), minisam.SE3(self.curr_se3))
        self.graph_factors.add(minisam.BetweenFactor(
                                                minisam.key('x', self.prev_node_idx),               # between factor - prev, curr node idx, odom se3, cov
                                                minisam.key('x', self.curr_node_idx), 
                                                minisam.SE3(odom_transform), 
                                                self.odom_cov))

    # loop factor 생성
    def addLoopFactor(self, loop_transform, loop_idx):  # 이전에 왔던 장소 : loop의 idx와, 이전-현재 사이 transform 정보 들어감!
        self.graph_factors.add(minisam.BetweenFactor(
                                            minisam.key('x', loop_idx),                             # between factor - loop idx, curr idx, loop se3, cov
                                            minisam.key('x', self.curr_node_idx),  
                                            minisam.SE3(loop_transform), 
                                            self.loop_cov))

    # optimized pose 계산
    def optimizePoseGraph(self):
        self.graph_optimized = minisam.Variables()
        status = self.opt.optimize(self.graph_factors, self.graph_initials, self.graph_optimized)
        if status != minisam.NonlinearOptimizationStatus.SUCCESS:
            print("optimization error: ", status)

        # correct current pose 
        pose_trans, pose_rot = getGraphNodePose(self.graph_optimized, self.curr_node_idx)
        self.curr_se3[:3, :3] = pose_rot
        self.curr_se3[:3, 3] = pose_trans
        