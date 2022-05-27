import random
import numpy as np

def readScan(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1,4))                         # narrow한 xyzi 4열 matrix 
    ptcloud_xyz = scan[:,:-1]
    return ptcloud_xyz

def random_sampling(orig_points, num_points):
    assert orig_points.shape[0] > num_points            # 원본 poit cloud 개수가, 미리 정해둔 num_point 보다 커야만 진행됨.
                                                        # 즉, down sample point 개수보단 더 많은 point가 들어와야 된다.
                                                        
    point_down_idx = random.sample(range(orig_points.shape[0]), num_points)     # raw point cloud 중, down sample pt num만큼 random 선정 
    down_ptcloud = orig_points[point_down_idx,:]                                # random 선정된 행의 data들만 뽑아 행렬 결정
    return down_ptcloud
    