import os
import numpy as np

class ScanDirManager:
    def __init__(self,data_base_dir,sequence_idx,LiDAR_name):
        self.scan_dir = os.path.join(data_base_dir,sequence_idx,LiDAR_name)                         # args 값 받아 directory 이름 완성
        self.scanfile_list = os.listdir(self.scan_dir)                                              # 해당 dir내의 파일명을 list로 만들어 
        self.scanfile_list.sort()                                                                   # 파일명 list를 sort하여 순서대로. 
        
        self.scan_fullpaths = [os.path.join(self.scan_dir, name) for name in self.scanfile_list]    # bin 파일명까지 붙여 각 frame 파일의 절대경로 완성 
        self.num_scans = len(self.scanfile_list)                                                    # 총 frame 수 계산 
        

class PoseDirManager:
    def __init__(self):
        self.data = np.loadtxt('/home/gil/PROJECT/LiDAR_SLAM/src/evo/test/data/00.txt',delimiter=" ",dtype=np.float32)
        
    
    def getPose(self):
        x = self.data[:,3]
        y = self.data[:,7]
        z = 7
        return x,y,z