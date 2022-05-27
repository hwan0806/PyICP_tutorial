from xml.dom.expatbuilder import theDOMImplementation
from click import secho
import numpy as np
np.set_printoptions(precision=4)

from scipy import spatial

# xy좌표의 4사분면 위치에 따른 각도값 도출 - 
def xy2theta(x, y):
    if (x >= 0 and y >= 0): 
        theta = 180/np.pi * np.arctan(y/x);
    if (x < 0 and y >= 0): 
        theta = 180 - ((180/np.pi) * np.arctan(y/(-x)));
    if (x < 0 and y < 0): 
        theta = 180 + ((180/np.pi) * np.arctan(y/x));
    if ( x >= 0 and y < 0):
        theta = 360 - ((180/np.pi) * np.arctan((-y)/x));

    return theta

# 단일 point 하나가 속한 bin 영역을 찾는 함수
def pt2rs(point, gap_ring, gap_sector, num_ring, num_sector):
    x = point[0]
    y = point[1]
    
    # 0으로 나누는 것 방지 
    if(x == 0.0):
        x = 0.001
    if(y == 0.0):
        y = 0.001
        
    theta = xy2theta(x,y)
    faraway = np.sqrt(x*x + y*y)
    
    # xy 좌표를 sc 행렬 idx로 만들어주는 과정 
    idx_ring = np.divmod(faraway, gap_ring)[0]                             # [0]을 붙이는 이유는???
    idx_sector = np.divmod(theta, gap_sector)[0]                           # => 그냥 return 하면 nparray 형식이기 때문에 일반 데이터 형으로 바꿔주는 것
    
    # max range 넘어가는 범위에 point 존재할 때, 마지막 ring에 할당해줌 
    if(idx_ring >= num_ring):
        idx_ring = num_ring - 1
        
    return int(idx_ring), int(idx_sector)
    

# scan data를 scan context로 바꾸는 함수 
def ptcloud2sc(ptcloud,sc_shape,max_length):
    num_ring = sc_shape[0]
    num_sector = sc_shape[1]
    
    gap_ring = max_length/num_ring
    gap_sector = 360/num_sector
    
    enough_large = 500
    sc_storage = np.zeros([enough_large,num_ring,num_sector])               # sc 저장소 enough_large 개수만큼 수용 가능
    sc_counter = np.zeros([num_ring,num_sector])                            # counter의 의미? -> 하나의 bin에 중복되는 point의 z값들을 기록할 수 있는 공간
    
    num_points = ptcloud.shape[0]
    
    for pt_idx in range(num_points):                                         # scan data의 point 개수만큼 반복
        point = ptcloud[pt_idx,:]
        point_height = point[2] + 2.0                                        # point의 높이 대략적 추정 ( LiDAR 위치가 2m 상공이라는 뜻? )
    
        idx_ring, idx_sector = pt2rs(point, gap_ring, gap_sector, num_ring, num_sector)
        
        if sc_counter[idx_ring,idx_sector] >= enough_large:                  # 충분히 많은 point가 bin에 존재한다면, 나중 값들은 무시 
            continue
        
        sc_storage[int(sc_counter[idx_ring,idx_sector]), idx_ring,idx_sector] = point_height   # point의 z값이 counter idx인 행렬의 한 bin에 입력됨
        sc_counter[idx_ring,idx_sector] += 1
    
    sc = np.amax(sc_storage, axis=0)                                         # 0번 축 기준, max값 추출하여 20 * 60 Scan Context 생성!
    return sc
        
        
        

# ringkey 생성 - Scan Context의 행을 평균내어 열벡터 생성
def sc2rk(sc):
    return np.mean(sc,axis=1)

# 두 Scan Context간 비교를 통해 similarity 판단 - 한 열씩 shift하며 cosine similarity 계산 
def distance_sc(sc1,sc2):
    num_sector = sc1.shape[1]
    
    _one_step = 1
    sim_for_each_cols = np.zeros(num_sector)                # shifting된 열들의 cosine similarity를 계산하기 위해 행렬 생성 
    
    for i in range(num_sector):
        sc1 = np.roll(sc1, _one_step, axis=1)               # axis 1 ( = 열 방향) 하나씩 shift
        
        sum_cossim = 0
        num_col_engaged = 0
        
        for j in range(num_sector):
            col_j1 = sc1[:,j]
            col_j2 = sc2[:,j]
            
            if (~np.any(col_j1) or ~np.any(col_j2)):        # col이 모두 0인 경우를 제외하기 위하여 
                # to avoid being divided by zero when calculating cosine similarity
                # - but this part is quite slow in python, you can omit it.
                continue 
            
            cossim = np.dot(col_j1,col_j2) / (np.linalg.norm(col_j1) * np.linalg.norm(col_j2))
            
            sum_cossim += cossim
            num_col_engaged += 1
        
        sim_for_each_cols[i] = sum_cossim / num_col_engaged     # shift 별 similarity를 행렬화
    
    yaw_diff = np.argmax(sim_for_each_cols) + 1                 # 가장 similarity 가 높은 shift 정도를 도출해내기 위해서!
    sim = np.max(sim_for_each_cols)
    dist = 1- sim
    
    return dist, yaw_diff                   # dist와 yaw diff는 어디서 사용됨?????????????


class ScanContextManager:
    def __init__(self,shape,num_candidates,threshold):
        self.shape = shape
        self.num_candidates = num_candidates
        self.threshold = threshold
        
        self.max_length = 80                                # 최대 탐지 거리
        self.ENOUGH_LARGE = 15000                           # 얼마나 많은 frame을 담을 지
        
        self.ptclouds = [None] * self.ENOUGH_LARGE          # frame 당 만들어진 raw points들을 저장할 ENOUGH_LARGE 개수 만큼의 배열
        self.scancontexts = [None] * self.ENOUGH_LARGE      # frame 당 만들어진 sc들을 저장할 ENOUGH_LARGE 개수 만큼의 배열
        self.ringkeys = [None] * self.ENOUGH_LARGE          # frame 당 만들어진 ringkey들을 저장할 ENOUGH_LARGE 개수 만큼의 배열
        
        self.curr_node_idx = 0
        
    def addNode(self, node_idx, ptcloud):
        sc = ptcloud2sc(ptcloud, self.shape, self.max_length)       
        rk = sc2rk(sc)      # sc를 ringkey로 변환 : sc의 평균 취한 것
                
        self.curr_node_idx = node_idx
        self.ptclouds[node_idx] = ptcloud
        self.scancontexts[node_idx] = sc
        self.ringkeys[node_idx] = rk
    
    
    def getPtcloud(self, node_idx):
        return self.ptclouds[node_idx]

    ############################## 이해하기 ####################################
    def detectLoop(self):        
        exclude_recent_nodes = 30
        valid_recent_node_idx = self.curr_node_idx - exclude_recent_nodes                       # 가장 최근 30개를 제외한 영역을 관심영역으로 간주

        if(valid_recent_node_idx < 1):      # 0~29 idx까진 추출 X 
            return None, None, None
        else:
            # step 1
            ringkey_history = np.array(self.ringkeys[:valid_recent_node_idx])
            ringkey_tree = spatial.KDTree(ringkey_history)                                      # ringkey에 대한 KDTree 구축!

            ringkey_query = self.ringkeys[self.curr_node_idx]
            _, nncandidates_idx = ringkey_tree.query(ringkey_query, k=self.num_candidates)      # num_candidate만큼 ringkey 비슷한 상위 rinkgkey 선발

            # step 2
            query_sc = self.scancontexts[self.curr_node_idx]
            
            nn_dist = 1.0 # initialize with the largest value of distance
            nn_idx = None
            nn_yawdiff = None
            
            # ringkey 기준으로 만들어진 후보군들에서 sc 비교 
            for ith in range(self.num_candidates):
                candidate_idx = nncandidates_idx[ith]
                candidate_sc = self.scancontexts[candidate_idx]
                dist, yaw_diff = distance_sc(candidate_sc, query_sc)
                if(dist < nn_dist):                                                             # ringkey candidates 중 가장 distance 작은 값 추출하는 과정 
                    nn_dist = dist
                    nn_yawdiff = yaw_diff
                    nn_idx = candidate_idx

            if(nn_dist < self.threshold):                                                       # threshold보다 dist 작으면, loop detection 
                nn_yawdiff_deg = nn_yawdiff * (360/self.shape[1])
                return nn_idx, nn_dist, nn_yawdiff_deg # loop detected!
            else:
                return None, None, None

        
        