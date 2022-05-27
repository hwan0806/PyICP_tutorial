
from math import dist
from statistics import mean
import numpy as np
from sklearn.neighbors import NearestNeighbors

# prePt와 curPt 사이 가장 가까운 점들을 point correspondence로 설정 
def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()           # ravel을 통해 다차원 배열을 일차원으로 바꿔 


def best_fit_transform(curPt, prePt):
    assert prePt.shape == curPt.shape

    m = prePt.shape[1]

    center_prePt = np.mean(prePt, axis=0)
    center_curPt = np.mean(curPt, axis=0)

    prePt_dist2cent = prePt - center_prePt
    curPt_dist2cent = curPt - center_curPt

    H = np.dot(prePt_dist2cent.T,curPt_dist2cent)           # prePT.T,curPt 나 curPt.T,curPt와 같은 순서는 상관 없음. 단지 covariance 구하는 식이기 때문(출력: 3by3)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(U,Vt)

    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(U,Vt)        # prev -> curr 좌표계 변환하는 회전행렬 

    t = center_prePt.T - np.dot(R,center_curPt.T)

    T = np.identity(m+1)
    T[:m,:m] = R
    T[:m,m] = t

    return T, R, t



def icp(prePt, curPt, init_pose=None, max_iterations=20, tolerance=0.001):                     # tolerance 역할 : icp 통한 matching 얼마나 잘됐냐 

    assert prePt.shape == curPt.shape              

    m = curPt.shape[1]     # xyz -> 3

    # src : 이전 frame의 point, dst : 현재 frame의 point   -> wide matrix꼴로 변경
    src = np.ones((m+1,curPt.shape[0]))
    dst = np.ones((m+1,prePt.shape[0]))
    src[:m,:] = np.copy(curPt.T)
    dst[:m,:] = np.copy(prePt.T)

    if init_pose is not None:
        src = np.dot(init_pose,src)

    prev_error = 0

    for i in range(max_iterations):

        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        src = np.dot(T,src)

        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    T,_,_ = best_fit_transform(curPt, src[:m,:].T)

  

    return T, distances, i


