import numpy as np
import pickle

def read_bin_file(bin_file):
    point_cloud_np = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    # 只取x, y, z坐标，忽略反射率
    points = point_cloud_np[:, :3]
    return points,point_cloud_np
def read_npy_file(npy_file):
    painted_point_cloud=np.load(npy_file)
    painted_points=painted_point_cloud[:,:3]
    return painted_points,painted_point_cloud

def arrays_are_contained(set_array1, array2):
    for sub_array in array2:
        if tuple(sub_array) not in set_array1:
            return False
    return True
def read_pkl_file(pkl_file):
    with open(pkl_file, 'rb') as f:
        infos = pickle.load(f)
        return infos
    
points,point_cloud=read_bin_file("000005.bin")
painted_points,painted_point_cloud=read_npy_file("000005.npy")
points={tuple(row) for row in points}
check_results=arrays_are_contained(points,painted_points)
kitti_infos=read_pkl_file("kitti_infos_train.pkl")
print(check_results)
