import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.spatial import cKDTree

# 读取轨迹数据
traj_df = pd.read_csv('../data/traj.csv')

# 读取路网数据
road_df = pd.read_csv('../data/road.csv')


# 轨迹降噪
def kalman_filter(traj_df, dt=1.0):
    # 初始化 Kalman 滤波器
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # 状态向量 [x, y, vx, vy]
    kf.x = np.array([traj_df['coordinates'][0][0], traj_df['coordinates'][0][1], 0, 0])

    # 状态转移矩阵
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    # 观测矩阵
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    # 初始协方差矩阵
    kf.P *= 1000

    # 观测噪声协方差
    kf.R = np.array([[1, 0],
                     [0, 1]])

    # 过程噪声协方差
    kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.1)

    # 存储滤波后的坐标
    filtered_coordinates = []

    # 遍历轨迹数据
    for index, row in traj_df.iterrows():
        # 获取当前坐标
        z = np.array([row['coordinates'][0], row['coordinates'][1]])

        # 预测
        kf.predict()

        # 更新
        kf.update(z)

        # 存储滤波后的坐标
        filtered_coordinates.append(kf.x[:2].tolist())

    return filtered_coordinates


# 提取轨迹坐标
coordinates = traj_df['coordinates'].apply(eval).tolist()
traj_df['coordinates'] = coordinates

# 应用 Kalman 滤波
filtered_coordinates = kalman_filter(traj_df)

# 将滤波后的坐标重新放回 DataFrame
traj_df['filtered_coordinates'] = filtered_coordinates

# 路网匹配
# 提取路网坐标
road_coordinates = road_df['coordinates'].apply(eval).tolist()

# 将路网坐标展平为一维数组
road_coords_flattened = [coord for coords in road_coordinates for coord in coords]

# 构建 KDTree
tree = cKDTree(road_coords_flattened)


# 匹配轨迹点到最近的路网点
def match_to_road(filtered_coordinates, tree):
    matched_points = []
    for point in filtered_coordinates:
        _, index = tree.query(point)
        matched_points.append(road_coords_flattened[index])
    return matched_points


# 匹配轨迹
matched_coordinates = match_to_road(filtered_coordinates, tree)

# 将匹配后的坐标重新放回 DataFrame
traj_df['matched_coordinates'] = matched_coordinates

# 保存结果
traj_df.to_csv('matched_traj.csv', index=False)