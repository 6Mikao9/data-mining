import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
import ast

# 读取轨迹数据
traj_df = pd.read_csv('../data/jump_task.csv')
traj_df = traj_df.dropna(subset=['coordinates'])
# 读取路网数据
road_df = pd.read_csv('../data/road.csv')

# 提取轨迹坐标
traj_df['coordinates'] = traj_df['coordinates'].apply(ast.literal_eval)


# 路网插值
def interpolate_line(coords, num_points=10):
    interpolated_coords = []
    for i in range(len(coords) - 1):
        p1 = np.array(coords[i])
        p2 = np.array(coords[i + 1])
        t = np.linspace(0, 1, num_points)
        interpolated_points = (p1[None, :] * (1 - t)[:, None] + p2[None, :] * t[:, None]).tolist()
        interpolated_coords.extend(interpolated_points)
    interpolated_coords.append(coords[-1])
    return interpolated_coords


# 提取路网坐标
road_coordinates = road_df['coordinates'].apply(ast.literal_eval).tolist()

# 将路网坐标展平为一维数组，并保留每个点所属的路段 ID
road_coords_flattened = []
road_ids = []

for idx, coords in enumerate(road_coordinates):
    interpolated_coords = interpolate_line(coords)
    for coord in interpolated_coords:
        road_coords_flattened.append(coord)
        road_ids.append(idx)

# 构建 KDTree
tree = cKDTree(road_coords_flattened)


# 匹配轨迹点到最近的路网点，并获取对应的路段 ID
def match_to_road(traj_coordinates, tree, road_ids):
    matched_points = []
    matched_road_ids = []
    for point in traj_coordinates:
        _, index = tree.query(point)
        matched_points.append(road_coords_flattened[index])
        matched_road_ids.append(road_ids[index])
    return matched_points, matched_road_ids


# 匹配轨迹
matched_coordinates, matched_road_ids = match_to_road(traj_df['coordinates'].tolist(), tree, road_ids)

# 将匹配后的坐标和路段 ID 重新放回 DataFrame
traj_df['matched_coordinates'] = matched_coordinates
traj_df['matched_road_id'] = matched_road_ids
traj_df.to_csv('task4_matched.csv', index=False)

# 可视化
# 将路网数据转换为 GeoDataFrame
road_geometries = road_df['coordinates'].apply(lambda x: LineString(ast.literal_eval(x)))
road_gdf = gpd.GeoDataFrame(road_df, geometry=road_geometries)

# 将轨迹数据转换为 GeoDataFrame
traj_geometries = traj_df['matched_coordinates'].apply(lambda x: Point(x))
traj_gdf = gpd.GeoDataFrame(traj_df, geometry=traj_geometries)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制路网
road_gdf.plot(ax=ax, color='blue', linewidth=1, label='Road Network')

# 绘制匹配的轨迹点
traj_gdf.plot(ax=ax, color='red', markersize=5, label='Matched Trajectory Points')

# 添加图例
ax.legend()

# 设置标题和标签
ax.set_title('Road Network and Matched Trajectory Points')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# 显示图形
plt.show()
