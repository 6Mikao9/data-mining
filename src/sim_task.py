import pandas as pd
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString, Point

# 读取轨迹数据
matched_traj_df = pd.read_csv('matched_traj.csv')

# 读取需要匹配的轨迹数据
sim_task_df = pd.read_csv('../data/sim_task.csv')

# 提取匹配后的轨迹坐标
matched_trajectories = matched_traj_df.groupby('trajectory_id')['matched_coordinates'].apply(list).reset_index(name='matched_coordinates')

# 提取需要匹配的轨迹坐标
sim_trajectories = sim_task_df.groupby('trajectory_id')['coordinates'].apply(list).reset_index(name='coordinates')

def frechet_distance(traj1, traj2):
    """
    计算两条轨迹的Frechet距离
    """
    traj1 = np.array([eval(coord) for coord in traj1])
    traj2 = np.array([eval(coord) for coord in traj2])
    return max(directed_hausdorff(traj1, traj2)[0], directed_hausdorff(traj2, traj1)[0])


def find_most_similar_trajectory(query_traj, trajectories):
    """
    检索最相似的轨迹
    """
    distances = []
    for index, row in trajectories.iterrows():
        distance = frechet_distance(query_traj['coordinates'], row['matched_coordinates'])
        distances.append((row['trajectory_id'], distance))

    distances.sort(key=lambda x: x[1])
    return distances[0]  # 返回最相似的轨迹


# 检索每个需要匹配的轨迹的最相似轨迹
results = []
for index, row in sim_trajectories.iterrows():
    most_similar_traj = find_most_similar_trajectory(row, matched_trajectories)
    results.append((row['trajectory_id'], most_similar_traj[0], most_similar_traj[1]))

# 将结果保存到DataFrame
result_df = pd.DataFrame(results, columns=['sim_trajectory_id', 'matched_trajectory_id', 'distance'])
print(result_df)

# 将轨迹数据转换为GeoDataFrame
def create_geo_dataframe(trajectories, trajectory_id_col, coordinates_col):
    geo_data = []
    for index, row in trajectories.iterrows():
        traj_geom = [Point(eval(coord)) for coord in row[coordinates_col]]
        geo_data.append({'trajectory_id': row[trajectory_id_col], 'geometry': LineString(traj_geom)})
    return gpd.GeoDataFrame(geo_data)

# 创建查询轨迹的GeoDataFrame
query_traj_gdf = create_geo_dataframe(sim_trajectories, 'trajectory_id', 'coordinates')

# 创建匹配轨迹的GeoDataFrame
matched_traj_gdf = create_geo_dataframe(matched_trajectories, 'trajectory_id', 'matched_coordinates')

# 创建图形
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制查询轨迹
query_traj_gdf.plot(ax=ax, color='red', linewidth=2, label='Query Trajectories')

# 绘制最相似的轨迹
for index, row in result_df.iterrows():
    matched_traj = matched_traj_gdf[matched_traj_gdf['trajectory_id'] == row['matched_trajectory_id']]
    matched_traj.plot(ax=ax, color='green', linewidth=2, linestyle='--', label=f'Similar Trajectory {row["matched_trajectory_id"]}')

# 添加图例
ax.legend()

# 设置标题和标签
ax.set_title('Query Trajectories and Similar Trajectories')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# 显示图形
plt.show()