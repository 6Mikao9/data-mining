import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# 数据预处理函数
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    return data


# 特征构造函数
def generate_features(data, history_length=5):
    features = []
    targets = []
    trajectory_ids = []
    for trajectory_id, group in data.groupby('trajectory_id'):
        road_ids = group['matched_road_id'].values
        if len(road_ids) > history_length:
            for i in range(history_length, len(road_ids)):
                features.append(road_ids[i-history_length:i])  # 使用最后 history_length 个 road_id 作为特征
                targets.append(road_ids[i])  # 目标是下一个 road_id
                trajectory_ids.append(trajectory_id)
    return np.array(features), np.array(targets), trajectory_ids


# 新数据预测函数
def predict_next_road_id(model, test_data, history_length=5):
    predictions = []
    trajectory_ids = []
    for trajectory_id, group in test_data.groupby('trajectory_id'):
        road_ids = group['matched_road_id'].values
        if len(road_ids) >= history_length:
            features = road_ids[-history_length:]  # 提取最后 history_length 个 road_id 作为特征
            predicted_next = model.predict([features])[0]  # 预测下一个 road_id
            predictions.append(predicted_next)
            trajectory_ids.append(trajectory_id)
    return trajectory_ids, predictions


# 主程序
if __name__ == "__main__":
    # 训练集预处理
    train_file_path = 'matched.csv'
    train_data = preprocess_data(train_file_path)

    # 测试集预处理
    test_file_path = 'task4_matched.csv'
    test_data = preprocess_data(test_file_path)

    # 构造训练集特征
    history_length = 1  # 使用最近 5 条道路编号作为特征
    X_train, y_train, _ = generate_features(train_data, history_length)

    # 随机森林分类器
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    # 测试集预测
    trajectory_ids, predicted_next_road_ids = predict_next_road_id(
        model, test_data, history_length
    )

    # 打印预测效果
    results = pd.DataFrame({
        'trajectory_id': trajectory_ids,
        'predicted_next_road_id': predicted_next_road_ids
    })

    # 保存预测结果
    results.to_csv('predicted_road_id.csv', index=False)
    print("预测结果已保存到 predicted_road_id.csv")
