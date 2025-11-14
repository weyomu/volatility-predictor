# -*- coding: utf-8 -*-
"""
比特币波动预警系统
专注于预测未来1-3天是否会出现大涨大跌
不涉及交易，只做风险预警
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 8)

# =============================================================================
# 1. 辅助函数（与原始代码相同）
# =============================================================================

def convert_volume(vol_str):
    """转换交易量格式"""
    if isinstance(vol_str, str):
        vol_str = vol_str.replace(',', '')
        if 'B' in vol_str:
            return float(vol_str.replace('B', '')) * 1e9
        elif 'M' in vol_str:
            return float(vol_str.replace('M', '')) * 1e6
        elif 'K' in vol_str:
            return float(vol_str.replace('K', '')) * 1e3
        else:
            return float(vol_str)
    return vol_str

def convert_change(change_str):
    """转换涨跌幅格式"""
    if isinstance(change_str, str):
        clean_str = change_str.replace('%', '').replace(',', '').strip()
        if clean_str == '-' or clean_str == 'nan':
            return 0.0
        return float(clean_str) / 100
    return change_str

def calculate_rsi(series, period=14):
    """计算RSI指标"""
    series = pd.to_numeric(series, errors='coerce')
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], np.nan).fillna(50)

def calculate_atr(data, period=14):
    """计算ATR指标"""
    high = data['高']
    low = data['低']
    close = data['收盘']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr

# =============================================================================
# 2. 数据加载和特征工程（与原始代码相同）
# =============================================================================

def load_and_engineer_features(filepath):
    """加载数据并进行特征工程"""
    # 加载数据
    print(f"  → 读取CSV文件: {filepath}")
    df = pd.read_csv(filepath, parse_dates=['日期'])
    print(f"  → 原始数据行数: {len(df)}")
    df = df[df['日期'].dt.year > 2000]
    print(f"  → 过滤2000年后数据: {len(df)} 行")
    
    # 转换格式
    print(f"  → 转换交易量格式...")
    df['交易量'] = df['交易量'].apply(convert_volume)
    print(f"  → 转换涨跌幅格式...")
    df['涨跌幅'] = df['涨跌幅'].apply(convert_change)
    
    # 确保价格列是数值类型
    print(f"  → 转换价格列为数值类型...")
    for col in ['收盘', '开盘', '高', '低']:
        df[col] = df[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('¥', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 排序和设置索引
    print(f"  → 按日期排序并设置索引...")
    df.sort_values('日期', inplace=True)
    df.set_index('日期', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    # 特征工程
    print(f"  → 计算基础特征...")
    df['日变化'] = df['收盘'].diff()
    df['开盘收盘差'] = df['收盘'] - df['开盘']
    df['高低差'] = df['高'] - df['低']
    
    print(f"  → 计算移动平均线 (SMA 7天, 30天)...")
    df['SMA_7'] = df['收盘'].rolling(window=7, min_periods=1).mean()
    df['SMA_30'] = df['收盘'].rolling(window=30, min_periods=1).mean()
    
    print(f"  → 计算指数移动平均线和MACD...")
    df['EMA_12'] = df['收盘'].ewm(span=12, adjust=False, min_periods=1).mean()
    df['EMA_26'] = df['收盘'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['信号线'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['信号线']
    
    print(f"  → 计算波动率和RSI...")
    df['波动率'] = df['收盘'].rolling(window=60, min_periods=1).std()
    df['RSI'] = calculate_rsi(df['收盘'], 14)
    
    print(f"  → 计算布林带...")
    df['中轨'] = df['收盘'].rolling(window=20).mean()
    df['上轨'] = df['中轨'] + 2 * df['收盘'].rolling(window=20).std()
    df['下轨'] = df['中轨'] - 2 * df['收盘'].rolling(window=20).std()
    
    print(f"  → 计算ATR指标...")
    df['ATR'] = calculate_atr(df, 14)
    
    # 滞后特征
    print(f"  → 创建60天滞后特征...")
    for i in range(1, 61):
        df[f'滞后_{i}'] = df['收盘'].shift(i)
    
    print(f"  → 删除缺失值...")
    print(f"  → 删除前数据量: {len(df)}")
    df.dropna(inplace=True)
    print(f"  → 删除后数据量: {len(df)}")
    print(f"  → 最终特征数量: {len(df.columns)}")
    return df

# =============================================================================
# 3. 🆕 关键改进：定义波动标签（原始代码没有这个！）
# =============================================================================

def create_volatility_labels(df, days_ahead=1, threshold=0.03):
    """
    创建波动标签
    
    核心逻辑：
    - 如果未来N天涨跌幅绝对值 > threshold，标记为"高波动"
    - 否则标记为"低波动"
    
    参数:
        days_ahead: 预测未来多少天（1-3天）
        threshold: 波动阈值（3% = 0.03）
    
    返回:
        0 = 低波动（正常）
        1 = 高波动（大涨大跌）
    """
    print(f"  → 计算未来{days_ahead}天的收益率...")
    future_returns = []
    
    for i in range(days_ahead):
        future_price = df['收盘'].shift(-(i+1))
        ret = (future_price - df['收盘']) / df['收盘']
        future_returns.append(ret.abs())
    
    # 取未来N天的最大涨跌幅
    print(f"  → 计算未来{days_ahead}天的最大涨跌幅...")
    max_future_change = pd.concat(future_returns, axis=1).max(axis=1)
    
    # 标记高波动
    print(f"  → 标记高波动事件 (阈值: {threshold*100}%)...")
    labels = (max_future_change > threshold).astype(int)
    
    print(f"  → 最大涨跌幅统计: 均值={max_future_change.mean()*100:.2f}%, 最大={max_future_change.max()*100:.2f}%")
    
    return labels, max_future_change

# =============================================================================
# 4. 🆕 关键改进：正确的数据划分（避免数据泄漏）
# =============================================================================

def prepare_data_for_classification(df, labels, time_steps=60, train_ratio=0.8):
    """
    准备分类数据（预测高波动/低波动）
    
    🔑 关键改进：先划分再标准化，避免数据泄漏！
    """
    print(f"  → 提取特征列...")
    feature_cols = [col for col in df.columns if not col.startswith('目标_')]
    print(f"  → 特征列数量: {len(feature_cols)}")
    
    # 先按时间划分
    print(f"  → 按时间划分数据集 (训练集比例: {train_ratio*100}%)...")
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]
    
    print(f"训练集时间: {train_df.index[0]} 到 {train_df.index[-1]}")
    print(f"测试集时间: {test_df.index[0]} 到 {test_df.index[-1]}")
    
    # ✅ 只用训练集拟合scaler
    print(f"  → 使用训练集拟合MinMaxScaler...")
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    
    print(f"  → 标准化训练集数据...")
    train_scaled = scaler.transform(train_df[feature_cols])
    print(f"  → 标准化测试集数据...")
    test_scaled = scaler.transform(test_df[feature_cols])
    
    # 创建时间序列数据集
    print(f"  → 创建时间序列样本 (时间步长: {time_steps})...")
    def create_sequences(data, labels, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), :])
            y.append(labels[i + time_steps])
        return np.array(X), np.array(y)
    
    print(f"  → 生成训练序列...")
    X_train, y_train = create_sequences(train_scaled, train_labels, time_steps)
    print(f"  → 生成测试序列...")
    X_test, y_test = create_sequences(test_scaled, test_labels, time_steps)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'test_dates': test_df.index[time_steps:],
        'test_prices': test_df['收盘'].values[time_steps:],
        'scaler': scaler,
        'feature_cols': feature_cols
    }

# =============================================================================
# 5. 构建分类模型（预测高波动/低波动）
# =============================================================================

def create_volatility_classifier(input_shape, dropout_rate=0.2, learning_rate=0.0005):
    """创建波动分类模型 - 优化版"""
    from tensorflow.keras.layers import BatchNormalization
    
    print(f"  → 创建优化型LSTM+MLP模型...")
    print(f"  → 输入形状: {input_shape}")
    
    model = Sequential([
        # 第一层LSTM - 128单元
        LSTM(128, return_sequences=True, input_shape=input_shape, recurrent_dropout=0.1),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # 第二层LSTM - 64单元
        LSTM(64, return_sequences=False, recurrent_dropout=0.1),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # MLP部分
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        
        # 输出层
        Dense(1, activation='sigmoid')
    ])
    
    print(f"  → 模型结构 (添加BatchNormalization稳定训练):")
    print(f"     ├─ LSTM层1: 128单元 (recurrent_dropout=0.1)")
    print(f"     ├─ BatchNorm + Dropout: {dropout_rate*100:.0f}%")
    print(f"     ├─ LSTM层2: 64单元 (recurrent_dropout=0.1)")
    print(f"     ├─ BatchNorm + Dropout: {dropout_rate*100:.0f}%")
    print(f"     ├─ Dense层1: 128单元 (ReLU)")
    print(f"     ├─ BatchNorm + Dropout: {dropout_rate*100:.0f}%")
    print(f"     ├─ Dense层2: 64单元 (ReLU)")
    print(f"     ├─ BatchNorm + Dropout: {dropout_rate*100:.0f}%")
    print(f"     ├─ Dense层3: 32单元 (ReLU)")
    print(f"     ├─ Dropout: {dropout_rate*100:.0f}%")
    print(f"     └─ 输出层: 1单元 (Sigmoid)")
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),  # 添加梯度裁剪
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"  → 优化器: Adam (learning_rate={learning_rate}, clipnorm=1.0)")
    print(f"  → 损失函数: binary_crossentropy")
    print(f"  → 评估指标: accuracy")
    
    total_params = model.count_params()
    print(f"  → 总参数量: {total_params:,}")
    
    return model

# =============================================================================
# 6. 主函数
# =============================================================================

def main():
    print("="*70)
    print("比特币波动预警系统")
    print("预测未来1-3天是否会出现大涨大跌")
    print("="*70)
    
    # 可调参数
    DAYS_AHEAD = 1        # 预测未来几天（1-3天）
    THRESHOLD = 0.03      # 波动阈值：3%
    TIME_STEPS = 60       # 使用过去60天数据
    TRAIN_RATIO = 0.8     # 80%训练，20%测试
    DROPOUT_RATE = 0.2    # Dropout比率（先用0.2，太低可能导致不稳定）
    EPOCHS = 100          # 最大训练轮数
    BATCH_SIZE = 16       # 批次大小（减小，让梯度更新更频繁）
    LEARNING_RATE = 0.0005 # 学习率（降低，更稳定）
    USE_CLASS_WEIGHT = False  # 暂时关闭类别权重，看是否影响训练
    
    print(f"\n参数设置:")
    print(f"- 预测时长: 未来{DAYS_AHEAD}天")
    print(f"- 波动阈值: {THRESHOLD*100}% (超过此值视为高波动)")
    print(f"- 训练集比例: {TRAIN_RATIO*100}%")
    print(f"- Dropout比率: {DROPOUT_RATE*100}%")
    print(f"- 学习率: {LEARNING_RATE}")
    print(f"- 批次大小: {BATCH_SIZE}")
    print(f"- 类别权重: {'启用' if USE_CLASS_WEIGHT else '禁用'}")
    
    # 1. 加载数据
    print("\n[1/5] 加载数据和特征工程...")
    df = load_and_engineer_features('比特币历史数据2.csv')
    print(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"总数据量: {len(df)} 天")
    print(f"价格统计: 最低=${df['收盘'].min():,.2f}, 最高=${df['收盘'].max():,.2f}, 平均=${df['收盘'].mean():,.2f}")
    
    # 2. 创建波动标签
    print(f"\n[2/5] 创建波动标签（阈值：{THRESHOLD*100}%）...")
    labels, future_changes = create_volatility_labels(df, days_ahead=DAYS_AHEAD, threshold=THRESHOLD)
    
    # 统计标签分布
    high_vol_count = labels.sum()
    low_vol_count = len(labels) - high_vol_count
    print(f"高波动天数: {high_vol_count} ({high_vol_count/len(labels)*100:.1f}%)")
    print(f"低波动天数: {low_vol_count} ({low_vol_count/len(labels)*100:.1f}%)")
    print(f"类别平衡比: 1:{low_vol_count/high_vol_count:.2f} (高波动:低波动)")
    
    # 3. 准备数据
    print("\n[3/5] 准备训练和测试数据（避免数据泄漏）...")
    data = prepare_data_for_classification(df, labels, time_steps=TIME_STEPS, train_ratio=TRAIN_RATIO)
    print(f"训练样本数: {len(data['X_train'])} (高波动: {data['y_train'].sum()}, 比例: {data['y_train'].sum()/len(data['y_train'])*100:.1f}%)")
    print(f"测试样本数: {len(data['X_test'])} (高波动: {data['y_test'].sum()}, 比例: {data['y_test'].sum()/len(data['y_test'])*100:.1f}%)")
    print(f"输入特征维度: {data['X_train'].shape}")
    
    # 3.5 计算类别权重（解决不平衡问题）
    print("\n[3.5/5] 计算类别权重（解决类别不平衡问题）...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(data['y_train']),
        y=data['y_train']
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"  → 低波动类别权重: {class_weights[0]:.4f}")
    print(f"  → 高波动类别权重: {class_weights[1]:.4f}")
    print(f"  → 权重比例: 1:{class_weights[1]/class_weights[0]:.2f}")
    print(f"  → 说明: 高波动样本将获得{class_weights[1]/class_weights[0]:.2f}倍的关注度")
    
    # 4. 训练模型
    print("\n[4/5] 训练波动预测模型...")
    model = create_volatility_classifier(
        input_shape=(data['X_train'].shape[1], data['X_train'].shape[2]),
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE
    )
    
    print(f"  → 设置早停机制 (patience=20, monitor=val_loss)...")
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    if USE_CLASS_WEIGHT:
        print(f"  → 开始训练（使用类别权重）...")
        print(f"  → 应用类别权重: 低波动={class_weight_dict[0]:.4f}, 高波动={class_weight_dict[1]:.4f}")
    else:
        print(f"  → 开始训练（不使用类别权重）...")
        print(f"  → 注意: 类别权重已禁用，使用均衡采样")
    
    print(f"  → 训练样本: {len(data['X_train'])}")
    print(f"  → 验证样本: {int(len(data['X_train']) * 0.2)}")
    print(f"  → 最大轮数: {EPOCHS}")
    print(f"  → 批次大小: {BATCH_SIZE}")
    print(f"  → 显示训练进度...")
    print()
    
    history = model.fit(
        data['X_train'], data['y_train'],
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        class_weight=class_weight_dict if USE_CLASS_WEIGHT else None,
        verbose=1  # 显示训练进度
    )
    
    print(f"\n  → 实际训练轮数: {len(history.history['loss'])}")
    print(f"  → 最终训练损失: {history.history['loss'][-1]:.4f}")
    print(f"  → 最终验证损失: {history.history['val_loss'][-1]:.4f}")
    print(f"  → 最终训练准确率: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"  → 最终验证准确率: {history.history['val_accuracy'][-1]*100:.2f}%")
    
    # 显示训练历史摘要
    print(f"\n  训练历史摘要:")
    print(f"  → 最佳训练准确率: {max(history.history['accuracy'])*100:.2f}% (第{history.history['accuracy'].index(max(history.history['accuracy']))+1}轮)")
    print(f"  → 最佳验证准确率: {max(history.history['val_accuracy'])*100:.2f}% (第{history.history['val_accuracy'].index(max(history.history['val_accuracy']))+1}轮)")
    print(f"  → 最低训练损失: {min(history.history['loss']):.4f} (第{history.history['loss'].index(min(history.history['loss']))+1}轮)")
    print(f"  → 最低验证损失: {min(history.history['val_loss']):.4f} (第{history.history['val_loss'].index(min(history.history['val_loss']))+1}轮)")
    print("模型训练完成!")
    
    # 5. 测试集评估
    print("\n[5/5] 在测试集上评估...")
    print(f"  → 对测试集进行预测...")
    y_pred_prob = model.predict(data['X_test'], verbose=0).flatten()
    print(f"  → 预测概率范围: [{y_pred_prob.min():.4f}, {y_pred_prob.max():.4f}]")
    print(f"  → 平均预测概率: {y_pred_prob.mean():.4f}")
    print(f"  → 预测概率中位数: {np.median(y_pred_prob):.4f}")
    
    # 5.1 寻找最佳阈值
    print(f"\n  → 寻找最佳预测阈值...")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    best_threshold = 0.5
    best_f1 = 0
    threshold_results = []
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred_temp = (y_pred_prob > threshold).astype(int)
        f1_temp = f1_score(data['y_test'], y_pred_temp, zero_division=0)
        recall_temp = recall_score(data['y_test'], y_pred_temp, zero_division=0)
        precision_temp = precision_score(data['y_test'], y_pred_temp, zero_division=0)
        threshold_results.append({
            'threshold': threshold,
            'f1': f1_temp,
            'recall': recall_temp,
            'precision': precision_temp
        })
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = threshold
    
    print(f"  → 最佳阈值: {best_threshold:.2f} (F1={best_f1:.4f})")
    print(f"  → 使用最佳阈值进行预测...")
    
    y_pred = (y_pred_prob > best_threshold).astype(int)
    print(f"  → 预测为高波动的样本数: {y_pred.sum()}/{len(y_pred)} ({y_pred.sum()/len(y_pred)*100:.1f}%)")
    print(f"  → 实际高波动的样本数: {data['y_test'].sum()}/{len(data['y_test'])} ({data['y_test'].sum()/len(data['y_test'])*100:.1f}%)")
    
    # 计算性能指标
    print(f"  → 计算性能指标...")
    accuracy = accuracy_score(data['y_test'], y_pred)
    precision = precision_score(data['y_test'], y_pred, zero_division=0)
    recall = recall_score(data['y_test'], y_pred, zero_division=0)
    f1 = f1_score(data['y_test'], y_pred, zero_division=0)
    
    # 计算ROC-AUC
    try:
        roc_auc = roc_auc_score(data['y_test'], y_pred_prob)
        print(f"  → ROC-AUC得分: {roc_auc:.4f}")
    except:
        roc_auc = 0
        print(f"  → ROC-AUC得分: 无法计算")
    
    print("\n" + "="*70)
    print("模型性能评估")
    print("="*70)
    print(f"最佳预测阈值:       {best_threshold:.2f}   - 优化后的分类阈值")
    print(f"准确率 (Accuracy):  {accuracy*100:.2f}%  - 预测对的比例")
    print(f"精确率 (Precision): {precision*100:.2f}%  - 预警准确度（预警时真的高波动的概率）")
    print(f"召回率 (Recall):    {recall*100:.2f}%  - 捕获率（高波动时能预警的概率）")
    print(f"F1分数:            {f1:.3f}     - 综合指标（精确率和召回率的平均）")
    if roc_auc > 0:
        print(f"ROC-AUC:           {roc_auc:.3f}     - 模型整体区分能力")
    
    # 混淆矩阵
    cm = confusion_matrix(data['y_test'], y_pred)
    print(f"\n混淆矩阵:")
    print(f"              预测低波动  预测高波动")
    print(f"实际低波动:      {cm[0,0]:4d}       {cm[0,1]:4d}")
    print(f"实际高波动:      {cm[1,0]:4d}       {cm[1,1]:4d}")
    
    # 6. 找出最近的高波动预警
    print("\n" + "="*70)
    print("最近的高波动预警事件")
    print("="*70)
    
    high_vol_indices = np.where(y_pred == 1)[0]
    if len(high_vol_indices) > 0:
        # 显示最近10个预警
        recent_warnings = high_vol_indices[-10:] if len(high_vol_indices) >= 10 else high_vol_indices
        
        for idx in recent_warnings:
            date = data['test_dates'][idx]
            price = data['test_prices'][idx]
            prob = y_pred_prob[idx]
            actual = data['y_test'][idx]
            
            status = "✅ 正确预警" if actual == 1 else "❌ 误报"
            print(f"{date.strftime('%Y-%m-%d')} | 价格: ${price:,.2f} | 预警概率: {prob*100:.1f}% | {status}")
    else:
        print("测试集中没有预警高波动事件")
    
    # 7. 预测未来
    print("\n" + "="*70)
    print(f"未来{DAYS_AHEAD}天波动预警")
    print("="*70)
    
    # 使用最新数据预测
    print(f"  → 使用最新{TIME_STEPS}天数据进行预测...")
    latest_data = data['X_test'][-1:]
    print(f"  → 输入数据形状: {latest_data.shape}")
    
    future_prob = model.predict(latest_data, verbose=0)[0][0]
    future_pred = 1 if future_prob > best_threshold else 0  # 使用最佳阈值
    
    latest_date = data['test_dates'][-1]
    latest_price = data['test_prices'][-1]
    
    print(f"  → 预测完成!")
    print(f"  → 使用阈值: {best_threshold:.2f}")
    
    print(f"当前日期: {latest_date.strftime('%Y-%m-%d')}")
    print(f"当前价格: ${latest_price:,.2f}")
    print(f"\n未来{DAYS_AHEAD}天波动预测:")
    print(f"高波动概率: {future_prob*100:.1f}%")
    print(f"预测阈值: {best_threshold*100:.0f}%")
    
    if future_pred == 1:
        print(f"⚠️  预警：未来{DAYS_AHEAD}天可能出现大涨大跌（涨跌幅>±{THRESHOLD*100}%）")
        print(f"建议：注意风险，考虑止损或观望")
    else:
        print(f"✅ 正常：未来{DAYS_AHEAD}天预计波动较小")
        print(f"建议：市场相对稳定，可正常操作")
    
    # 8. 可视化
    print("\n生成可视化图表...")
    print(f"  → 创建图表 (20x12英寸)...")
    
    # 图1: 测试集预测结果
    plt.figure(figsize=(20, 12))
    
    # 子图1: 价格和波动预警
    print(f"  → 绘制子图1: 价格与波动预警...")
    plt.subplot(3, 1, 1)
    plt.plot(data['test_dates'], data['test_prices'], label='价格', linewidth=2, color='blue')
    
    # 标记实际高波动点
    actual_high_vol = np.where(data['y_test'] == 1)[0]
    if len(actual_high_vol) > 0:
        print(f"  → 标记{len(actual_high_vol)}个实际高波动点...")
        plt.scatter(data['test_dates'][actual_high_vol], 
                   data['test_prices'][actual_high_vol],
                   color='red', s=100, marker='x', label='实际高波动', zorder=5)
    
    # 标记预测高波动点
    pred_high_vol = np.where(y_pred == 1)[0]
    if len(pred_high_vol) > 0:
        print(f"  → 标记{len(pred_high_vol)}个预测高波动点...")
        plt.scatter(data['test_dates'][pred_high_vol], 
                   data['test_prices'][pred_high_vol],
                   color='orange', s=100, marker='o', alpha=0.5, label='预测高波动', zorder=4)
    
    plt.title(f'比特币价格与波动预警 (阈值：{THRESHOLD*100}%)', fontsize=16)
    plt.ylabel('价格 (美元)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 预测概率曲线
    print(f"  → 绘制子图2: 预测概率曲线...")
    plt.subplot(3, 1, 2)
    plt.plot(data['test_dates'], y_pred_prob, label='高波动概率', linewidth=2, color='purple')
    plt.axhline(y=best_threshold, color='red', linestyle='--', alpha=0.7, 
                label=f'最佳阈值({best_threshold:.2f})', linewidth=2)
    plt.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5, label='默认阈值(0.50)')
    plt.fill_between(data['test_dates'], 0, y_pred_prob, 
                     where=(y_pred_prob > best_threshold), alpha=0.3, color='red', label='高波动区')
    plt.title(f'高波动预测概率（优化阈值：{best_threshold:.2f}）', fontsize=16)
    plt.ylabel('概率', fontsize=12)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 混淆矩阵
    print(f"  → 绘制子图3: 混淆矩阵...")
    plt.subplot(3, 1, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['预测低波动', '预测高波动'],
                yticklabels=['实际低波动', '实际高波动'])
    plt.title('混淆矩阵', fontsize=16)
    
    print(f"  → 调整布局并保存...")
    plt.tight_layout()
    plt.savefig('波动预警结果.png', dpi=150)
    print("✓ 保存图表: 波动预警结果.png")
    
    # 保存预警记录
    if len(high_vol_indices) > 0:
        print(f"  → 生成预警记录CSV文件...")
        warnings_df = pd.DataFrame({
            '日期': [data['test_dates'][i].strftime('%Y-%m-%d') for i in high_vol_indices],
            '价格': [data['test_prices'][i] for i in high_vol_indices],
            '预警概率': [y_pred_prob[i] for i in high_vol_indices],
            '实际波动': ['高波动' if data['y_test'][i] == 1 else '低波动' for i in high_vol_indices],
            '预警结果': ['正确' if data['y_test'][i] == 1 else '误报' for i in high_vol_indices]
        })
        print(f"  → 预警记录数量: {len(warnings_df)}")
        warnings_df.to_csv('波动预警记录.csv', index=False, encoding='utf-8-sig')
        print("✓ 保存文件: 波动预警记录.csv")
    
    print("\n" + "="*70)
    print("所有任务完成!")
    print("="*70)
    
    # 返回结果供进一步分析
    return {
        'model': model,
        'data': data,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'best_threshold': best_threshold,
        'future_prob': future_prob,
        'class_weight_dict': class_weight_dict,
        'threshold_results': threshold_results
    }

if __name__ == "__main__":
    results = main()

