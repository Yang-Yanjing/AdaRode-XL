import pandas as pd
from sklearn.model_selection import train_test_split

# === 加载数据 ===
df = pd.read_csv('./PIK.csv')

# === 统计各类标签数量（0: normal, 1: sql, 2: xss）===
label_counts = df['Label'].value_counts().sort_index()
print("📊 样本分布:")
for label, count in label_counts.items():
    label_name = {0: 'normal', 1: 'sql injection', 2: 'xss'}.get(label, 'unknown')
    print(f"{label_name:<15}: {count}")

# === 划分数据集（8:1:1） stratify 保持标签比例 ===
train_val, test = train_test_split(df, test_size=0.1, stratify=df['Label'], random_state=42)
train, val = train_test_split(train_val, test_size=0.1111, stratify=train_val['Label'], random_state=42)  # 0.1111*0.9 ≈ 0.1

# === 输出各集大小 ===
print("\n📦 数据划分结果:")
print(f"Train     : {len(train)}")
print(f"Validation: {len(val)}")
print(f"Test      : {len(test)}")

# === 保存数据集 ===
train.to_csv('./train.csv', index=False)
val.to_csv('./val.csv', index=False)
test.to_csv('./test.csv', index=False)
