import pandas as pd
import os

# 替换成你的 CSV 文件路径
file_path = 'test_set.csv'

# 读取 CSV
df = pd.read_csv(file_path)

# 获取文件名（不包含扩展名）
base_name = os.path.splitext(os.path.basename(file_path))[0]

# 计算每份的大小
chunk_size = len(df) // 10
remainder = len(df) % 10

start = 0
for i in range(10):
    # 处理余数，确保分得尽可能均匀
    end = start + chunk_size + (1 if i < remainder else 0)
    chunk = df[start:end]
    
    split_file_name = f"{base_name}_split{i+1}.csv"
    chunk.to_csv(split_file_name, index=False)
    
    print(f"{split_file_name}: {len(chunk)} rows")
    
    start = end
