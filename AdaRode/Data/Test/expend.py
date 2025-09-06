import pandas as pd
import os

# 配置
folder = "./"  # 你的 test_split 文件所在的文件夹
num_splits = 10       # 总分片数量
window_size = 5       # 每次合并几个文件

# 循环生成新的组合
for i in range(1, num_splits + 1):
    dfs = []
    for j in range(window_size):
        idx = ((i - 1 + j) % num_splits) + 1  # 循环取编号
        file_path = os.path.join(folder, f"test_set_split{idx}.csv")
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(folder, f"test_set_expand{i}.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"生成 {output_path} ，共 {len(merged_df)} 条数据")

print("全部完成！")
