import os
import pandas as pd

# 设置路径
mri_folder = r'/data3/wangchangmiao/shenxy/ADNI/AIBL/MRI'
pet_folder = r'/data3/wangchangmiao/shenxy/ADNI/AIBL/PET'
csv_file = r'/data3/wangchangmiao/shenxy/ADNI/AIBL/AIBL_Clinical.csv'
output_file = 'intersected_clinical_data.csv'

# 获取 MRI 文件夹中的所有编号（去掉 .nii 后缀）
mri_files = {f.split('.')[0] for f in os.listdir(mri_folder) if f.endswith('.nii')}

# 获取 PET 文件夹中的所有编号
pet_files = {f.split('.')[0] for f in os.listdir(pet_folder) if f.endswith('.nii')}

# 读取 CSV 文件
df = pd.read_csv(csv_file)

# 确保第一列是 ID，可以根据实际列名修改
id_column = df.columns[0]  # 默认取第一列作为 ID

# 将 CSV 中的 ID 转换为字符串类型（防止数字编号被误处理）
df[id_column] = df[id_column].astype(str)

# 获取 CSV 中的 ID 集合
csv_ids = set(df[id_column])

# 找出三者交集
common_ids = mri_files & pet_files & csv_ids

# 筛选出交集 ID 的行
filtered_df = df[df[id_column].isin(common_ids)]

# 保存为新的 CSV 文件
filtered_df.to_csv(output_file, index=False)

print(f"交集 ID 数量: {len(common_ids)}")
print(f"提取完成，结果已保存到 {output_file}")