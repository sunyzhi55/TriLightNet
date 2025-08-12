import nibabel as nib
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from skimage import transform as skt
import numpy as np

def get_clinical(sub_id, clin_df):
    '''Gets clinical features vector by searching dataframe for image id'''
    # 用-1初始化数组，表示缺失值
    clinical = np.full(9, -1.0)

    if sub_id in clin_df["PTID"].values:
        row = clin_df.loc[clin_df["PTID"] == sub_id].iloc[0]

        # GENDER (1表示Male, 2表示Female，缺失默认设置为 -1)
        if pd.isnull(row["PTGENDER"]):
            clinical[0] = -1
        else:
            clinical[0] = 1 if row["PTGENDER"] == 1 else (2 if row["PTGENDER"] == 2 else 0)

        # AGE (用-1标记缺失值)
        clinical[1] = row["AGE"] if not pd.isnull(row["AGE"]) else 0

        # Education (用-1标记缺失值)
        clinical[2] = row["PTEDUCAT"] if not pd.isnull(row["PTEDUCAT"]) else 0

        # FDG_bl (用-1标记缺失值)
        clinical[3] = row["FDG_bl"] if not pd.isnull(row["FDG_bl"]) else 0

        # TAU_bl (用-1标记缺失值)
        clinical[4] = row["TAU_bl"] if not pd.isnull(row["TAU_bl"]) else 0

        # PTAU_bl (用-1标记缺失值)
        clinical[5] = row["PTAU_bl"] if not pd.isnull(row["PTAU_bl"]) else 0

        # APOE4 (保留原有处理方式，缺失则处理为 -1)
        apoe4_allele = row["APOE4"]
        if pd.isnull(apoe4_allele):
            clinical[6], clinical[7], clinical[8] = 0, 0, 0  # 标记缺失值
        elif apoe4_allele == 0:
            clinical[6], clinical[7], clinical[8] = 1, 0, 0
        elif apoe4_allele == 1:
            clinical[6], clinical[7], clinical[8] = 0, 1, 0
        elif apoe4_allele == 2:
            clinical[6], clinical[7], clinical[8] = 0, 0, 1

    return clinical


class NoNan:  # Python3默认继承object类
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        nan_mask = np.isnan(data)
        data[nan_mask] = 0.0
        data = np.expand_dims(data, axis=0)
        data /= np.max(data)
        return data  # 返回预处理后的图像


class Numpy2Torch:  # Python3默认继承object类
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        data = torch.from_numpy(data)
        return data  # 返回预处理后的图像


class Resize:  # Python3默认继承object类
    def __init__(self, output_shape):
        self.output_shape = output_shape
    def __call__(self, data):  # __call___，让类实例变成一个可以被调用的对象，像函数
        data = skt.resize(data, output_shape=self.output_shape, order=1)
        return data  # 返回预处理后的图像

# MRI + PET + CLI + Two Label 数据集
class MriPetCliDatasetWithTowLabel(Dataset):
    def __init__(self, mri_dir, pet_dir, cli_dir, csv_file, resize_shape=(96, 128, 96), valid_group=("pMCI", "sMCI")):
        """
        Args:
            mri_dir (string or Path): MRI 文件所在的文件夹路径。
            pet_dir (string or Path): PET 文件所在的文件夹路径。
            cli_dir (string or Path): Clinical 文件所在的文件夹路径。
            csv_file (string or Path): CSV 文件路径，其中第一列是文件名，第二列是标签。
            transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.mri_dir = Path(mri_dir)
        if pet_dir  == '':
            self.pet_dir = ''
        else:
            self.pet_dir = Path(pet_dir)
        self.cli_dir = pd.read_csv(cli_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2,  'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        self.transform = transforms.Compose([
            Resize(resize_shape),
            NoNan(),
            Numpy2Torch(),
            # transforms.Normalize([0.5], [0.5])
        ])

        # 过滤只保留 valid_group 中的有效数据
        self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()
        self.convert_label = lambda x: [1, 0] if x == 0 else [0, 1]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # 获取过滤后的索引
        filtered_idx = self.filtered_indices[idx]

        # 获取对应的文件名和标签
        img_name = self.labels_df.iloc[filtered_idx, 0]
        label_str = self.labels_df.iloc[filtered_idx, 1]  # 标签

        # MRI 文件路径
        mri_img_path = self.mri_dir / (img_name + '.nii')
        mri_img_numpy = nib.load(str(mri_img_path)).get_fdata()
        mri_img_torch = self.transform(mri_img_numpy)
        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1
        # 将一维的label转换为二维
        label_2d = self.convert_label(label)

        clinical_features = get_clinical(img_name, self.cli_dir)
        clin_tab_torch = torch.from_numpy(clinical_features).float()


        # PET 文件路径
        pet_img_path = self.pet_dir / (img_name + '.nii')
        # print('pet_img_path', pet_img_path)
        pet_img_numpy = nib.load(str(pet_img_path)).get_fdata()
        pet_img_torch = self.transform(pet_img_numpy)
        batch = {
            "mri": mri_img_torch.float(),
            "pet": pet_img_torch.float(),
            "clinical": clin_tab_torch,
            "label": label,
            "label_2d": torch.Tensor(label_2d)
        }
        return batch

# MRI + PET + CLI 数据集
class MriPetCliDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, cli_dir, csv_file, resize_shape=(96, 128, 96), valid_group=("pMCI", "sMCI")):
        """
        Args:
            mri_dir (string or Path): MRI 文件所在的文件夹路径。
            pet_dir (string or Path): PET 文件所在的文件夹路径。
            cli_dir (string or Path): Clinical 文件所在的文件夹路径。
            csv_file (string or Path): CSV 文件路径，其中第一列是文件名，第二列是标签。
            transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.mri_dir = Path(mri_dir)
        if pet_dir  == '':
            self.pet_dir = ''
        else:
            self.pet_dir = Path(pet_dir)
        self.cli_dir = pd.read_csv(cli_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2,  'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        self.transform = transforms.Compose([
            Resize(resize_shape),
            NoNan(),
            Numpy2Torch(),
            # transforms.Normalize([0.5], [0.5])
        ])

        # 过滤只保留 valid_group 中的有效数据
        self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # 获取过滤后的索引
        filtered_idx = self.filtered_indices[idx]

        # 获取对应的文件名和标签
        img_name = self.labels_df.iloc[filtered_idx, 0]
        label_str = self.labels_df.iloc[filtered_idx, 1]  # 标签

        # MRI 文件路径
        mri_img_path = self.mri_dir / (img_name + '.nii')
        mri_img_numpy = nib.load(str(mri_img_path)).get_fdata()
        mri_img_torch = self.transform(mri_img_numpy)
        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1

        clinical_features = get_clinical(img_name, self.cli_dir)
        clin_tab_torch = torch.from_numpy(clinical_features).float()

        # 只有MRI,没有PET,用于eval阶段
        # if self.pet_dir == '':
        #     return mri_img_torch.float(), label
        # PET 文件路径
        pet_img_path = self.pet_dir / (img_name + '.nii')
        # print('pet_img_path', pet_img_path)
        pet_img_numpy = nib.load(str(pet_img_path)).get_fdata()
        pet_img_torch = self.transform(pet_img_numpy)
        batch = {
            "image_name": img_name,
            "mri": mri_img_torch.float(),
            "pet": pet_img_torch.float(),
            "clinical": clin_tab_torch,
            "label": label
        }

        return batch

class AIBLDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, cli_dir, csv_file, resize_shape=(96, 128, 96), valid_group=("pMCI", "sMCI")):
        self.mri_dir = Path(mri_dir)
        self.pet_dir = Path(pet_dir) if pet_dir else ''
        self.cli_df = pd.read_csv(cli_dir)  # 临床数据
        self.labels_df = pd.read_csv(csv_file)  # 第一列是文件名，第二列是标签
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 1, 'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        self.convert_label = lambda x: [1, 0] if x == 0 else [0, 1]
        self.transform = transforms.Compose([
            Resize(resize_shape),
            NoNan(),
            Numpy2Torch(),
        ])

        # 过滤只保留 valid_group 中的有效数据
        self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        filtered_idx = self.filtered_indices[idx]
        img_name = str(self.labels_df.iloc[filtered_idx, 0])
        label_str = self.labels_df.iloc[filtered_idx, 1]

        # MRI 加载
        mri_img_path = self.mri_dir / (img_name + '.nii')
        # print(f"mri_img_path: {mri_img_path}")
        mri_img_numpy = nib.load(str(mri_img_path)).get_fdata()
        mri_img_torch = self.transform(mri_img_numpy)

        # 获取临床特征（第2到第6列）
        clinical_row = self.cli_df[self.cli_df.iloc[:, 0].astype(str) == img_name]  # 注意这里使用 astype(str)
        if clinical_row.empty:
            raise ValueError(f"No clinical data found for image: {img_name}")
        clinical_data = clinical_row.iloc[0, 1:6].values
        clinical_tensor = torch.from_numpy(clinical_data.astype(np.float32))

        label = self.groups.get(label_str, -1)
        label_2d = self.convert_label(label)

        # if self.pet_dir == '':
        #     return mri_img_torch.float(), label, clinical_tensor

        # PET 加载
        pet_img_path = self.pet_dir / (img_name + '.nii')
        pet_img_numpy = nib.load(str(pet_img_path)).get_fdata()
        pet_img_torch = self.transform(pet_img_numpy)

        return {
            "image_name": img_name,
            "mri": mri_img_torch.float(),
            "pet": pet_img_torch.float(),
            "clinical": clinical_tensor,
            "label": label,
            "label_2d": torch.Tensor(label_2d)
        }

# MRI + PET 数据集
class MriPetDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, cli_dir, csv_file, resize_shape=(96, 128, 96), valid_group=("pMCI", "sMCI")):
        """
        Args:
            mri_dir (string or Path): MRI 文件所在的文件夹路径。
            pet_dir (string or Path): PET 文件所在的文件夹路径。
            csv_file (string or Path): CSV 文件路径，其中第一列是文件名，第二列是标签。
            transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.mri_dir = Path(mri_dir)
        if pet_dir  == '':
            self.pet_dir = ''
        else:
            self.pet_dir = Path(pet_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2,  'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        self.transform = transforms.Compose([
            Resize(resize_shape),
            NoNan(),
            Numpy2Torch(),
            # transforms.Normalize([0.5], [0.5])
        ])

        # 过滤只保留 valid_group 中的有效数据
        self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # 获取过滤后的索引
        filtered_idx = self.filtered_indices[idx]

        # 获取对应的文件名和标签
        img_name = self.labels_df.iloc[filtered_idx, 0]
        label_str = self.labels_df.iloc[filtered_idx, 1]  # 标签

        # MRI 文件路径
        mri_img_path = self.mri_dir / (img_name + '.nii')
        mri_img_numpy = nib.load(str(mri_img_path)).get_fdata()
        mri_img_torch = self.transform(mri_img_numpy)
        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1

        # 只有MRI,没有PET,用于eval阶段
        # if self.pet_dir == '':
        #     return mri_img_torch.float(), label

        # PET 文件路径
        pet_img_path = self.pet_dir / (img_name + '.nii')
        # print('pet_img_path', pet_img_path)
        pet_img_numpy = nib.load(str(pet_img_path)).get_fdata()
        pet_img_torch = self.transform(pet_img_numpy)
        batch = {
            "mri": mri_img_torch.float(),
            "pet": pet_img_torch.float(),
            "label": label
        }
        return batch

# 单MRI 数据集
class MriDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, cli_dir, csv_file, resize_shape=(96, 128, 96), valid_group=("pMCI", "sMCI")):
        """
        Args:
            mri_dir (string or Path): MRI 文件所在的文件夹路径。
            pet_dir (string or Path): PET 文件所在的文件夹路径。
            csv_file (string or Path): CSV 文件路径，其中第一列是文件名，第二列是标签。
            transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.mri_dir = Path(mri_dir)
        if pet_dir  == '':
            self.pet_dir = ''
        else:
            self.pet_dir = Path(pet_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2,  'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        self.transform = transforms.Compose([
            Resize(resize_shape),
            NoNan(),
            Numpy2Torch(),
            # transforms.Normalize([0.5], [0.5])
        ])

        # 过滤只保留 valid_group 中的有效数据
        self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # 获取过滤后的索引
        filtered_idx = self.filtered_indices[idx]

        # 获取对应的文件名和标签
        img_name = self.labels_df.iloc[filtered_idx, 0]
        label_str = self.labels_df.iloc[filtered_idx, 1]  # 标签

        # MRI 文件路径
        mri_img_path = self.mri_dir / (img_name + '.nii')
        mri_img_numpy = nib.load(str(mri_img_path)).get_fdata()
        mri_img_torch = self.transform(mri_img_numpy)
        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1

        batch = {
            "mri": mri_img_torch.float(),
            "label": label
        }

        return batch

# MRI + CLI 数据集
class MriCliDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, cli_dir, csv_file, resize_shape=(96, 128, 96), valid_group=("pMCI", "sMCI")):
        """
        Args:
            mri_dir (string or Path): MRI 文件所在的文件夹路径。
            pet_dir (string or Path): PET 文件所在的文件夹路径。
            cli_dir (string or Path): Clinical 文件所在的文件夹路径。
            csv_file (string or Path): CSV 文件路径，其中第一列是文件名，第二列是标签。
            transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.mri_dir = Path(mri_dir)
        self.cli_dir = pd.read_csv(cli_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2,  'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        # self.only_tabular = False
        self.transform = transforms.Compose([
            Resize(resize_shape),
            NoNan(),
            Numpy2Torch(),
            # transforms.Normalize([0.5], [0.5])
        ])


        # 过滤只保留 valid_group 中的有效数据
        self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # 获取过滤后的索引
        filtered_idx = self.filtered_indices[idx]

        # 获取对应的文件名和标签
        img_name = self.labels_df.iloc[filtered_idx, 0]
        label_str = self.labels_df.iloc[filtered_idx, 1]  # 标签

        # MRI 文件路径
        mri_img_path = self.mri_dir / (img_name + '.nii')
        mri_img_numpy = nib.load(str(mri_img_path)).get_fdata()
        mri_img_torch = self.transform(mri_img_numpy)
        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1
        clinical_features = get_clinical(img_name, self.cli_dir)
        clin_tab_torch = torch.from_numpy(clinical_features).float()
        batch = {
            "mri": mri_img_torch.float(),
            "clinical": clin_tab_torch,
            "label": label
        }
        return batch

# 自定义 Dataset 类 GM WM PET
class GMWMPETDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, cli_dir, csv_file, resize_shape=(96, 128, 96), valid_group=("pMCI", "sMCI")):
        """
        Args:
            mri_dir (string or Path): MRI 文件所在的文件夹路径。
            pet_dir (string or Path): PET 文件所在的文件夹路径。
            cli_dir (string or Path): Clinical 文件所在的文件夹路径。
            csv_file (string or Path): CSV 文件路径，其中第一列是文件名，第二列是标签。
            transform (callable, optional): 可选的转换操作，应用于样本。
        """
        self.mri_dir = Path(mri_dir)
        self.GM_dir = Path(mri_dir) / 'GM'
        self.WM_dir = Path(mri_dir) / 'WM'
        self.pet_dir = Path(pet_dir)
        self.cli_dir = pd.read_csv(cli_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2,  'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        self.transform = transforms.Compose([
            Resize(resize_shape),
            NoNan(),
            Numpy2Torch(),
            # transforms.Normalize([0.5], [0.5])
        ])

        # 过滤只保留 valid_group 中的有效数据
        self.filtered_indices = self.labels_df[self.labels_df.iloc[:, 1].isin(self.valid_group)].index.tolist()

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # 获取过滤后的索引
        filtered_idx = self.filtered_indices[idx]
        # 获取对应的文件名和标签
        img_name = self.labels_df.iloc[filtered_idx, 0]
        label_str = self.labels_df.iloc[filtered_idx, 1]  # 标签
        # GM 文件路径
        gm_img_path = self.GM_dir / (img_name + '.nii.gz')
        gm_img_numpy = nib.load(str(gm_img_path)).get_fdata()
        gm_img_torch = self.transform(gm_img_numpy)
        # WM 文件路径
        wm_img_path = self.WM_dir / (img_name + '.nii.gz')
        wm_img_numpy = nib.load(str(wm_img_path)).get_fdata()
        wm_img_torch = self.transform(wm_img_numpy)
        # PET 文件路径
        pet_img_path = self.pet_dir / (img_name + '.nii')
        # print('pet_img_path', pet_img_path)
        pet_img_numpy = nib.load(str(pet_img_path)).get_fdata()
        pet_img_torch = self.transform(pet_img_numpy)
        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1
        batch = {
            "gm": gm_img_torch.float(),
            "wm": wm_img_torch.float(),
            "pet": pet_img_torch.float(),
            "label": label
        }
        return batch

# if __name__ == '__main__':
#     mri_dir = r'/data3/wangchangmiao/shenxy/ADNI/ADNI1/MRI'  # 替换为 MRI 文件的路径
#     pet_dir = r'/data3/wangchangmiao/shenxy/ADNI/ADNI1/PET'  # 替换为 PET 文件的路径
#     cli_dir = r'./csv/ADNI_Clinical.csv'
#     csv_file = r'./csv/ADNI1_match.csv'  # 替换为 CSV 文件路径
#     batch_size = 8  # 设置批次大小
#
#     dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("AD", "MCI", "CN"))
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # 测试读取数据
#     print('dataloader', len(dataloader))
#     print('dataset', len(dataloader.dataset))
#     # for i, (mri_imgs, pet_imgs, cli_tab, labels) in enumerate(dataloader):
#     for i, (mri_imgs, pet_imgs, labels) in enumerate(dataloader):
#         print(f"{i} MRI Images batch shape: {mri_imgs.shape}")  # ([8, 1, 96, 128, 96])
#         print(f"{i} PET Images batch shape: {pet_imgs.shape}")  # ([8, 1, 96, 128, 96])
#         # print(f"{i} Clinical Table batch shape: {cli_tab.shape}")  # ([8, 9])
#         print(f"{i} Labels batch shape: {labels}")

if __name__ == '__main__':
    mri_dir = r'/data3/wangchangmiao/shenxy/ADNI/AIBL/MRI'  # 替换为 MRI 文件的路径
    pet_dir = r'/data3/wangchangmiao/shenxy/ADNI/AIBL/PET'  # 替换为 PET 文件的路径
    cli_dir = r'./csv/AIBL_Clinical.csv'
    csv_file = r'./csv/AIBL_label.csv'  # 替换为 CSV 文件路径
    batch_size = 8  # 设置批次大小

    dataset = AIBLDataset(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("AD", "CN"))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 测试读取数据
    print('dataloader', len(dataloader))
    print('dataset', len(dataloader.dataset))
    # for i, (mri_imgs, pet_imgs, cli_tab, labels) in enumerate(dataloader):
    for i, batch in enumerate(dataloader):
        print(f"{i} MRI Images batch shape: {batch['mri'].shape}")  # ([8, 1, 96, 128, 96])
        print(f"{i} PET Images batch shape: {batch['pet'].shape}")  # ([8, 1, 96, 128, 96])
        print(f"{i} Clinical Table batch shape: {batch['clinical'].shape}")  # ([8, 9])
        print(f"{i} Labels batch shape: {batch['label']}")