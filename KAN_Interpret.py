import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from Net.TriLightNet import TriLightNet
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from skimage import transform as skt
import nibabel as nib
from Dataset import MriPetCliDataset
import seaborn as sns

if __name__ == "__main__":
    # resize_shape = (96, 128, 96)
    # self_transform = transforms.Compose([
    #     Resize(resize_shape),
    #     NoNan(),
    #     Numpy2Torch(),
    #     transforms.Normalize([0.5], [0.5])
    # ])
    # mri_img_path = r'D:\dataset\final\freesurfer\ADNI2\MRI\003_S_4142.nii'
    # mri_img_numpy = nib.load(str(mri_img_path)).get_fdata()
    # mri_img_torch = self_transform(mri_img_numpy).float()
    # mri_img_torch = mri_img_torch.unsqueeze(0)
    # pet_img_path = r'D:\dataset\final\freesurfer\ADNI2\PET\003_S_4142.nii'
    # pet_img_numpy = nib.load(str(pet_img_path)).get_fdata()
    # pet_img_torch = self_transform(pet_img_numpy).float()
    # pet_img_torch = pet_img_torch.unsqueeze(0)
    # 初始化模型
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = TriLightNet(num_classes=2)
    pretrained_path = r"/data3/wangchangmiao/shenxy/Code/AD/AweSome/2025_7_14/TriLightNet_best_model_fold1.pth"
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model = model.to(device)
    model.eval()  # 将模型设置为评估模式

    experiment_settings = {
        'shape': (96, 128, 96),
        'task': ('pMCI', 'sMCI')
    }

    # 提取KAN网络
    model_kan = model.Table.fc1

    mri_dir = '/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/MRI'
    pet_dir = '/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/PET'
    cli_dir = './csv/ADNI_Clinical.csv'
    csv_file = './csv/ADNI1_2_pmci_smci_validation.csv'
    dataset = MriPetCliDataset(mri_dir, pet_dir, cli_dir, csv_file, experiment_settings['shape'], experiment_settings['task'])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    train_bar = tqdm(dataloader, desc=f"IG ", unit="batch")
    beta = 100
    batch = next(iter(train_bar))
    # for ii, batch in enumerate(train_bar):
    #     mri_images = batch.get("mri").to(device)
    #     pet_images = batch.get("pet").to(device)
    #     cli_tab = batch.get("clinical").to(device)
    #     label = batch.get("label").to(device)
    #     # image_name = batch.get("image_name")
    #     target_class = label.item()
    cli_tab = batch.get("clinical").to(device)

    model_kan(cli_tab)
    model_kan.plot(beta=beta)
