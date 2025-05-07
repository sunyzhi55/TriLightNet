
import os
import tempfile
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.transforms import (
    EnsureChannelFirstd,
    CenterSpatialCropd,
    Lambdad,
    LoadImaged,
    Resized,
    ScaleIntensityd,
    Compose, LoadImage, EnsureChannelFirst,
    ScaleIntensity, RandFlip, RandAffine,
    RandGaussianNoise, RandGaussianSmooth,
    RandGibbsNoise, EnsureType, ToTensor,Resize,
    NormalizeIntensity
)
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
import nibabel as nib  # 添加在文件开头
from monai.transforms import AsDiscrete  # 可选：用于转成离散值

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
        if pet_dir == '':
            self.pet_dir = ''
        else:
            self.pet_dir = Path(pet_dir)
        self.labels_df = pd.read_csv(csv_file)  # 读取 CSV 文件
        self.groups = {'CN': 0, 'MCI': 1, 'AD': 2, 'pMCI': 1, 'sMCI': 0}
        self.valid_group = valid_group
        # monai 变换
        self.load_transform = Compose([
            LoadImage(image_only=True),  # 第一步：加载图像数据
            EnsureType()
        ])
        self.process_transform = Compose([
            EnsureChannelFirst(),  # 添加通道维度
            NormalizeIntensity(nonzero=True),
            ScaleIntensity(minv=0.0, maxv=1.0),  # 强度归一化
            Resize(resize_shape),  # 调整尺寸
            EnsureType()  # 最终确保输出为tensor
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

        # 1. 加载MRI并处理NaN
        mri_path = str(self.mri_dir / (img_name + '.nii'))
        mri_data = self.load_transform(mri_path)
        mri_data = torch.nan_to_num(mri_data, nan=0.0)  # 处理NaN
        mri_img_torch = self.process_transform(mri_data)  # 应用后续转换

        # 2. 加载PET并处理NaN
        pet_path = str(self.pet_dir / (img_name + '.nii'))
        pet_data = self.load_transform(pet_path)
        pet_data = torch.nan_to_num(pet_data, nan=0.0)
        pet_img_torch = self.process_transform(pet_data)


        label = self.groups.get(label_str, -1)  # 获取标签，默认值为 -1

        batch = {
            # "image": mri_img_torch.float(),
            "image": pet_img_torch.float(),
            "label": label
        }

        return batch

def eval_sample(model, scheduler, device, inferer):
    model.eval()
    noise = torch.randn((1, 1, 96, 128, 96)).to(device)
    scheduler.set_timesteps(num_inference_steps=1000)
    with torch.no_grad():
        image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler)

    # 只取第一个batch并去除通道维度，得到 (32, 40, 32)
    image_np = image[0, 0].cpu().numpy()

    # 将生成的图像保存为NIfTI文件
    affine = np.eye(4)  # 简单的单位矩阵，表示无空间变换（如无特别要求）
    nii_image = nib.Nifti1Image(image_np, affine)

    save_path = "./sample_output.nii.gz"
    nib.save(nii_image, save_path)
    print(f"Saved generated 3D image to {save_path}")

    # 可视化展示切片
    plt.style.use("default")
    plotting_image_0 = np.concatenate([image[0, 0, :, :, 15].cpu(), np.flipud(image[0, 0, :, 20, :].cpu().T)], axis=1)
    plotting_image_1 = np.concatenate([np.flipud(image[0, 0, 15, :, :].cpu().T), np.zeros((32, 32))], axis=1)
    plt.imshow(np.concatenate([plotting_image_0, plotting_image_1], axis=0), vmin=0, vmax=1, cmap="gray")
    plt.tight_layout()
    plt.axis("off")
    plt.savefig("sample_output.png")

def ddpm_train(train_loader, val_loader, model, inferer, device, scheduler, scaler, optimizer, n_epochs,
               val_interval, checkpoints_dir, image_dir):
    epoch_loss_list = []
    val_epoch_loss_list = []
    total_start = time.time()

    # 初始化最小验证损失
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            images = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True, device_type="cuda"):
                noise = torch.randn_like(images).to(device)
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()
                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / len(train_loader)})

        epoch_loss_list.append(epoch_loss / (step + 1))

        # 验证过程
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            with torch.no_grad():
                val_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=70, desc=f"Validation Epoch {epoch}")
                for step, batch in val_bar:
                    images = batch["image"].to(device)
                    noise = torch.randn_like(images).to(device)

                    with autocast(enabled=True, device_type="cuda"):
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()
                        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())

                    val_epoch_loss += val_loss.item()
                    val_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

            avg_val_loss = val_epoch_loss / len(val_loader)
            val_epoch_loss_list.append(avg_val_loss)

            # ✅ 如果当前验证损失是最小的，就保存模型
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save({
                    'model': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                }, Path(checkpoints_dir) / "best_checkpoint.pth")
                print(f"🔥 Saved new best model at epoch {epoch}, val_loss: {best_val_loss:.6f}")

            # 采样
            image = torch.randn((1, 1, 48, 64, 48)).to(device)
            scheduler.set_timesteps(num_inference_steps=1000)
            with autocast(enabled=True, device_type="cuda"):
                image = inferer.sample(input_noise=image, diffusion_model=model, scheduler=scheduler)
                # 可视化展示切片
                plt.style.use("default")
                plotting_image_0 = np.concatenate(
                    [image[0, 0, :, :, 24].cpu(), np.flipud(image[0, 0, :, 32, :].cpu().T)], axis=1)
                plotting_image_1 = np.concatenate([np.flipud(image[0, 0, 24, :, :].cpu().T), np.zeros((48, 48))],
                                                  axis=1)
                plt.imshow(np.concatenate([plotting_image_0, plotting_image_1], axis=0), vmin=0, vmax=1, cmap="gray")
                plt.tight_layout()
                plt.axis("off")
                save_path_2d = Path(image_dir) / f"sample_output_{epoch}.png"
                plt.savefig(save_path_2d)
                # plt.clf()
                # plt.close()
                print(f"Epoch {epoch} Saved generated image to {save_path_2d}")

                # 只取第一个batch并去除通道维度，得到 (96, 128, 96)
                image_np = image[0, 0].cpu().numpy()
                # 将生成的图像保存为NIfTI文件
                affine = np.eye(4)  # 简单的单位矩阵，表示无空间变换（如无特别要求）
                nii_image = nib.Nifti1Image(image_np, affine)
                save_path_3d = Path(image_dir) / f"sample_output_{epoch}.nii"
                nib.save(nii_image, save_path_3d)
                print(f"Saved generated 3D image to {save_path_3d}")

    total_time = time.time() - total_start
    # 将total_time 转换为 h m s
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Train completed, total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    # print(f"train completed, total time: {total_time:.2f}s.")

    # 绘制损失曲线
    plt.clf()
    plt.close()
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, n_epochs + 1), epoch_loss_list, color="C0", linewidth=2.0, label="Train Loss")
    plt.plot(
        np.arange(val_interval, n_epochs + 1, val_interval),
        val_epoch_loss_list,
        color="C1",
        linewidth=2.0,
        label="Validation Loss",
    )
    plt.title("Learning Curves", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(prop={"size": 14})
    plt.grid(True)
    loss_plot_path = "diffusion_training_loss.png"
    plt.savefig(loss_plot_path)
    print(f"📉 Saved loss curve to {loss_plot_path}")



if __name__ == "__main__":
    print_config()
    set_determinism(42)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def extract_channel_1(x):
        return x[:, :, :, 1]
    data_transform = Compose(
        [
            LoadImaged(keys=["image"]),
            Lambdad(keys="image", func=extract_channel_1),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"]),
            CenterSpatialCropd(keys=["image"], roi_size=[160, 200, 155]),
            Resized(keys=["image"], spatial_size=(32, 40, 32)),
        ]
    )

    batch_size = 2
    mri_dir = r"/data4/wangchangmiao/shenxy/ADNI/ADNI1_2/MRI"
    pet_dir = r"/data4/wangchangmiao/shenxy/ADNI/ADNI1_2/PET"
    cli_dir = r"./csv/ADNI_Clinical.csv"
    train_csv_file = r"./csv/ADNI1_2_pmci_smci_train.csv"
    val_csv_file = r"./csv/ADNI1_2_pmci_smci_validation.csv"

    train_ds = MriDataset(mri_dir, pet_dir, cli_dir, train_csv_file, resize_shape=(48, 64, 48), valid_group=("pMCI", "sMCI"))
    # 创建 DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_ds = MriDataset(mri_dir, pet_dir, cli_dir, val_csv_file, resize_shape=(48, 64, 48), valid_group=("pMCI", "sMCI"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)


    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=[128, 256, 512],
        attention_levels=[False, False, True],
        num_head_channels=[4, 16, 512],
        num_res_blocks=4,
    )
    # load the pretrained model
    # path = "./checkpoints/best_checkpoint.pth"
    # model_dict = torch.load(path)["model"]
    # model.load_state_dict(model_dict)
    model.to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195)
    # scheduler.load_state_dict(torch.load(path)['scheduler'])

    inferer = DiffusionInferer(scheduler)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5)
    n_epochs = 3000
    val_interval = 10
    epoch_loss_list = []
    val_epoch_loss_list = []
    scaler = GradScaler()
    checkpoints_dir = "./checkpoints"
    Path(checkpoints_dir).mkdir(exist_ok=True)
    image_dir = "./image_output"
    Path(image_dir).mkdir(exist_ok=True)
    print("=============Start Training=================")
    ddpm_train(train_loader, val_loader, model, inferer, device, scheduler, scaler, optimizer, n_epochs,
               val_interval, checkpoints_dir, image_dir)


    # print("=============Start Testing=================")
    #
    # # 设置设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # 初始化模型结构并加载权重
    # model = DiffusionModelUNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=1,
    #     num_channels=[256, 256, 512],
    #     attention_levels=[False, False, True],
    #     num_head_channels=[0, 0, 512],
    #     num_res_blocks=2,
    # )
    # model.load_state_dict(torch.load('model.pth')['model'])
    # model.to(device)
    #
    # # 初始化scheduler并加载状态
    # scheduler = DDPMScheduler(
    #     num_train_timesteps=1000,
    #     schedule="scaled_linear_beta",
    #     beta_start=0.0005,
    #     beta_end=0.0195
    # )
    # scheduler.load_state_dict(torch.load('model.pth')['scheduler'])
    #
    # # 构建inferer
    # inferer = DiffusionInferer(scheduler)
    #
    # # 调用采样
    # eval_sample(model, scheduler, device, inferer)

