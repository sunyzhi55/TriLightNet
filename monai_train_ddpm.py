import os
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.utils import first, set_determinism
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm
from monai.inferers import LatentDiffusionInferer
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from monai.networks.schedulers import DDPMScheduler
import time
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset
import pandas as pd
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
    RandGibbsNoise, EnsureType, ToTensor, Resize,
    NormalizeIntensity
)

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
            "image": mri_img_torch.float(),
            # "image": pet_img_torch.float(),
            "label": label
        }

        return batch

def kl_loss(z_mu, z_sigma):
    klloss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(klloss) / klloss.shape[0]


def train_autoencoder(train_loader, val_loader, autoencoder, discriminator,
                      optimizer_g, optimizer_d,
                      l1_loss, adv_loss, loss_perceptual,
                      adv_weight, perceptual_weight, kl_weight,
                      checkpoints_dir, image_dir,
                      max_epochs, autoencoder_warm_up_n_epochs, val_interval, n_example_images):
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    val_recon_epoch_loss_list = []
    intermediary_images = []
    best_val_recon_epoch_loss = 100.
    for epoch in range(max_epochs):
        autoencoder.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)  # choose only one of Brats channels

            # Generator part
            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)
            klloss = kl_loss(z_mu, z_sigma)

            recons_loss = l1_loss(reconstruction.float(), images.float())
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * klloss + perceptual_weight * p_loss

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                # Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            epoch_loss += recons_loss.item()

            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

            progress_bar.set_postfix(
                {
                    "recons_loss": epoch_loss / (step + 1),
                    "gen_loss": gen_epoch_loss / (step + 1),
                    "disc_loss": disc_epoch_loss / (step + 1),
                }
            )
        # validation
        if epoch % val_interval == 0:
            autoencoder.eval()
            val_recon_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)  # choose only one of Brats channels
                with torch.no_grad():
                    reconstruction, z_mu, z_sigma = autoencoder(images)
                    recons_loss = l1_loss(
                        reconstruction.float(), images.float()
                    ) + perceptual_weight * loss_perceptual(reconstruction.float(), images.float())

                val_recon_epoch_loss += recons_loss.item()

            val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
            # save best model
            if val_recon_epoch_loss < best_val_recon_epoch_loss:
                best_val_recon_epoch_loss = val_recon_epoch_loss
                trained_g_path = Path(checkpoints_dir) / "best_autoEncoder.pth"
                trained_d_path = Path(checkpoints_dir) / "best_discriminator.pth"
                torch.save(autoencoder.module.state_dict(), trained_g_path)
                torch.save(discriminator.module.state_dict(), trained_d_path)
                print(f"Epoch: {epoch}, Got best val recon loss: {best_val_recon_epoch_loss}")
                print("Save trained autoencoder to", trained_g_path)
                print("Save trained discriminator to", trained_d_path)
        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
        epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

    del discriminator
    del loss_perceptual
    torch.cuda.empty_cache()
    plt.style.use("ggplot")
    plt.title("Learning Curves", fontsize=20)
    plt.plot(epoch_recon_loss_list)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(f"recon_loss.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.title("Adversarial Training Curves", fontsize=20)
    plt.plot(epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
    plt.plot(epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(f"adversarial_loss.png", dpi=300, bbox_inches="tight")

def train_ddpm(train_loader, val_loader,
               unet, autoencoder, inferer, optimizer_diff, device, ddpm_max_epochs, scaler, ddpm_val_interval, ):
    autoencoder.eval()
    first_batch = first(train_loader)
    z = autoencoder.encode_stage_2_inputs(first_batch["image"].to(device))
    epoch_loss_list = []
    val_epoch_loss_list = []
    total_start = time.time()
    # 初始化最小验证损失
    best_val_loss = float('inf')

    for epoch in range(ddpm_max_epochs):
        unet.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            optimizer_diff.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=True):
                # Generate random noise
                noise = torch.randn_like(z).to(device)
                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()
                # Get model prediction
                noise_pred = inferer(
                    inputs=images, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps
                )
                loss = F.mse_loss(noise_pred.float(), noise.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_loss_list.append(epoch_loss / (step + 1))
        # 验证过程
        if (epoch + 1) % ddpm_val_interval == 0:
            unet.eval()
            autoencoder.eval()
            val_epoch_loss = 0
            val_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=70, desc=f"Validation Epoch {epoch}")

            with torch.no_grad():
                for step, batch in val_bar:
                    images = batch["image"].to(device)
                    noise = torch.randn_like(z).to(device)
                    with autocast(enabled=True, device_type="cuda"):
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()
                        noise_pred = inferer(inputs=images, autoencoder_model=autoencoder,diffusion_model=unet, noise=noise, timesteps=timesteps)

                        val_loss = F.mse_loss(noise_pred.float(), noise.float())

                    val_epoch_loss += val_loss.item()
                    val_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

            avg_val_loss = val_epoch_loss / len(val_loader)
            val_epoch_loss_list.append(avg_val_loss)

            # ✅ 如果当前验证损失是最小的，就保存模型
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save({
                    'model': unet.module.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                }, Path(checkpoints_dir) / "best_checkpoint.pth")
                print(f"🔥 Saved new best model at epoch {epoch}, val_loss: {best_val_loss:.6f}")

            # 采样
            autoencoder.eval()
            unet.eval()
            image = torch.randn_like(z).to(device)
            print("image", image.shape)
            scheduler.set_timesteps(num_inference_steps=1000)
            with autocast(enabled=True, device_type="cuda"):
                image = inferer.sample(input_noise=image, autoencoder_model=autoencoder, diffusion_model=unet.module, scheduler=scheduler)
                # 可视化展示切片
                plt.style.use("default")
                plotting_image_0 = np.concatenate(
                    [image[0, 0, :, :, 48].cpu(), np.flipud(image[0, 0, :, 64, :].cpu().T)], axis=1)
                plotting_image_1 = np.concatenate([np.flipud(image[0, 0, 48, :, :].cpu().T), np.zeros((96, 96))],
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
                image_np = image_np.astype(np.float32)  # 强制转换为 float32
                # 将生成的图像保存为NIfTI文件
                affine = np.eye(4)  # 简单的单位矩阵，表示无空间变换（如无特别要求）
                nii_image = nib.Nifti1Image(image_np, affine)
                save_path_3d = Path(image_dir) / f"sample_output_{epoch}.nii"
                nib.save(nii_image, save_path_3d)
                print(f"Saved generated 3D image to {save_path_3d}")
    # ===========
    # 绘制损失曲线
    plt.clf()
    plt.close()
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, ddpm_max_epochs + 1), epoch_loss_list, color="C0", linewidth=2.0, label="Train Loss")
    plt.plot(
        np.arange(ddpm_val_interval, ddpm_max_epochs + 1, ddpm_val_interval),
        val_epoch_loss_list,
        color="C1",
        linewidth=2.0,
        label="Validation Loss",
    )
    plt.title("DDPM loss Curves", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(prop={"size": 14})
    plt.grid(True)
    loss_plot_path = "ddpm_loss.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    print(f"📉 Saved loss curve to {loss_plot_path}")




if __name__ == "__main__":
    print_config()
    set_determinism(55656)

    # 获取所有可用的GPU设备
    # device_ids = list(range(torch.cuda.device_count()))
    device_ids = [1, 2, 3, 4]
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using devices: {device_ids}")
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
    batch_size = 4
    mri_dir = r"/data4/wangchangmiao/shenxy/ADNI/ADNI1_2/MRI"
    pet_dir = r"/data4/wangchangmiao/shenxy/ADNI/ADNI1_2/PET"
    cli_dir = r"./csv/ADNI_Clinical.csv"
    train_csv_file = r"./csv/ADNI1_2_pmci_smci_train.csv"
    val_csv_file = r"./csv/ADNI1_2_pmci_smci_validation.csv"
    train_ds = MriDataset(mri_dir, pet_dir, cli_dir, train_csv_file, resize_shape=(96, 128, 96),
                          valid_group=("pMCI", "sMCI"))
    # 创建 DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_ds = MriDataset(mri_dir, pet_dir, cli_dir, val_csv_file, resize_shape=(96, 128, 96),
                        valid_group=("pMCI", "sMCI"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    # Define Autoencoder KL network
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 64),
        latent_channels=3,
        num_res_blocks=1,
        norm_num_groups=16,
        attention_levels=(False, False, True),
    )
    discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, channels=32, in_channels=1, out_channels=1)

    if len(device_ids) > 1:
        autoencoder = nn.DataParallel(autoencoder, device_ids=device_ids)
        discriminator = nn.DataParallel(discriminator, device_ids=device_ids)

    # load the pretrained model
    # autoencoder_path = "./checkpoints/best_autoEncoder.pth"
    # discriminator_path = "./checkpoints/best_discriminator.pth"
    # autoencoder_dict = torch.load(autoencoder_path, map_location=device)
    # autoencoder.load_state_dict(autoencoder_dict)
    # # 去掉 DataParallel 包装（如果之前是多卡训练保存的）
    # if isinstance(autoencoder, nn.DataParallel):
    #     autoencoder = autoencoder.module
    autoencoder.to(device)

    # discriminator_dict = torch.load(discriminator_path, map_location=device)
    # discriminator.load_state_dict(discriminator_dict)
    # # 去掉 DataParallel 包装（如果之前是多卡训练保存的）
    # if isinstance(discriminator_dict, nn.DataParallel):
    #     discriminator_dict = discriminator_dict.module
    discriminator.to(device)

    # Define the loss
    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)
    adv_weight = 0.01
    perceptual_weight = 0.001
    kl_weight = 1e-6

    # define the optimizer
    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

    # train autoencoder
    autoencoder_max_epochs = 1000
    autoencoder_warm_up_n_epochs = 5
    autoencoder_val_interval = 5
    n_example_images = 4
    checkpoints_dir = "./checkpoints"
    Path(checkpoints_dir).mkdir(exist_ok=True)
    image_dir = "./image_output"
    Path(image_dir).mkdir(exist_ok=True)
    train_autoencoder(train_loader, val_loader, autoencoder, discriminator,
                          optimizer_g, optimizer_d,
                          l1_loss, adv_loss, loss_perceptual,
                          adv_weight, perceptual_weight, kl_weight,
                          checkpoints_dir, image_dir,
                          autoencoder_max_epochs, autoencoder_warm_up_n_epochs, autoencoder_val_interval, n_example_images)
    # visual
    # Plot axial, coronal and sagittal slices of a training sample
    autoencoder = autoencoder.module.to(device)
    discriminator = discriminator.module.to(device)
    channel = 0
    for step, batch in enumerate(val_loader):
        images = batch["image"].to(device)  # choose only one of Brats channels
        with torch.no_grad():
            reconstruction, z_mu, z_sigma = autoencoder(images)
            idx = 0
            img = reconstruction[idx, channel].detach().cpu().numpy()

            # 只取第一个batch并去除通道维度，得到 (96, 128, 96)

            # 将生成的图像保存为NIfTI文件
            affine = np.eye(4)  # 简单的单位矩阵，表示无空间变换（如无特别要求）
            nii_image = nib.Nifti1Image(img, affine)
            save_path_3d = Path(image_dir) / f"sample_output_{step}.nii"
            nib.save(nii_image, save_path_3d)
            print(f"Saved generated 3D image to {save_path_3d}")

            fig, axs = plt.subplots(nrows=1, ncols=3)
            for ax in axs:
                ax.axis("off")
            ax = axs[0]
            ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
            ax = axs[1]
            ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
            ax = axs[2]
            ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
            plt.savefig(Path(image_dir) / f"sample_output_{step}.png")
            plt.close(fig)

    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=3,
        num_res_blocks=1,
        channels=(32, 64, 64),
        attention_levels=(False, True, True),
        num_head_channels=(0, 64, 64),
    )
    if len(device_ids) > 1:
        unet = nn.DataParallel(unet, device_ids=device_ids)
    unet.to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015,
                              beta_end=0.0195)
    check_data = first(train_loader)
    with torch.no_grad():
        with autocast("cuda", enabled=True):
            z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))

    print(f"Scaling factor set to {1 / torch.std(z)}")
    scale_factor = 1 / torch.std(z)
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)
    # train the ddpm
    ddpm_max_epochs = 1000
    scaler = GradScaler("cuda")
    ddpm_val_interval = 5

    train_ddpm(train_loader, val_loader,
               unet, autoencoder, inferer, optimizer_diff, device, ddpm_max_epochs, scaler, ddpm_val_interval, )

