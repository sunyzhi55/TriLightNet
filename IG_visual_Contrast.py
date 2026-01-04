import torch
from captum.attr import IntegratedGradients
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from AweSomeNet.AweNet import AweSomeNet
import numpy as np
import torchvision.transforms as transforms
from skimage import transform as skt
import nibabel as nib
from MultimodalADNet.multimodalNet import MultimodalADNet
from Dataset import MriPetCliDataset
import seaborn as sns
import time
from Dataset import *
from Net import *
from MDL_Net.MDL_Net import generate_model
from HyperFusionNet.HyperFusion_AD_model import HyperFusion_AD
from vapformer.model_components import thenet
from DiamondNet.DiaMond import DiaMond, Head
from DiamondNet.regbn import RegBN
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

class CompleteModel(nn.Module):
    def __init__(self, diamond, regbn_kwargs, num_classes=2, device='cuda:2', checkpoint_path=None):
        super(CompleteModel, self).__init__()
        checkpoint = torch.load(checkpoint_path, map_location=device) if checkpoint_path else None
        # 初始化各子模型
        body_model= diamond.body_all(
            modality="multi",
            block_size=32,
            image_size=128,
            patch_size=8,
            num_classes=num_classes,
            channels=1,
            dim=512,
            depth=4,
            heads=8,
            dropout=0.0,
            mlp_dim=309
        )

        [m.load_state_dict(checkpoint['model_state_dict'][i]) for i, m in enumerate(body_model)]
        body_model = [m.to(device) for m in body_model]

        self.model_pet, self.model_mri, self.model_mp = body_model

        self.head = diamond.head(
            block_size=32,
            image_size=128,
            num_classes=num_classes,
            channels=1,
        )
        self.head.load_state_dict(checkpoint['head_state_dict'])
        self.head.to(device)

        # self.regbn = RegBN(**regbn_kwargs)
        self.regbn = RegBN(**regbn_kwargs).to(device)

        self.kwargs_regbn_train = {"is_training": False}

    def forward(self, pet_input, mri_input):
        # PET分支
        output_pet = self.model_pet(pet_input)

        # MRI分支
        output_mri = self.model_mri(mri_input)

        # RegBN处理
        output_pet, output_mri = self.regbn(output_pet, output_mri, **self.kwargs_regbn_train)

        # 多模态融合分支
        output_mp = self.model_mp(pet_input, mri_input)

        # 三路输出融合
        output = (output_pet + output_mri + output_mp) / 3

        # Head处理
        output = self.head(output)

        return output
def visualize_ig_heatmap(original, attributions, slice_index=(48, 64, 48), channel=0, output_path=None):
    """
    可视化 MRI 或 PET 归因热力图，并绘制三个视角的切面图：轴向 (Axial)、冠状 (Coronal) 和矢状 (Sagittal)。

    参数:
        original: 原始输入图像 (numpy 数组)，形状为 [batch_size, channels, depth, height, width]
        attributions: IG 归因值 (numpy 数组)，形状与 original 相同
        slice_index: 要可视化的切片索引
        channel: 要可视化的通道索引
        output_path: 保存图像的路径（如果为 None，则不保存）
    """
    # 提取指定切片和通道
    original_axial = original[0, channel, :, :, slice_index[0]]  # 轴向切面
    original_coronal = original[0, channel, :, slice_index[1], :]  # 冠状切面
    original_sagittal = original[0, channel, slice_index[2], :, :]  # 矢状切面

    attribution_axial = attributions[0, channel, :, :, slice_index[0]]  # 轴向归因
    attribution_coronal = attributions[0, channel, :, slice_index[1], :]  # 冠状归因
    attribution_sagittal = attributions[0, channel, slice_index[2], :, :]  # 矢状归因

    # 归一化归因值
    def normalize(attribution_slice):
        attribution_slice = np.abs(attribution_slice)
        if np.max(attribution_slice) - np.min(attribution_slice) > 0:
            return (attribution_slice - np.min(attribution_slice)) / (np.max(attribution_slice) - np.min(attribution_slice))
        else:
            return np.zeros_like(attribution_slice)

    attribution_axial = normalize(attribution_axial)
    attribution_coronal = normalize(attribution_coronal)
    attribution_sagittal = normalize(attribution_sagittal)

    # 创建画布
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    # 绘制轴向切面
    original_coronal_rotated = np.rot90(original_axial, k=1)  # 逆时针旋转 90 度
    attribution_coronal_rotated = np.rot90(attribution_axial, k=1)  # 同样旋转归因值
    axes[0, 0].imshow(original_coronal_rotated, cmap="gray", aspect='equal')
    axes[0, 0].set_title("Original Axial", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(original_coronal_rotated, cmap="gray", aspect='equal')
    axes[0, 1].imshow(attribution_coronal_rotated, cmap="jet", alpha=0.5, aspect='equal')
    axes[0, 1].set_title("Overlayed Axial", fontsize=12)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(attribution_coronal_rotated, cmap="jet", aspect='equal')
    axes[0, 2].set_title("Attribution Axial", fontsize=12)
    axes[0, 2].axis("off")


    # 绘制冠状切面（逆时针旋转 90 度）
    original_coronal_rotated = np.rot90(original_coronal, k=1)  # 逆时针旋转 90 度
    attribution_coronal_rotated = np.rot90(attribution_coronal, k=1)  # 同样旋转归因值
    axes[1, 0].imshow(original_coronal_rotated, cmap="gray", aspect='equal')
    axes[1, 0].set_title("Original Coronal", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(original_coronal_rotated, cmap="gray", aspect='equal')
    axes[1, 1].imshow(attribution_coronal_rotated, cmap="jet", alpha=0.5, aspect='equal')
    axes[1, 1].set_title("Overlayed Coronal", fontsize=12)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(attribution_coronal_rotated, cmap="jet", aspect='equal')
    axes[1, 2].set_title("Attribution Coronal", fontsize=12)
    axes[1, 2].axis("off")

    # 绘制矢状切面（逆时针旋转 90 度）
    original_sagittal_rotated = np.rot90(original_sagittal, k=1)  # 逆时针旋转 90 度
    attribution_sagittal_rotated = np.rot90(attribution_sagittal, k=1)  # 同样旋转归因值

    axes[2, 0].imshow(original_sagittal_rotated, cmap="gray", aspect='equal')
    axes[2, 0].set_title("Original Sagittal", fontsize=12)
    axes[2, 0].axis("off")

    axes[2, 1].imshow(original_sagittal_rotated, cmap="gray", aspect='equal')
    axes[2, 1].imshow(attribution_sagittal_rotated, cmap="jet", alpha=0.5, aspect='equal')
    axes[2, 1].set_title("Overlayed Sagittal", fontsize=12)
    axes[2, 1].axis("off")

    axes[2, 2].imshow(attribution_sagittal_rotated, cmap="jet", aspect='equal')
    axes[2, 2].set_title("Attribution Sagittal", fontsize=12)
    axes[2, 2].axis("off")

    plt.savefig(output_path, bbox_inches="tight", dpi=300)

def visualize_ig_heatmap_seaborn(original, attributions, slice_index=(48, 64, 48), channel=0, output_path=None):
    """
    使用 Seaborn 绘制 MRI 或 PET 归因热力图，并展示三个视角：轴向 (Axial)、冠状 (Coronal) 和矢状 (Sagittal)。

    参数:
        original: 原始输入图像 (numpy 数组)，形状为 [batch_size, channels, depth, height, width]
        attributions: IG 归因值 (numpy 数组)，形状与 original 相同
        slice_index: 要可视化的切片索引 (z, y, x)
        channel: 要可视化的通道索引
        output_path: 保存图像的路径（如果为 None，则不保存）
    """
    # 提取指定切片和通道
    views = {
        "Axial": original[0, channel, :, :, slice_index[0]],  # 轴向切面
        "Coronal": original[0, channel, :, slice_index[1], :],  # 冠状切面
        "Sagittal": original[0, channel, slice_index[2], :, :],  # 矢状切面
    }

    attribution_views = {
        "Axial": attributions[0, channel, :, :, slice_index[0]],  # 轴向归因
        "Coronal": attributions[0, channel, :, slice_index[1], :],  # 冠状归因
        "Sagittal": attributions[0, channel, slice_index[2], :, :],  # 矢状归因
    }

    # 归一化归因值
    def normalize(attribution_slice):
        attribution_slice = np.abs(attribution_slice)
        if np.max(attribution_slice) - np.min(attribution_slice) > 0:
            return (attribution_slice - np.min(attribution_slice)) / (np.max(attribution_slice) - np.min(attribution_slice))
        else:
            return np.zeros_like(attribution_slice)

    # 创建画布
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    # 遍历每个视角
    for i, view_name in enumerate(["Axial", "Coronal", "Sagittal"]):
        original_slice = views[view_name]
        attribution_slice = normalize(attribution_views[view_name])
        original_slice_rotated = np.rot90(original_slice, k=1)  # 逆时针旋转 90 度
        attribution_slice_rotated = np.rot90(attribution_slice, k=1)  # 同样旋转归因值
        # 创建掩码：屏蔽零值区域
        mask = original_slice_rotated == 0  # 掩码为 True 的部分将被隐藏
        # 绘制原始图像
        sns.heatmap(original_slice_rotated, cmap="gray", cbar=False, ax=axes[i, 0], mask=mask)
        axes[i, 0].set_title(f"Original {view_name}", fontsize=12)
        axes[i, 0].axis("off")
        axes[i, 0].set_aspect('equal')  # 手动设置坐标轴比例为 1:1

        # 叠加热力图
        sns.heatmap(original_slice_rotated, cmap="gray", cbar=False, ax=axes[i, 1], mask=mask)
        sns.heatmap(attribution_slice_rotated, cmap="jet", alpha=0.7, cbar=False, ax=axes[i, 1], mask=mask)
        axes[i, 1].set_title(f"Overlayed {view_name}", fontsize=12)
        axes[i, 1].axis("off")
        axes[i, 1].set_aspect('equal')  # 手动设置坐标轴比例为 1:1

        # 单独绘制热力图
        sns.heatmap(attribution_slice_rotated, cmap="jet", cbar=True, ax=axes[i, 2], mask=mask)
        axes[i, 2].set_title(f"Attribution {view_name}", fontsize=12)
        axes[i, 2].axis("off")
        axes[i, 2].set_aspect('equal')  # 手动设置坐标轴比例为 1:1

    # 保存图像
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

def visualize_ig_heatmap_seaborn_6(original, attributions, slice_index=(48, 64, 48), channel=0, output_path=None):
    """
    使用 Seaborn 绘制 MRI 或 PET 归因热力图，并展示三个视角：轴向 (Axial)、冠状 (Coronal) 和矢状 (Sagittal)。

    参数:
        original: 原始输入图像 (numpy 数组)，形状为 [batch_size, channels, depth, height, width]
        attributions: IG 归因值 (numpy 数组)，形状与 original 相同
        slice_index: 要可视化的切片索引 (z, y, x)
        channel: 要可视化的通道索引
        output_path: 保存图像的路径（如果为 None，则不保存）
    """
    # 提取指定切片和通道
    views = {
        "Axial": original[0, channel, :, :, slice_index[0]],  # 轴向切面
        "Coronal": original[0, channel, :, slice_index[1], :],  # 冠状切面
        "Sagittal": original[0, channel, slice_index[2], :, :],  # 矢状切面
    }

    attribution_views = {
        "Axial": attributions[0, channel, :, :, slice_index[0]],  # 轴向归因
        "Coronal": attributions[0, channel, :, slice_index[1], :],  # 冠状归因
        "Sagittal": attributions[0, channel, slice_index[2], :, :],  # 矢状归因
    }

    # 归一化归因值
    def normalize(attribution_slice):
        attribution_slice = np.abs(attribution_slice)
        if np.max(attribution_slice) - np.min(attribution_slice) > 0:
            return (attribution_slice - np.min(attribution_slice)) / (np.max(attribution_slice) - np.min(attribution_slice))
        else:
            return np.zeros_like(attribution_slice)

    # 创建画布
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 只需要 2 行 3 列
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    # 遍历每个视角
    for i, view_name in enumerate(["Axial", "Coronal", "Sagittal"]):
        original_slice = views[view_name]
        attribution_slice = normalize(attribution_views[view_name])
        original_slice_rotated = np.rot90(original_slice, k=1)  # 逆时针旋转 90 度
        attribution_slice_rotated = np.rot90(attribution_slice, k=1)  # 同样旋转归因值
        # 创建掩码：屏蔽零值区域
        # mask = original_slice_rotated == 0  # 掩码为 True 的部分将被隐藏

        # 第一行：绘制原始图像
        sns.heatmap(original_slice_rotated, cmap="gray", cbar=False, ax=axes[0, i])
        axes[0, i].set_title(f"Original {view_name}", fontsize=12)
        axes[0, i].axis("off")
        axes[0, i].set_aspect('equal')  # 手动设置坐标轴比例为 1:1


        # 第二行：叠加热力图
        sns.heatmap(original_slice_rotated, cmap="gray", cbar=False, ax=axes[1, i])
        sns.heatmap(attribution_slice_rotated, cmap="jet", alpha=0.7, cbar=False, ax=axes[1, i])
        axes[1, i].set_title(f"Overlayed {view_name}", fontsize=12)
        axes[1, i].axis("off")
        axes[1, i].set_aspect('equal')  # 手动设置坐标轴比例为 1:1

    # 自动调整布局，避免重叠
    plt.tight_layout()
    # 保存图像
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

def visualize_ig_heatmap_seaborn_save_all_son(original, attributions, slice_index=(48, 64, 48), channel=0, output_path=None):
    """
    使用 Seaborn 绘制 MRI 或 PET 归因热力图，并展示三个视角：轴向 (Axial)、冠状 (Coronal) 和矢状 (Sagittal)。
    每个子图会分别保存到指定路径中。
    """
    # 提取指定切片和通道
    views = {
        "Axial": original[0, channel, :, :, slice_index[0]],  # 轴向切面
        "Coronal": original[0, channel, :, slice_index[1], :],  # 冠状切面
        "Sagittal": original[0, channel, slice_index[2], :, :],  # 矢状切面
    }

    attribution_views = {
        "Axial": attributions[0, channel, :, :, slice_index[0]],  # 轴向归因
        "Coronal": attributions[0, channel, :, slice_index[1], :],  # 冠状归因
        "Sagittal": attributions[0, channel, slice_index[2], :, :],  # 矢状归因
    }

    # 归一化归因值
    def normalize(attribution_slice):
        attribution_slice = np.abs(attribution_slice)
        if np.max(attribution_slice) - np.min(attribution_slice) > 0:
            return (attribution_slice - np.min(attribution_slice)) / (np.max(attribution_slice) - np.min(attribution_slice))
        else:
            return np.zeros_like(attribution_slice)


    output_dir = Path(output_path).parent  # 提取父目录 "./IG_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 提取文件名部分并去掉扩展名
    base_name = Path(output_path).stem  # 提取文件名（不带扩展名）

    # 遍历每个视角并绘制子图
    for i, view_name in enumerate(["Axial", "Coronal", "Sagittal"]):
        original_slice = views[view_name]
        attribution_slice = normalize(attribution_views[view_name])
        original_slice_rotated = np.rot90(original_slice, k=1)  # 逆时针旋转 90 度
        attribution_slice_rotated = np.rot90(attribution_slice, k=1)  # 同样旋转归因值
        # mask = original_slice_rotated == 0  # 掩码为 True 的部分将被隐藏

        # 第一个子图：原始图像
        fig, ax = plt.subplots(figsize=(5, 5))  # 单独创建画布
        sns.heatmap(original_slice_rotated, cmap="gray", cbar=False, ax=ax)
        ax.axis("off")
        ax.set_aspect('equal')  # 手动设置坐标轴比例为 1:1
        # ax.set_title(f"Original {view_name}", fontsize=12)

        # 保存当前子图
        if output_path:
            original_output_path = output_dir / f"{base_name}_Original_{view_name}.png"
            plt.savefig(original_output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        # 第二个子图：叠加归因热力图
        fig, ax = plt.subplots(figsize=(5, 5))  # 单独创建画布
        sns.heatmap(original_slice_rotated, cmap="gray", cbar=False, ax=ax)
        sns.heatmap(
            attribution_slice_rotated,
            cmap="jet",
            alpha=0.7,
            cbar=False,  # 显示热力条
            ax=ax,
            # cbar_kws={"shrink": 0.5}  # 调整热力条大小
        )
        ax.axis("off")
        ax.set_aspect('equal')  # 手动设置坐标轴比例为 1:1
        # ax.set_title(f"Overlayed {view_name}", fontsize=12)


        overlayed_output_path = output_dir / f"{base_name}_Overlayed_{view_name}.png"
        plt.savefig(overlayed_output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def visualize_cli_attributions_heatmap(cli_input, cli_attr, output_path):
    """
    cli_input: shape (feature_dim,)
    cli_attr: shape (feature_dim,)
    output_path: 保存路径
    """
    feature_dim = cli_input.shape[0]
    data = {
        # "Feature Value": cli_input,
        "Attribution": cli_attr
    }
    output_dir = Path(output_path).parent  # 提取父目录 "./IG_output"
    # 提取文件名部分并去掉扩展名
    base_name = Path(output_path).stem  # 提取文件名（不带扩展名）

    # 合并成一个DataFrame

    df = pd.DataFrame(
        data,
        index=["Gender", "Age", "Education", "FDG_bl", "TAU_bl", "PTAU_bl", "APOE4_0", "APOE4_1", "APOE4_2"]
    )
    # 绘制热力图
    plt.figure(figsize=(6, max(3, feature_dim * 0.4)))
    sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Clinical Features Attribution")
    plt.tight_layout()

    cli_output_path = output_dir / f"{base_name}_cli.png"
    plt.savefig(cli_output_path, bbox_inches="tight", dpi=300)

    plt.close()




# def visualize_ig_heatmap(original, attributions, slice_index=0, channel=0, output_path=None):
#     """
#     可视化 MRI 或 PET 归因热力图。
#
#     参数:
#         original: 原始输入图像 (numpy 数组)，形状为 [batch_size, channels, depth, height, width]
#         attributions: IG 归因值 (numpy 数组)，形状与 original 相同
#         slice_index: 要可视化的切片索引
#         channel: 要可视化的通道索引
#     """
#     # 提取指定切片和通道
#     original_slice = original[0, channel, :, slice_index].squeeze()
#
#     attribution_slice = attributions[0, channel, :, slice_index].squeeze()
#
#
#     # 归一化归因值
#     attribution_slice = np.abs(attribution_slice)
#     attribution_slice = (attribution_slice - np.min(attribution_slice)) / (
#                 np.max(attribution_slice) - np.min(attribution_slice))
#
#     # 绘制原始图像
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(original_slice, cmap="gray")
#     plt.axis("off")
#
#     # 绘制热力图
#     plt.subplot(1, 2, 2)
#     plt.title("Attribution Heatmap")
#     plt.imshow(original_slice, cmap="gray")
#     plt.imshow(attribution_slice, cmap="jet", alpha=0.5)  # 使用半透明叠加热力图
#     plt.axis("off")
#     # plt.show()
#     plt.savefig(output_path)

def AweSomeNet_IG(mri_dir, pet_dir, cli_dir, csv_file, device, pretrained_path, experiment_settings, output_dir):
    def forward_func_mri(mri_input, pet_input, cli_input):
        return model(mri_input, pet_input, cli_input)

    def forward_func_pet(pet_input, mri_input, cli_input):
        return model(mri_input, pet_input, cli_input)

    def forward_func_cli(cli_input, mri_input, pet_input):
        return model(mri_input, pet_input, cli_input)
    model = AweSomeNet(num_classes=2)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model = model.to(device)
    model.eval()  # 将模型设置为评估模式
    dataset = MriPetCliDataset(mri_dir, pet_dir, cli_dir, csv_file, experiment_settings['shape'],
                               experiment_settings['task'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    train_bar = tqdm(dataloader, desc=f"AweSomeNet IG ", unit="batch")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ii, batch in enumerate(train_bar):
        mri_images = batch.get("mri").to(device)
        pet_images = batch.get("pet").to(device)
        cli_tab = batch.get("clinical").to(device)
        label = batch.get("label").to(device)
        image_name = batch.get("image_name")
        # 定义基线
        mri_baseline = torch.zeros_like(mri_images).to(device)  # MRI 基线
        pet_baseline = torch.zeros_like(pet_images).to(device)  # PET 基线
        cli_baseline = torch.zeros_like(cli_tab).to(device)  # 临床数据基线
        # 目标类别 (假设我们关注第 0 类)
        target_class = label.item()
        # 计算 MRI 归因
        # 初始化 Integrated Gradients
        ig_mri = IntegratedGradients(forward_func_mri)
        mri_attributions, mri_delta = ig_mri.attribute(
            inputs=mri_images,
            baselines=mri_baseline,
            additional_forward_args=(pet_images, cli_tab),
            target=target_class,
            return_convergence_delta=True
        )
        # 计算 PET 归因
        ig_pet = IntegratedGradients(forward_func_pet)
        pet_attributions, pet_delta = ig_pet.attribute(
            inputs=pet_images,
            baselines=pet_baseline,
            additional_forward_args=(mri_images, cli_tab),
            target=target_class,
            return_convergence_delta=True
        )
        # 计算 Clinical 数据归因
        ig_cli = IntegratedGradients(forward_func_cli)
        cli_attributions, cli_delta = ig_cli.attribute(
            inputs=cli_tab,
            baselines=cli_baseline,
            additional_forward_args=(mri_images, pet_images),
            target=target_class,
            return_convergence_delta=True
        )
        # 打印结果
        print("MRI Attributions:", mri_attributions.shape)
        print("PET Attributions:", pet_attributions.shape)
        print("Clinical Attributions:", cli_attributions)
        # 示例调用
        mri_input_np = mri_images.detach().cpu().numpy()
        mri_attributions_np = mri_attributions.detach().cpu().numpy()
        mri_output_path = Path(output_dir) / f"mri_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=mri_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=mri_output_path)
        pet_input_np = pet_images.detach().cpu().numpy()
        pet_attributions_np = pet_attributions.detach().cpu().numpy()
        pet_output_path = Path(output_dir) / f"pet_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=pet_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=pet_output_path)
        cli_tab_np = cli_tab[0].detach().cpu().numpy()
        cli_attributions_np = cli_attributions[0].detach().cpu().numpy()
        cli_output_path = Path(output_dir) / f"cli_{image_name}_label_{target_class}.jpg"
        visualize_cli_attributions_heatmap(cli_tab_np, cli_attributions_np, output_path=cli_output_path)
def Resnet_IG(mri_dir, pet_dir, cli_dir, csv_file, device, pretrained_path, experiment_settings, output_dir):
    def forward_func_mri(mri_input, pet_input):
        return model(mri_input, pet_input)

    def forward_func_pet(pet_input, mri_input):
        return model(mri_input, pet_input)
    model = ResnetMriPet(num_classes=2)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model = model.to(device)
    model.eval()  # 将模型设置为评估模式
    dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file, experiment_settings['shape'],
                               experiment_settings['task'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    train_bar = tqdm(dataloader, desc=f"Resnet IG ", unit="batch")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ii, batch in enumerate(train_bar):
        mri_images = batch.get("mri").to(device)
        pet_images = batch.get("pet").to(device)
        # cli_tab = batch.get("clinical").to(device)
        # mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
        label = batch.get("label").to(device)
        image_name = batch.get("image_name")
        # 定义基线
        mri_baseline = torch.zeros_like(mri_images).to(device)  # MRI 基线
        pet_baseline = torch.zeros_like(pet_images).to(device)  # PET 基线
        # 目标类别 (假设我们关注第 0 类)
        target_class = label.item()
        # 计算 MRI 归因
        # 初始化 Integrated Gradients
        ig_mri = IntegratedGradients(forward_func_mri)
        mri_attributions, mri_delta = ig_mri.attribute(
            inputs=mri_images,
            baselines=mri_baseline,
            additional_forward_args=(pet_images),
            target=target_class,
            return_convergence_delta=True
        )

        # 计算 PET 归因
        ig_pet = IntegratedGradients(forward_func_pet)
        pet_attributions, pet_delta = ig_pet.attribute(
            inputs=pet_images,
            baselines=pet_baseline,
            additional_forward_args=(mri_images),
            target=target_class,
            return_convergence_delta=True
        )

        # 打印结果
        print("MRI Attributions:", mri_attributions.shape)
        print("PET Attributions:", pet_attributions.shape)
        # 示例调用
        mri_input_np = mri_images.detach().cpu().numpy()
        mri_attributions_np = mri_attributions.detach().cpu().numpy()
        mri_output_path = Path(output_dir) / f"mri_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=mri_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=mri_output_path)
        pet_input_np = pet_images.detach().cpu().numpy()
        pet_attributions_np = pet_attributions.detach().cpu().numpy()
        pet_output_path = Path(output_dir) / f"pet_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=pet_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=pet_output_path)
        del mri_attributions, mri_delta
        del pet_attributions, pet_delta
        torch.cuda.empty_cache()
def nnMamba_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, pretrained_path, experiment_settings, output_dir):
    def forward_func_mri(mri_input, pet_input):
        return model(mri_input, pet_input)

    def forward_func_pet(pet_input, mri_input):
        return model(mri_input, pet_input)
        # 初始化模型并加载到主设备（通常是cuda:0）

    main_device = f'cuda:{device_ids[0]}'
    model = nnMambaMriPet(num_classes=2).to(main_device)
    model.load_state_dict(torch.load(pretrained_path, map_location=main_device))

    # 使用DataParallel包装模型
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.eval()
    dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file, experiment_settings['shape'],
                               experiment_settings['task'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    train_bar = tqdm(dataloader, desc=f"nnMamba IG ", unit="batch")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ii, batch in enumerate(train_bar):
        mri_images = batch.get("mri").to(main_device)
        pet_images = batch.get("pet").to(main_device)
        # cli_tab = batch.get("clinical").to(device)
        # mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
        label = batch.get("label").to(main_device)
        image_name = batch.get("image_name")
        # 定义基线
        mri_baseline = torch.zeros_like(mri_images).to(main_device)
        pet_baseline = torch.zeros_like(pet_images).to(main_device)


        target_class = label.item()
        # 计算 MRI 归因
        # 初始化 Integrated Gradients
        ig_mri = IntegratedGradients(forward_func_mri)
        mri_attributions, mri_delta = ig_mri.attribute(
            inputs=mri_images,
            baselines=mri_baseline,
            additional_forward_args=(pet_images),
            target=target_class,
            return_convergence_delta=True
        )

        # 计算 PET 归因
        ig_pet = IntegratedGradients(forward_func_pet)
        pet_attributions, pet_delta = ig_pet.attribute(
            inputs=pet_images,
            baselines=pet_baseline,
            additional_forward_args=(mri_images),
            target=target_class,
            return_convergence_delta=True
        )

        # 打印结果
        print("MRI Attributions:", mri_attributions.shape)
        print("PET Attributions:", pet_attributions.shape)
        # 示例调用
        mri_input_np = mri_images.detach().cpu().numpy()
        mri_attributions_np = mri_attributions.detach().cpu().numpy()
        mri_output_path = Path(output_dir) / f"mri_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=mri_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=mri_output_path)
        pet_input_np = pet_images.detach().cpu().numpy()
        pet_attributions_np = pet_attributions.detach().cpu().numpy()
        pet_output_path = Path(output_dir) / f"pet_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=pet_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=pet_output_path)
        del mri_attributions, mri_delta
        del pet_attributions, pet_delta
        torch.cuda.empty_cache()


def MDL_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, pretrained_path, experiment_settings, output_dir):
    def forward_func_mri(mri_input, pet_input):
        return model(mri_input, pet_input)

    def forward_func_pet(pet_input, mri_input):
        return model(mri_input, pet_input)
        # 初始化模型并加载到主设备（通常是cuda:0）

    main_device = f'cuda:{device_ids[0]}'
    model = generate_model(model_depth=18, in_planes=1, num_classes=2).to(main_device)
    model.load_state_dict(torch.load(pretrained_path, map_location=main_device))

    # 使用DataParallel包装模型
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.train()
    dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file, experiment_settings['shape'],
                               experiment_settings['task'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    train_bar = tqdm(dataloader, desc=f"nnMamba IG ", unit="batch")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ii, batch in enumerate(train_bar):
        mri_images = batch.get("mri").to(main_device)
        pet_images = batch.get("pet").to(main_device)
        # cli_tab = batch.get("clinical").to(device)
        # mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
        label = batch.get("label").to(main_device)
        image_name = batch.get("image_name")
        # 定义基线
        mri_baseline = torch.zeros_like(mri_images).to(main_device)
        pet_baseline = torch.zeros_like(pet_images).to(main_device)


        target_class = label.item()
        # 计算 MRI 归因
        # 初始化 Integrated Gradients
        ig_mri = IntegratedGradients(forward_func_mri)
        mri_attributions, mri_delta = ig_mri.attribute(
            inputs=mri_images,
            baselines=mri_baseline,
            additional_forward_args=(pet_images),
            target=target_class,
            return_convergence_delta=True
        )

        # 计算 PET 归因
        ig_pet = IntegratedGradients(forward_func_pet)
        pet_attributions, pet_delta = ig_pet.attribute(
            inputs=pet_images,
            baselines=pet_baseline,
            additional_forward_args=(mri_images),
            target=target_class,
            return_convergence_delta=True
        )

        # 打印结果
        print("MRI Attributions:", mri_attributions.shape)
        print("PET Attributions:", pet_attributions.shape)
        # 示例调用
        mri_input_np = mri_images.detach().cpu().numpy()
        mri_attributions_np = mri_attributions.detach().cpu().numpy()
        mri_output_path = Path(output_dir) / f"mri_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=mri_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=mri_output_path)
        pet_input_np = pet_images.detach().cpu().numpy()
        pet_attributions_np = pet_attributions.detach().cpu().numpy()
        pet_output_path = Path(output_dir) / f"pet_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=pet_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=pet_output_path)
        del mri_attributions, mri_delta
        del pet_attributions, pet_delta
        torch.cuda.empty_cache()
def IMF_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, pretrained_path, experiment_settings, output_dir):
    def forward_func_mri(mri_input, pet_input, cli_input):
        return model(mri_input, pet_input, cli_input)

    def forward_func_pet(pet_input, mri_input, cli_input):
        return model(mri_input, pet_input, cli_input)

    def forward_func_cli(cli_input, mri_input, pet_input):
        return model(mri_input, pet_input, cli_input)

    main_device = f'cuda:{device_ids[0]}'
    model = Interactive_Multimodal_Fusion_Model(num_classes=2).to(main_device)
    model.load_state_dict(torch.load(pretrained_path, map_location=main_device))

    # 使用DataParallel包装模型
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.eval()
    dataset = MriPetCliDatasetWithTowLabel(mri_dir, pet_dir, cli_dir, csv_file, experiment_settings['shape'],
                               experiment_settings['task'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    train_bar = tqdm(dataloader, desc=f"IMF IG ", unit="batch")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ii, batch in enumerate(train_bar):
        mri_images = batch.get("mri").to(main_device)
        pet_images = batch.get("pet").to(main_device)
        cli_tab = batch.get("clinical").to(main_device)
        # mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
        label = batch.get("label").to(main_device)
        image_name = batch.get("image_name")
        # 定义基线
        mri_baseline = torch.zeros_like(mri_images).to(main_device)
        pet_baseline = torch.zeros_like(pet_images).to(main_device)
        cli_baseline = torch.zeros_like(cli_tab).to(main_device)
        target_class = label.item()
        # 计算 MRI 归因
        # 初始化 Integrated Gradients
        ig_mri = IntegratedGradients(forward_func_mri)
        mri_attributions, mri_delta = ig_mri.attribute(
            inputs=mri_images,
            baselines=mri_baseline,
            additional_forward_args=(pet_images, cli_tab),
            target=target_class,
            return_convergence_delta=True
        )
        # 计算 PET 归因
        ig_pet = IntegratedGradients(forward_func_pet)
        pet_attributions, pet_delta = ig_pet.attribute(
            inputs=pet_images,
            baselines=pet_baseline,
            additional_forward_args=(mri_images, cli_tab),
            target=target_class,
            return_convergence_delta=True
        )
        # 计算 Clinical 数据归因
        ig_cli = IntegratedGradients(forward_func_cli)
        cli_attributions, cli_delta = ig_cli.attribute(
            inputs=cli_tab,
            baselines=cli_baseline,
            additional_forward_args=(mri_images, pet_images),
            target=target_class,
            return_convergence_delta=True
        )
        # 打印结果
        print("MRI Attributions:", mri_attributions.shape)
        print("PET Attributions:", pet_attributions.shape)
        print("CLI Attributions:", cli_attributions.shape)
        # 示例调用
        mri_input_np = mri_images.detach().cpu().numpy()
        mri_attributions_np = mri_attributions.detach().cpu().numpy()
        mri_output_path = Path(output_dir) / f"mri_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=mri_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=mri_output_path)
        pet_input_np = pet_images.detach().cpu().numpy()
        pet_attributions_np = pet_attributions.detach().cpu().numpy()
        pet_output_path = Path(output_dir) / f"pet_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=pet_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=pet_output_path)
        cli_tab_np = cli_tab[0].detach().cpu().numpy()
        cli_attributions_np = cli_attributions[0].detach().cpu().numpy()
        cli_output_path = Path(output_dir) / f"cli_{image_name}_label_{target_class}.jpg"
        visualize_cli_attributions_heatmap(cli_tab_np, cli_attributions_np, output_path=cli_output_path)
        del mri_attributions, mri_delta
        del pet_attributions, pet_delta
        del cli_attributions, cli_delta
        torch.cuda.empty_cache()
def HFBsurv_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, pretrained_path, experiment_settings, output_dir):
    def forward_func_mri(mri_input, pet_input, cli_input):
        return model(mri_input, pet_input, cli_input)

    def forward_func_pet(pet_input, mri_input, cli_input):
        return model(mri_input, pet_input, cli_input)

    def forward_func_cli(cli_input, mri_input, pet_input):
        return model(mri_input, pet_input, cli_input)

    main_device = f'cuda:{device_ids[0]}'
    model = HFBSurv(num_classes=2).to(main_device)
    model.load_state_dict(torch.load(pretrained_path, map_location=main_device))

    # 使用DataParallel包装模型
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.eval()
    dataset = MriPetCliDataset(mri_dir, pet_dir, cli_dir, csv_file, experiment_settings['shape'],
                               experiment_settings['task'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    train_bar = tqdm(dataloader, desc=f"HFBsurv IG ", unit="batch")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ii, batch in enumerate(train_bar):
        mri_images = batch.get("mri").to(main_device)
        pet_images = batch.get("pet").to(main_device)
        cli_tab = batch.get("clinical").to(device)
        # mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
        label = batch.get("label").to(main_device)
        image_name = batch.get("image_name")
        # 定义基线
        mri_baseline = torch.zeros_like(mri_images).to(main_device)
        pet_baseline = torch.zeros_like(pet_images).to(main_device)
        cli_baseline = torch.zeros_like(cli_tab).to(main_device)
        target_class = label.item()
        # 计算 MRI 归因
        # 初始化 Integrated Gradients
        ig_mri = IntegratedGradients(forward_func_mri)
        mri_attributions, mri_delta = ig_mri.attribute(
            inputs=mri_images,
            baselines=mri_baseline,
            additional_forward_args=(pet_images, cli_tab),
            target=target_class,
            return_convergence_delta=True
        )
        # 计算 PET 归因
        ig_pet = IntegratedGradients(forward_func_pet)
        pet_attributions, pet_delta = ig_pet.attribute(
            inputs=pet_images,
            baselines=pet_baseline,
            additional_forward_args=(mri_images, cli_tab),
            target=target_class,
            return_convergence_delta=True
        )
        # 计算 Clinical 数据归因
        ig_cli = IntegratedGradients(forward_func_cli)
        cli_attributions, cli_delta = ig_cli.attribute(
            inputs=cli_tab,
            baselines=cli_baseline,
            additional_forward_args=(mri_images, pet_images),
            target=target_class,
            return_convergence_delta=True
        )
        # 打印结果
        print("MRI Attributions:", mri_attributions.shape)
        print("PET Attributions:", pet_attributions.shape)
        print("CLI Attributions:", cli_attributions.shape)
        # 示例调用
        mri_input_np = mri_images.detach().cpu().numpy()
        mri_attributions_np = mri_attributions.detach().cpu().numpy()
        mri_output_path = Path(output_dir) / f"mri_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=mri_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=mri_output_path)
        pet_input_np = pet_images.detach().cpu().numpy()
        pet_attributions_np = pet_attributions.detach().cpu().numpy()
        pet_output_path = Path(output_dir) / f"pet_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=pet_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=pet_output_path)
        cli_tab_np = cli_tab[0].detach().cpu().numpy()
        cli_attributions_np = cli_attributions[0].detach().cpu().numpy()
        cli_output_path = Path(output_dir) / f"cli_{image_name}_label_{target_class}.jpg"
        visualize_cli_attributions_heatmap(cli_tab_np, cli_attributions_np, output_path=cli_output_path)
        del mri_attributions, mri_delta
        del pet_attributions, pet_delta
        del cli_attributions, cli_delta
        torch.cuda.empty_cache()


def ITCFN_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, pretrained_path, experiment_settings, output_dir):
    def forward_func_mri(mri_input, pet_input, cli_input):
        return model(mri_input, pet_input, cli_input)

    def forward_func_pet(pet_input, mri_input, cli_input):
        return model(mri_input, pet_input, cli_input)

    def forward_func_cli(cli_input, mri_input, pet_input):
        return model(mri_input, pet_input, cli_input)

    main_device = f'cuda:{device_ids[0]}'
    model = Triple_model_CoAttention_Fusion(num_classes=2).to(main_device)
    model.load_state_dict(torch.load(pretrained_path, map_location=main_device))

    # 使用DataParallel包装模型
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.eval()
    dataset = MriPetCliDataset(mri_dir, pet_dir, cli_dir, csv_file, experiment_settings['shape'],
                               experiment_settings['task'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    train_bar = tqdm(dataloader, desc=f"ITCFN IG ", unit="batch")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ii, batch in enumerate(train_bar):
        mri_images = batch.get("mri").to(main_device)
        pet_images = batch.get("pet").to(main_device)
        cli_tab = batch.get("clinical").to(main_device)
        # mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
        label = batch.get("label").to(main_device)
        image_name = batch.get("image_name")
        # 定义基线
        mri_baseline = torch.zeros_like(mri_images).to(main_device)
        pet_baseline = torch.zeros_like(pet_images).to(main_device)
        cli_baseline = torch.zeros_like(cli_tab).to(main_device)
        target_class = label.item()
        # 计算 MRI 归因
        # 初始化 Integrated Gradients
        ig_mri = IntegratedGradients(forward_func_mri)
        mri_attributions, mri_delta = ig_mri.attribute(
            inputs=mri_images,
            baselines=mri_baseline,
            additional_forward_args=(pet_images, cli_tab),
            target=target_class,
            return_convergence_delta=True
        )
        # 计算 PET 归因
        ig_pet = IntegratedGradients(forward_func_pet)
        pet_attributions, pet_delta = ig_pet.attribute(
            inputs=pet_images,
            baselines=pet_baseline,
            additional_forward_args=(mri_images, cli_tab),
            target=target_class,
            return_convergence_delta=True
        )
        # 计算 Clinical 数据归因
        ig_cli = IntegratedGradients(forward_func_cli)
        cli_attributions, cli_delta = ig_cli.attribute(
            inputs=cli_tab,
            baselines=cli_baseline,
            additional_forward_args=(mri_images, pet_images),
            target=target_class,
            return_convergence_delta=True
        )
        # 打印结果
        print("MRI Attributions:", mri_attributions.shape)
        print("PET Attributions:", pet_attributions.shape)
        print("CLI Attributions:", cli_attributions.shape)
        # 示例调用
        mri_input_np = mri_images.detach().cpu().numpy()
        mri_attributions_np = mri_attributions.detach().cpu().numpy()
        mri_output_path = Path(output_dir) / f"mri_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=mri_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=mri_output_path)
        pet_input_np = pet_images.detach().cpu().numpy()
        pet_attributions_np = pet_attributions.detach().cpu().numpy()
        pet_output_path = Path(output_dir) / f"pet_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=pet_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=pet_output_path)
        cli_tab_np = cli_tab[0].detach().cpu().numpy()
        cli_attributions_np = cli_attributions[0].detach().cpu().numpy()
        cli_output_path = Path(output_dir) / f"cli_{image_name}_label_{target_class}.jpg"
        visualize_cli_attributions_heatmap(cli_tab_np, cli_attributions_np, output_path=cli_output_path)
        del mri_attributions, mri_delta
        del pet_attributions, pet_delta
        del cli_attributions, cli_delta
        torch.cuda.empty_cache()


def MultimodalADNet_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, pretrained_path, experiment_settings, output_dir):
    def forward_func_mri(mri_input, pet_input, cli_input):
        return model(mri_input, pet_input, cli_input)

    def forward_func_pet(pet_input, mri_input, cli_input):
        return model(mri_input, pet_input, cli_input)

    def forward_func_cli(cli_input, mri_input, pet_input):
        return model(mri_input, pet_input, cli_input)

    main_device = f'cuda:{device_ids[0]}'
    model = MultimodalADNet(num_classes=2).to(main_device)
    model.load_state_dict(torch.load(pretrained_path, map_location=main_device))

    # 使用DataParallel包装模型
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.eval()
    dataset = MriPetCliDataset(mri_dir, pet_dir, cli_dir, csv_file, experiment_settings['shape'],
                               experiment_settings['task'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    train_bar = tqdm(dataloader, desc=f"MultimodalADNet IG ", unit="batch")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ii, batch in enumerate(train_bar):
        mri_images = batch.get("mri").to(main_device)
        pet_images = batch.get("pet").to(main_device)
        cli_tab = batch.get("clinical").to(main_device)
        # mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
        label = batch.get("label").to(main_device)
        image_name = batch.get("image_name")
        # 定义基线
        mri_baseline = torch.zeros_like(mri_images).to(main_device)
        pet_baseline = torch.zeros_like(pet_images).to(main_device)
        cli_baseline = torch.zeros_like(cli_tab).to(main_device)
        target_class = label.item()
        # 计算 MRI 归因
        # 初始化 Integrated Gradients
        ig_mri = IntegratedGradients(forward_func_mri)
        mri_attributions, mri_delta = ig_mri.attribute(
            inputs=mri_images,
            baselines=mri_baseline,
            additional_forward_args=(pet_images, cli_tab),
            target=target_class,
            return_convergence_delta=True
        )
        # 计算 PET 归因
        ig_pet = IntegratedGradients(forward_func_pet)
        pet_attributions, pet_delta = ig_pet.attribute(
            inputs=pet_images,
            baselines=pet_baseline,
            additional_forward_args=(mri_images, cli_tab),
            target=target_class,
            return_convergence_delta=True
        )
        # 计算 Clinical 数据归因
        ig_cli = IntegratedGradients(forward_func_cli)
        cli_attributions, cli_delta = ig_cli.attribute(
            inputs=cli_tab,
            baselines=cli_baseline,
            additional_forward_args=(mri_images, pet_images),
            target=target_class,
            return_convergence_delta=True
        )
        # 打印结果
        print("MRI Attributions:", mri_attributions.shape)
        print("PET Attributions:", pet_attributions.shape)
        print("CLI Attributions:", cli_attributions.shape)
        # 示例调用
        mri_input_np = mri_images.detach().cpu().numpy()
        mri_attributions_np = mri_attributions.detach().cpu().numpy()
        mri_output_path = Path(output_dir) / f"mri_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=mri_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=mri_output_path)
        pet_input_np = pet_images.detach().cpu().numpy()
        pet_attributions_np = pet_attributions.detach().cpu().numpy()
        pet_output_path = Path(output_dir) / f"pet_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=pet_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=pet_output_path)
        cli_tab_np = cli_tab[0].detach().cpu().numpy()
        cli_attributions_np = cli_attributions[0].detach().cpu().numpy()
        cli_output_path = Path(output_dir) / f"cli_{image_name}_label_{target_class}.jpg"
        visualize_cli_attributions_heatmap(cli_tab_np, cli_attributions_np, output_path=cli_output_path)
        del mri_attributions, mri_delta
        del pet_attributions, pet_delta
        del cli_attributions, cli_delta
        torch.cuda.empty_cache()

def Diamond_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, pretrained_path, experiment_settings, output_dir):
    def forward_func_pet(pet_input, mri_input):
        return model(pet_input, mri_input)

    def forward_func_mri(mri_input, pet_input):
        return model(pet_input, mri_input)
        # 初始化模型并加载到主设备（通常是cuda:0）

    main_device = f'cuda:{device_ids[0]}'
    diamond = DiaMond()
    regbn_kwargs = {
        'gpu': 0,
        'f_num_channels': 128, # if num_classes == 3 else 128,
        'g_num_channels': 128, # if num_classes == 3 else 128,
        'f_layer_dim': [],
        'g_layer_dim': [],
        'normalize_input': True,
        'normalize_output': True,
        'affine': True,
        'sigma_THR': 0.0,
        'sigma_MIN': 0.0,
    }
    # 创建完整模型
    model = CompleteModel(diamond, regbn_kwargs, 2, device=main_device,
                          checkpoint_path=pretrained_path)
    # model.load_state_dict(torch.load(pretrained_path, map_location=main_device))

    # 使用DataParallel包装模型
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.eval()
    dataset = MriPetDatasetForDiamond(mri_dir, pet_dir, cli_dir, csv_file, experiment_settings['shape'],
                               experiment_settings['task'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    train_bar = tqdm(dataloader, desc=f"Diamond IG ", unit="batch")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ii, batch in enumerate(train_bar):
        mri_images = batch.get("mri").to(main_device)
        pet_images = batch.get("pet").to(main_device)
        # cli_tab = batch.get("clinical").to(device)
        # mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
        label = batch.get("label").to(main_device)
        image_name = batch.get("image_name")
        # 定义基线
        mri_baseline = torch.zeros_like(mri_images).to(main_device)
        pet_baseline = torch.zeros_like(pet_images).to(main_device)


        target_class = label.item()
        # 计算 MRI 归因
        # 初始化 Integrated Gradients
        ig_mri = IntegratedGradients(forward_func_mri)
        mri_attributions, mri_delta = ig_mri.attribute(
            inputs=mri_images,
            baselines=mri_baseline,
            additional_forward_args=(pet_images),
            target=target_class,
            return_convergence_delta=True
        )

        # 计算 PET 归因
        ig_pet = IntegratedGradients(forward_func_pet)
        pet_attributions, pet_delta = ig_pet.attribute(
            inputs=pet_images,
            baselines=pet_baseline,
            additional_forward_args=(mri_images),
            target=target_class,
            return_convergence_delta=True
        )

        # 打印结果
        print("MRI Attributions:", mri_attributions.shape)
        print("PET Attributions:", pet_attributions.shape)
        # 示例调用
        mri_input_np = mri_images.detach().cpu().numpy()
        mri_attributions_np = mri_attributions.detach().cpu().numpy()
        mri_output_path = Path(output_dir) / f"mri_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=mri_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(mri_input_np, mri_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=mri_output_path)
        pet_input_np = pet_images.detach().cpu().numpy()
        pet_attributions_np = pet_attributions.detach().cpu().numpy()
        pet_output_path = Path(output_dir) / f"pet_{image_name}_label_{target_class}.jpg"
        # visualize_ig_heatmap(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48), channel=0,
        #                      output_path=pet_output_path)
        visualize_ig_heatmap_seaborn_save_all_son(pet_input_np, pet_attributions_np, slice_index=(48, 64, 48),
                                                  channel=0,
                                                  output_path=pet_output_path)
        del mri_attributions, mri_delta
        del pet_attributions, pet_delta
        torch.cuda.empty_cache()
if __name__ == "__main__":
    # 初始化模型
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device_ids = [1, 2, 3, 4, 5]
    mri_dir = '/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/MRI'
    pet_dir = '/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/PET'
    cli_dir = './csv/ADNI_Clinical.csv'
    csv_file = './csv/ADNI1_2_pmci_smci_validation.csv'
    experiment_settings = {
        'shape': (96, 128, 96),
        'task': ('pMCI', 'sMCI')
    }

    # Resnet
    # Resnet_pretrained_path = r"/data3/wangchangmiao/shenxy/Code/AD/pmci_smci/pmci_smci/resnet/ResnetMriPet_best_model_fold2.pth"
    # Resnet_output_dir = "./Resnet_IG_output"
    # Resnet_IG(mri_dir, pet_dir, cli_dir, csv_file, device, Resnet_pretrained_path, experiment_settings, Resnet_output_dir)
    # nnMamba
    # nnMamba_pretrained_path = r"/data3/wangchangmiao/shenxy/Code/AD/pmci_smci/pmci_smci/nnMamba/nnMamba_best_model_fold2.pth"
    # nnMamba_output_dir = "./nnMamba_IG_output"
    # nnMamba_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, nnMamba_pretrained_path, experiment_settings, nnMamba_output_dir)
    # Diamond
    # Diamond_pretrained_path = r"/data3/wangchangmiao/shenxy/Code/AD/DiaMond/pmci_smci/DiaMond_multi_split2_bestval.pt"
    # Diamond_output_dir = "./Diamond_IG_output"
    # Diamond_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, Diamond_pretrained_path, experiment_settings, Diamond_output_dir)
    # MDL
    # MDL_pretrained_path = r"/data3/wangchangmiao/shenxy/Code/AD/pmci_smci/pmci_smci/MDL/MDL_Net_best_model_fold2.pth"
    # MDL_output_dir = "./MDL_IG_output"
    # MDL_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, MDL_pretrained_path, experiment_settings, MDL_output_dir)
    # IMF
    # IMF_pretrained_path = r"/data3/wangchangmiao/shenxy/Code/AD/pmci_smci/pmci_smci/IMF/Interactive_Multimodal_Fusion_Model_best_model_fold2.pth"
    # IMF_output_dir = "./IMF_IG_output"
    # IMF_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, IMF_pretrained_path, experiment_settings, IMF_output_dir)
    # HFBsurv
    HFBsurv_pretrained_path = r"/data3/wangchangmiao/shenxy/Code/AD/pmci_smci/pmci_smci/HFBSurv/HFBSurv_best_model_fold2.pth"
    HFBsurv_output_dir = "./HFBsurv_IG_output"
    HFBsurv_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, HFBsurv_pretrained_path, experiment_settings, HFBsurv_output_dir)
    # ITCFN
    # ITCFN_pretrained_path = r"/data3/wangchangmiao/shenxy/Code/AD/pmci_smci/pmci_smci/ITCFN/Triple_model_CoAttention_Fusion_best_model_fold2.pth"
    # ITCFN_output_dir = "./ITCFN_IG_output"
    # ITCFN_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, ITCFN_pretrained_path, experiment_settings, ITCFN_output_dir)
    # MultimodalADNet
    # MultimodalADNet_pretrained_path = r"/data3/wangchangmiao/shenxy/Code/AD/pmci_smci/pmci_smci/MultimodalADNet/MultimodalADNet_best_model_fold2.pth"
    # MultimodalADNet_output_dir = "./MultimodalADNet_IG_output"
    # MultimodalADNet_IG(mri_dir, pet_dir, cli_dir, csv_file, device_ids, MultimodalADNet_pretrained_path, experiment_settings, MultimodalADNet_output_dir)
    # AweSomeNet
    # AweSomeNet_pretrained_path = r"/data3/wangchangmiao/shenxy/Code/AD/AweSome/AwesomeNetFinal_lr00001/pmci_smci_2/AweSomeNet_best_model_fold2.pth"
    # AweSomeNet_output_dir = "./AwesomNet_IG_output"
    # AweSomeNet_IG(mri_dir, pet_dir, cli_dir, csv_file, device, AweSomeNet_pretrained_path, experiment_settings, AweSomeNet_output_dir)




