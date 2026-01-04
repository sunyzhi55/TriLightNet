import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from Net.TriLightNet import TriLightNet

def visualize_mri_gradcam(model, mri_tensor, class_idx=None, target_layer=None, slice_index=None, use_cuda=True):
    """
    可视化 AweSomeNet 模型中 MRI 分支的 Grad-CAM 热力图（使用中间切片）。

    Args:
        model: 已加载权重的 AweSomeNet 模型
        mri_tensor: MRI 图像张量，形状为 (1, 1, D, H, W)
        class_idx: 想要可视化的类别索引，默认自动获取预测类别
        target_layer: 指定 MRI 分支中的目标卷积层，默认使用 mri_branch 的最后一层
        slice_index: 可视化的切片索引，默认为中间切片
        use_cuda: 是否使用 GPU，默认为 True
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    model = model.to(device)
    mri_tensor = mri_tensor.to(device)

    # 选择切片
    if slice_index is None:
        slice_index = mri_tensor.shape[2] // 2  # 默认取中间层
    input_2d = mri_tensor[:, :, slice_index, :, :]  # shape: (1, 1, H, W)

    # 归一化
    input_2d_normalized = (input_2d - input_2d.min()) / (input_2d.max() - input_2d.min())

    # 自动获取预测类别
    if class_idx is None:
        with torch.no_grad():
            # outputs = model(mri_tensor, torch.zeros_like(mri_tensor), torch.zeros((1, 10)).to(device))  # dummy inputs for PET/tabular
            outputs = model(mri_tensor)  # dummy inputs for PET/tabular
            class_idx = outputs.argmax().item()

    # 自动获取目标层
    if target_layer is None:
        target_layer = model.mri_branch[-1]  # 默认用 MRI 分支最后一层卷积

    # 初始化 GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=mri_tensor, targets=[ClassifierOutputTarget(class_idx)])

    # 构造 RGB 图像用于叠加
    input_np = input_2d_normalized.squeeze().detach().cpu().numpy()
    rgb_img = np.repeat(input_np[..., np.newaxis], 3, axis=-1)
    cam_image = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

    # 可视化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('Original MRI Slice')
    plt.imshow(input_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Grad-CAM')
    plt.imshow(grayscale_cam[0], cmap='jet')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.imshow(cam_image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('gradcam_results.png')

if __name__ == '__main__':
    # 假设 mri_input 是一个大小为 (1, 1, 96, 128, 96) 的 MRI 图像张量
    model = AweSomeNet().MriExtraction
    model.eval()
    mri_input = torch.randn((1, 1, 96, 128, 96))

    target_layer = model.layer4[-1]
    visualize_mri_gradcam(model, mri_input, class_idx=1, target_layer=target_layer)
"""
假如我现在自定义了一个医学方面的三模态融合分类模型，
模型的输入有MRI（batch, 1, 96, 128, 96）、PET(batch, 1, 96, 128, 96)和clinical_table(batch, 8)
输出分类结果（batch, num_classes）
现在我要进行热力图可视化，请问如何实现，能不能调用已有的包实现
"""