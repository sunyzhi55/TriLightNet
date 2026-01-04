import numpy as np
from sklearn.manifold import TSNE  # 导入 t-SNE
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from Net.TriLightNet import TriLightNet
from Dataset import MriPetCliDataset
from torch.utils.data import DataLoader
# 假设您已经训练好模型，并且可以从模型中获取二维输出
def get_model_output(model, mri_data, pet_data, clinical_data):
    """
    获取模型的二维输出
    :param model: 训练好的模型
    :param mri_data: MRI 数据 (b, 1, 96, 128, 96)
    :param pet_data: PET 数据 (b, 1, 96, 128, 96)
    :param clinical_data: 临床表格数据 (b, 9)
    :return: 模型的二维输出 (b, 2)
    """
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        outputs, cls = model(mri_data, pet_data, clinical_data)  # 直接获取模型的二维输出
        # outputs = torch.softmax(outputs, dim=1)
    return outputs.cpu().numpy() # (batch, 512)

# 可视化
def plot_embeddings(embeddings_2d, labels=None):
    """
    绘制二维嵌入向量
    :param embeddings_2d: 二维嵌入向量 (b, 2)
    :param labels: 样本标签 (可选)，用于颜色区分
    """
    plt.figure(figsize=(10, 8))
    if labels is not None:
        # 如果有标签，按标签着色
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = labels == label
            plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=f'Class {label}', alpha=0.7)
        plt.legend()
    else:
        # 如果没有标签，统一着色
        sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], alpha=0.7)

    plt.title('Visualization of Model Output')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('tsne.png')
    # plt.show()
# 可视化
def plot_embeddings_3d(embeddings_3d, labels=None):
    """
    绘制三维嵌入向量
    :param embeddings_3d: 三维嵌入向量 (b, 3)
    :param labels: 样本标签 (可选)，用于颜色区分
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')  # 创建 3D 子图

    if labels is not None:
        # 如果有标签，按标签着色
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = labels == label
            ax.scatter(embeddings_3d[idx, 0], embeddings_3d[idx, 1], embeddings_3d[idx, 2],
                       label=f'Class {label}', alpha=0.7)
        ax.legend()
    else:
        # 如果没有标签，统一着色
        ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], alpha=0.7)

    ax.set_title('t-SNE 3D Visualization of Model Embeddings')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')

    plt.savefig('tsne_3d.png')
    # plt.show()
# 使用 t-SNE 进行降维
def apply_tsne(embeddings, n_components=2, perplexity=30):
    """
    使用 t-SNE 对高维嵌入向量进行降维
    :param embeddings: 高维嵌入向量 (b, embedding_dim)
    :param n_components: 目标维度（通常为 2 或 3）
    :param perplexity: t-SNE 的超参数，控制邻域大小
    :return: 降维后的嵌入向量 (b, n_components)
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate='auto', init='pca', random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

if __name__ == "__main__":
    # 示例输入数据
    # mri_data = torch.randn(100, 1, 96, 128, 96)  # 假设有 100 个样本
    # pet_data = torch.randn(100, 1, 96, 128, 96)
    # clinical_data = torch.randn(100, 9)

    mri_dir = '/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/MRI'
    pet_dir = '/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/PET'
    cli_dir = './csv/ADNI_Clinical.csv'
    csv_file = './csv/ADNI1_2_pmci_smci.csv'
    experiment_settings = {
        'shape': (96, 128, 96),
        'task': ('pMCI', 'sMCI')
    }
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    dataset = MriPetCliDataset(mri_dir, pet_dir, cli_dir, csv_file,
                               resize_shape=experiment_settings['shape'],
                               valid_group=experiment_settings['task'])
    dataloader = DataLoader(dataset, batch_size=505, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    mri_data = batch.get("mri").to(device)
    pet_data = batch.get("pet").to(device)
    clinical_data = batch.get("clinical").to(device)
    label_data = batch.get("label")
    print("mri_data", mri_data.shape)
    print("pet_data", pet_data.shape)
    print("clinical_data", clinical_data.shape)
    # 假设模型已经加载并训练完成
    model = TriLightNet().to(device)
    path = r'/data3/wangchangmiao/shenxy/Code/AD/AweSome/AwesomeNetCrossCBAM1DCasCade_MoE/checkpoints_2025-04-22_22-56/AweSomeNet_best_model_fold3.pth'
    model.load_state_dict(torch.load(path))
    embeddings  = get_model_output(model, mri_data, pet_data, clinical_data)
    # 使用 t-SNE 进行降维
    # embeddings_2d = apply_tsne(embeddings, n_components=2, perplexity=30)
    embeddings_3d = apply_tsne(embeddings, n_components=3, perplexity=30)
    # 替换为实际的二维输出
    # embeddings_2d = np.random.randn(100, 2)  # 假设我们有100个样本的二维输出

    # 示例标签（假设每个样本有一个类别标签）
    # labels = np.random.randint(0, 2, size=100)  # 假设有 3 类
    label_data = label_data.numpy()

    # 调用绘图函数
    # plot_embeddings(embeddings_2d, label_data)
    # 调用绘图函数
    plot_embeddings_3d(embeddings_3d, label_data)