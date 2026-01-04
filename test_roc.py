import torch
from torch.utils.data import DataLoader
from Dataset import MriPetDataset, MriPetCliDatasetWithTowLabel
from Net.ComparisonNet import HFBSurv, Interactive_Multimodal_Fusion_Model
import numpy as np
from PIL import Image  # 处理高质量图片
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    pretrainedModelPath = r'/data3/wangchangmiao/shenxy/Code/MICCAI/TripleNet_Contrast/HFSurv/adni1_fold1_best_model.pth'
    model = HFBSurv()
    # model = Interactive_Multimodal_Fusion_Model()
    model.load_state_dict(torch.load(pretrainedModelPath, map_location=device))
    model = model.to(device)
    model.eval()

    # 加载数据
    mri_dir = r'/data3/wangchangmiao/shenxy/ADNI/ADNI1/MRI'
    pet_dir = r'/data3/wangchangmiao/shenxy/ADNI/ADNI1/PET'
    cli_dir = r'./csv/ADNI_Clinical.csv'
    csv_file = r'./csv/ADNI1_match_validation.csv'
    batch_size = 8
    dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("pMCI", "sMCI"))
    # dataset = MriPetDatasetWithTowLabel(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("pMCI", "sMCI"))

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 假设你的标签是二分类问题，并且存储在label中
    all_labels = []
    all_probabilities = []

    # 在循环外部初始化列表以保存所有预测概率和真实标签

    with torch.no_grad():  # 关闭梯度计算
        for batch_idx, (mri, pet, clinical, label) in enumerate(data_loader):
            mri, pet = mri.to(device), pet.to(device)
            clinical = clinical.to(device)
            # 前向传播
            outputs = model(mri, pet, clinical)
            _, predictions = torch.max(outputs, dim=1)
            # prob_positive = outputs[:, 1]
            # 保存当前批次的概率和标签
            all_probabilities.extend(outputs[:, 1].cpu().numpy())  # 假设正类是索引为1的列
            all_labels.extend(label.cpu().numpy())

    # with torch.no_grad():  # 关闭梯度计算
    #     for batch_idx, (mri, pet, clinical, label, label_2d) in enumerate(data_loader):
    #         mri, pet = mri.to(device), pet.to(device)
    #         clinical = clinical.to(device)
    #         # 前向传播
    #         outputs = model(mri, pet, clinical)
    #         prob = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4.0
    #         _, predictions = torch.max(prob, dim=1)
    #         # 保存当前批次的概率和标签
    #         all_probabilities.extend(prob[:, 1].cpu().numpy())  # 假设正类是索引为1的列
    #         all_labels.extend(label.cpu().numpy())

    # 计算ROC曲线的相关指标
    print("all_labels",all_labels)
    print("all_probabilities", all_probabilities)
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    lw = 2  # 线宽
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('roc_curve.png')