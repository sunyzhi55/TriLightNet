import time
from Dataset import *
import numpy as np
from Net import *
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc


def find_best_model_for_ITCFN(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_dir, device, save_image_path, experiment_settings, num_classes=2):
    def evaluate_model(model, data_loader, device):
        """评估单个模型并返回AUC值"""
        model.to(device)
        model.eval()
        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                cli_tab = batch.get("clinical").to(device)
                label = batch.get("label").to(device)
                _, _, _, outputs_logit = model(mri_images, pet_images, cli_tab)
                prob = torch.softmax(outputs_logit, dim=1)
                prob_positive = prob[:, 1]  # 获取正类的概率
                all_probabilities.extend(prob_positive.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        result_dict = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            # 'labels': all_labels,
            # 'probs': all_probabilities
        }
        return result_dict
    """查找并评估所有fold模型，返回最佳模型"""
    # 准备数据集
    dataset = MriPetCliDataset(mri_dir, pet_dir, cli_dir, csv_file,
                            resize_shape=experiment_settings['shape'], valid_group=experiment_settings['task'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 使用Path对象
    model_dir = Path(model_dir)
    # 查找所有pth文件
    model_files = list(model_dir.glob('*fold*.pth'))
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} with pattern '*fold*.pth'")
    # 存储所有折的结果
    all_results = []
    best_dict = {
        "name": "ITCFN",
        "auc": -1,
        "fpr": None,
        "tpr": None,
        "fold": 0
    }
    # 评估每个模型
    for model_path in model_files:
        # 从文件名中提取fold编号
        filename = model_path.name.lower()
        fold_num = None
        if 'fold1' in filename:
            fold_num = 1
        elif 'fold2' in filename:
            fold_num = 2
        elif 'fold3' in filename:
            fold_num = 3
        elif 'fold4' in filename:
            fold_num = 4
        elif 'fold5' in filename:
            fold_num = 5
        if fold_num is None:
            print(f"Warning: Could not determine fold number for {filename}, skipping")
            continue
        print(f"Evaluating fold {fold_num} model: {model_path.name}")
        model = Triple_model_CoAttention_Fusion(num_classes=num_classes)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        result_dict = evaluate_model(model, data_loader, device)
        result_dict['fold'] = fold_num
        all_results.append(result_dict)
        print(f"Fold {fold_num} AUC: {result_dict['auc']:.4f}")
        if result_dict["auc"] > best_dict["auc"]:
            best_dict['auc'] = result_dict['auc']
            best_dict['fpr'] = result_dict['fpr']
            best_dict['tpr'] = result_dict['tpr']
            best_dict["fold"] = fold_num
    # 绘制所有折的ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, result in enumerate(all_results):
        plt.plot(result['fpr'], result['tpr'],
                 color=colors[i],
                 label=f'Fold {result["fold"]} (AUC = {result["auc"]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model {best_dict["name"]} ROC Curves for All Folds')
    plt.legend(loc='lower right')
    # 保存图片
    output_path = Path(save_image_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")
    return best_dict, all_results

def find_best_model_for_AwesomeNet(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_dir, device, save_image_path, experiment_settings, num_classes=2):
    def evaluate_model(model, data_loader, device):
        """评估单个模型并返回AUC值"""
        model.to(device)
        model.eval()
        all_labels = []
        all_probabilities = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                cli_tab = batch.get("clinical").to(device)
                label = batch.get("label").to(device)
                outputs_logit = model(mri_images, pet_images, cli_tab)
                prob = torch.softmax(outputs_logit, dim=1)
                prob_positive = prob[:, 1]  # 获取正类的概率
                all_probabilities.extend(prob_positive.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        result_dict = {
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            # 'labels': all_labels,
            # 'probs': all_probabilities
        }
        return result_dict
    """查找并评估所有fold模型，返回最佳模型"""
    # 准备数据集
    dataset = MriPetCliDataset(mri_dir, pet_dir, cli_dir, csv_file,
                            resize_shape=experiment_settings['shape'], valid_group=experiment_settings['task'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 使用Path对象
    model_dir = Path(model_dir)
    # 查找所有pth文件
    model_files = list(model_dir.glob('*fold*.pth'))
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} with pattern '*fold*.pth'")
    # 存储所有折的结果
    all_results = []
    best_dict = {
        "name": "AweSomeNet",
        "auc": -1,
        "fpr": None,
        "tpr": None,
        "fold": 0
    }
    # 评估每个模型
    for model_path in model_files:
        # 从文件名中提取fold编号
        filename = model_path.name.lower()
        fold_num = None
        if 'fold1' in filename:
            fold_num = 1
        elif 'fold2' in filename:
            fold_num = 2
        elif 'fold3' in filename:
            fold_num = 3
        elif 'fold4' in filename:
            fold_num = 4
        elif 'fold5' in filename:
            fold_num = 5
        if fold_num is None:
            print(f"Warning: Could not determine fold number for {filename}, skipping")
            continue
        print(f"Evaluating fold {fold_num} model: {model_path.name}")
        model = Triple_model_CoAttention_Fusion(num_classes=num_classes)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        result_dict = evaluate_model(model, data_loader, device)
        result_dict['fold'] = fold_num
        all_results.append(result_dict)
        print(f"Fold {fold_num} AUC: {result_dict['auc']:.4f}")
        if result_dict["auc"] > best_dict["auc"]:
            best_dict['auc'] = result_dict['auc']
            best_dict['fpr'] = result_dict['fpr']
            best_dict['tpr'] = result_dict['tpr']
            best_dict["fold"] = fold_num
    # 绘制所有折的ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, result in enumerate(all_results):
        plt.plot(result['fpr'], result['tpr'],
                 color=colors[i],
                 label=f'Fold {result["fold"]} (AUC = {result["auc"]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model {best_dict["name"]} ROC Curves for All Folds')
    plt.legend(loc='lower right')
    # 保存图片
    output_path = Path(save_image_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")
    return best_dict, all_results

if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # 加载数据 - 使用Path对象
    mri_dir = Path('/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/MRI')
    pet_dir = Path('/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/PET')
    cli_dir = Path('./csv/ADNI_Clinical.csv')
    csv_file = Path('./csv/ADNI1_2_ad_mci_cn_validation.csv')
    batch_size = 8
    experiment_settings = {
        'shape': (96, 128, 96),
        'task': ('CN', 'AD')
    }
    all_model_best = []
    start_time = time.time()


    # ITCFN
    # ITCFN_model_dir = Path(r'/data3/wangchangmiao/shenxy/Code/AD/ad_cn_87/ad_cn/ITCFN')
    # ITCFN_save_image_path = Path('./roc_curves_ITCFN.png')
    # # 获取最佳模型和所有结果
    # best_dict_ITCFN, all_results_ITCFN = find_best_model_for_ITCFN(
    #     mri_dir, pet_dir, cli_dir, csv_file, batch_size, ITCFN_model_dir, device,
    #     save_image_path = ITCFN_save_image_path,
    #     experiment_settings=experiment_settings,
    #     num_classes=2
    # )
    # print(f"model:{best_dict_ITCFN['name']} Fold result")
    # print(f"{all_results_ITCFN}")
    # all_model_best.append(best_dict_ITCFN)

    # AweSomeNet
    AwesomeNet_model_dir = Path(r'/data3/wangchangmiao/shenxy/Code/AD/AweSome/AwesomeNetCrossCBAM1DCasCade_MoE/ad_cn')
    AwesomeNet_save_image_path = Path('./roc_curves_AwesomeNet.png')
    best_dict_AwesomeNet, all_results_AwesomeNet = find_best_model_for_AwesomeNet(
        mri_dir, pet_dir, cli_dir, csv_file, batch_size, AwesomeNet_model_dir, device,
        save_image_path = AwesomeNet_save_image_path,
        experiment_settings=experiment_settings,
        num_classes=2
    )
    print(f"model:{best_dict_AwesomeNet['name']} Fold result")
    print(f"{all_results_AwesomeNet}")
    all_model_best.append(best_dict_AwesomeNet)
    end_time = time.time()
    use_time = end_time - start_time
    # 计算小时、分钟和秒
    hours = use_time // 3600
    minutes = (use_time % 3600) // 60
    seconds = use_time % 60
    # 打印总训练时间
    times_result = f'Total time: {hours}h {minutes}m {seconds}s'
    print(times_result)
    """
    已知有一个all_model_best为list,里面存放着各种模型的k折交叉验证中最好的指标，
    每个元素的结构为所示        best_dict = {
        "name": "ITCFN",
        "auc": -1,
        "fpr": 0.35,
        "tpr": 0.5,
        "fold": 0
    }
    请你根据all_model_best里的内容，画出所有模型的ROC曲线，放在一张图中，并且保存为图片
    """

