from torch.utils.data import DataLoader, Dataset, Subset
import time
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import torch.utils.data
from model_object import models
from Config import parse_args
from utils.observer import RuntimeObserver
from utils.api import *
#99480885

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        sample = self.subset[index]
        if self.transform:
            # 对MRI和PET分别应用转换
            if 'mri' in sample:
                sample['mri'] = self.transform(sample['mri'])
                sample['mri'] = sample['mri'].as_tensor()
            if 'pet' in sample:
                sample['pet'] = self.transform(sample['pet'])
                sample['pet'] = sample['pet'].as_tensor()
            if 'gm' in sample:
                sample['gm'] = self.transform(sample['gm'])
                sample['gm'] = sample['gm'].as_tensor()
            if 'wm' in sample:
                sample['wm'] = self.transform(sample['wm'])
                sample['wm'] = sample['wm'].as_tensor()
        return sample

    def __len__(self):
        return len(self.subset)

def prepare_to_train(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_index,
                     seed, device, data_parallel, n_splits, others_params):
    global experiment_settings
    assert torch.cuda.is_available(), "Please ensure codes are executed on cuda."
    try:
        experiment_settings = models[model_index]
    except KeyError:
        print('model not in model_object!')
    torch.cuda.empty_cache()

    # 初始化数据集
    # if model_index == 'IMF':
    #     dataset = MriPetDatasetWithTowLabel(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("pMCI", "sMCI"))
    # elif model_index == 'MDL':
    #     dataset = MriPetDatasetWithTwoInput(mri_dir, pet_dir, csv_file, valid_group=("pMCI", "sMCI"))
    # else:
    #     dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("pMCI", "sMCI"))
    # ADNI1_mri_dir = '/data3/wangchangmiao/shenxy/ADNI/ADNI1/MRI'
    # ADNI1_pet_dir = '/data3/wangchangmiao/shenxy/ADNI/ADNI1/PET'
    # ADNI1_csv_file = './csv/ADNI1_match.csv'
    # ADNI2_mri_dir = '/data3/wangchangmiao/shenxy/ADNI/ADNI2/MRI'
    # ADNI2_pet_dir = '/data3/wangchangmiao/shenxy/ADNI/ADNI2/PET'
    # ADNI2_csv_file = './csv/ADNI2_match.csv'
    # ADNI1_dataset = experiment_settings['dataset'](ADNI1_mri_dir, ADNI1_pet_dir, cli_dir, ADNI1_csv_file,
    #                                                resize_shape=experiment_settings['shape'],
    #                                                valid_group=experiment_settings['task'])
    # ADNI2_dataset = experiment_settings['dataset'](ADNI2_mri_dir, ADNI2_pet_dir, cli_dir, ADNI2_csv_file,
    #                                                resize_shape=experiment_settings['shape'],
    #                                                valid_group=experiment_settings['task'])
    # dataset = ADNI1_dataset + ADNI2_dataset


    dataset = experiment_settings['dataset'](mri_dir, pet_dir, cli_dir, csv_file,
                                             resize_shape=experiment_settings['shape'],
                                             valid_group=experiment_settings['task'])
    torch.manual_seed(seed)

    # K折交叉验证
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = [data.get("label") for data in dataset]  # 假设dataset[i]的第3项是label
    num_classes = len(experiment_settings['task'])
    # 存储每个fold的评估指标
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'balanceAccuracy':[],
        'Specificity': [],
        'auc': [],
        'CohenKappa':[],
        'f1': [],
    }
    # 训练日志和监控
    current_time = str(datetime.now().strftime('%Y-%m-%d_%H-%M'))
    target_dir = f'./{others_params.checkpoints_dir}_{current_time}/'
    Path(target_dir).mkdir(exist_ok=True)
    # for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
    for fold, (train_index, test_index) in enumerate(skf.split(dataset, labels), 1):
        observer = RuntimeObserver(log_dir=target_dir, device=device, num_classes=num_classes,
                                   task="multiclass" if num_classes > 2 else "binary",
                                   average='macro' if num_classes > 2 else 'micro',
                                   name=experiment_settings['Name'], seed=seed)
        observer.log(f'Fold {fold}/{n_splits}')
        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        val_sampler = torch.utils.data.SubsetRandomSampler(test_index)
        # train_subset = Subset(dataset, train_idx)
        # val_subset = Subset(dataset, val_idx)
        # train_dataset = TransformedSubset(train_subset, transform=total_transform['train_transforms'])
        # val_dataset = TransformedSubset(val_subset, transform=total_transform['validation_transforms'])

        trainDataLoader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size,
                                                      num_workers=4, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(dataset, sampler=val_sampler, batch_size=batch_size,
                                                     num_workers=4, drop_last=True)
        # 分割数据集
        # train_dataset = torch.utils.data.Subset(dataset, train_index)
        # test_dataset = torch.utils.data.Subset(dataset, test_index)
        #
        # trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
        #                                               num_workers=4, drop_last=True)
        # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
        #                                              num_workers=4)

        observer.log(f'[DEBUG] Observer init successfully, program start @{current_time}\n')
        # 模型加载
        _model = experiment_settings['Model']
        if model_index == 'MDL':
            model = _model(model_depth=18, in_planes=1, num_classes=num_classes)
        elif model_index == 'RLAD':
            _, model = _model()
        else:
            print(f"The name of model will run {_model}")
            model = _model(num_classes=num_classes)
        # 使用 DataParallel 进行多GPU训练
        # if torch.cuda.device_count() > 1 and data_parallel == 1:
        #     observer.log("Using " + str(torch.cuda.device_count()) + " GPUs for training.\n")
        #     model = torch.nn.DataParallel(model)

        observer.log(f'Use model : {str(experiment_settings)}\n')
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        observer.log("\n===============================================\n")
        observer.log("model parameters: " + str(num_params))
        observer.log("\n===============================================\n")

        # 超参数设置
        # 获取优化器类和学习率
        optimizer_class = experiment_settings['Optimizer']
        learning_rate = experiment_settings['Lr']
        # 检查 weight_decay 是否存在
        if 'weight_decay' in experiment_settings and experiment_settings['weight_decay'] is not None:
            weight_decay = experiment_settings['weight_decay']
            optimizer = optimizer_class(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            optimizer = optimizer_class(
                model.parameters(),
                lr=learning_rate
            )
        # 定义一个filter，只传入requires_grad=True的模型参数
        # optimizer = experiment_settings['Optimizer'](filter(lambda p: p.requires_grad, model.parameters()),
        #                                              experiment_settings['Lr'])

        # MDL
        # optimizer = experiment_settings['Optimizer'](model.parameters(), lr=experiment_settings['Lr'],
        #                                              weight_decay=experiment_settings['weight_decay'],
        #                                              momentum=experiment_settings['momentum'])

        # RLAD
        # optimizer = experiment_settings['Optimizer'](model.parameters(), experiment_settings['Lr'],
        #                                              weight_decay=experiment_settings['weight_decay'])

        scheduler = experiment_settings['Scheduler'](optimizer, others_params)

        if 'w1' in experiment_settings:
            criterion = experiment_settings['Loss'](w1=experiment_settings['w1'], w2=experiment_settings['w2'])
        elif experiment_settings['Name'] == 'MDL_Net':
            criterion = experiment_settings['Loss'](label_smoothing=experiment_settings['label_smoothing'])
        else:
            criterion = experiment_settings['Loss']()

        print("Prepare completed for fold {}! Launch training!\U0001F680".format(fold))

        # 启动训练
        _run = experiment_settings['Run']
        _run(observer, others_params.epochs, trainDataLoader, testDataLoader, model, device,
             optimizer, criterion, scheduler, fold)

        # 收集评估指标
        metrics['accuracy'].append(observer.best_dicts['Accuracy'])
        metrics['precision'].append(observer.best_dicts['Precision'])
        metrics['recall'].append(observer.best_dicts['Recall'])
        metrics['Specificity'].append(observer.best_dicts['Specificity'])
        metrics['balanceAccuracy'].append(observer.best_dicts['BalanceAccuracy'])
        metrics['auc'].append(observer.best_dicts['AuRoc'])
        metrics['CohenKappa'].append(observer.best_dicts['CohenKappa'])
        metrics['f1'].append(observer.best_dicts['F1'])
    print("Cross-validation training completed for all folds.")
    return metrics

if __name__ == "__main__":
    args = parse_args()
    print(args)
    time_start = time.time()
    best_metrics = prepare_to_train(mri_dir=args.mri_dir, pet_dir=args.pet_dir, cli_dir=args.cli_dir,
                                    csv_file=args.csv_file, batch_size=args.batch_size, model_index=args.model,
                                    seed=args.seed, device=args.device, data_parallel=args.data_parallel,
                                    n_splits=args.n_splits, others_params=args)
    time_over = time.time()
    use_time = time_over - time_start
    # 计算小时、分钟和秒
    hours = use_time // 3600
    minutes = (use_time % 3600) // 60
    seconds = use_time % 60
    # 打印总训练时间
    times_result = f'Total time: {hours}h {minutes}m {seconds}s'
    print(f"Metrics:{best_metrics}")
    print(times_result)
    with open(args.logs, 'a') as f:
        f.write(f"Metrics:{best_metrics} \n ")
        f.write(times_result + '\n')

