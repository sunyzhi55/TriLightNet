from torch import optim
from torch.nn import init
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix,matthews_corrcoef
import matplotlib.pyplot as plt
from Net.ComparisonNet import *
# 定义scheduler
def get_scheduler(optimizer, opt):
    """
    scheduler definition
    :param optimizer:  original optimizer
    :param opt: corresponding parameters
    :return: corresponding scheduler
    """
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=0.00001)
    elif opt.lr_policy == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
def estimate(y_true, y_logit, y_prob):
    train_cm = confusion_matrix(y_true, y_logit)
    tp = train_cm[1, 1]  # 正类被正确预测为正类的数量
    fp = train_cm[0, 1]  # 负类被错误预测为正类的数量
    fn = train_cm[1, 0]  # 正类被错误预测为负类的数量
    tn = train_cm[0, 0]  # 负类被正确预测为负类的数量（在二分类中通常不单独使用）

    acc = accuracy_score(y_true, y_logit)
    recall = recall_score(y_true, y_logit, average='binary')  # 对于二分类问题
    precision = precision_score(y_true, y_logit, average='binary')  # 对于二分类问题
    f1_scores = f1_score(y_true, y_logit, average='binary')  # 对于二分类问题
    # 计算Specificity（真负率）
    specificity = tn / (tn + fp)
    # Sensitivity其实就是recall，但为了完整性，我们也可以在这里返回它
    sensitivity = recall
    # 计算AUC（需要y_true和正类的预测概率）
    auc = roc_auc_score(y_true, y_prob)
    # 计算MCC
    mcc = matthews_corrcoef(y_true, y_logit)
    return tp, fp, fn, tn, acc, recall, precision, f1_scores, specificity, sensitivity, auc, mcc

# 画图展示
def plot_training_results(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))

    # 绘制损失变化图
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率变化图
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('training_results.png')