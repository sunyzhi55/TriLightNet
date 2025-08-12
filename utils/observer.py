from torchmetrics import Accuracy, Recall, Precision, Specificity, F1Score, CohenKappa
from torchmetrics import AUROC, MetricCollection, ConfusionMatrix, MatthewsCorrCoef
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import Literal

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,  patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class RuntimeObserver:
    def __init__(self, log_dir, device, num_classes=2,
                 task: Literal["binary", "multiclass"]="binary",
                 average: Literal["micro", "macro", "weighted", "none"] = "micro",
                 **kwargs):
        """
        The Observer of training, which contains a log file(.txt), computing tools(torchmetrics) and tensorboard writer
        :author windbell
        :param log_dir: output dir
        :param device: Default to 'cuda'
        :param kwargs: Contains the experiment name and random number seed
        """
        self.best_dicts = {'epoch': 0, 'confusionMatrix':None, 'Accuracy': 0.,
                           'Recall': 0., 'Precision': 0., 'Specificity': 0., 'BalanceAccuracy':0.,
                           'F1': 0., 'AuRoc': 0.,
                           'CohenKappa': 0.}
        self.log_dir = str(log_dir)
        self.log_file = self.log_dir + 'log.txt'
        self._kwargs = {'name': kwargs['name'] if kwargs.__contains__('name') else 'None',
                   'seed': kwargs['seed'] if kwargs.__contains__('seed') else 'None'}
        self.task = task

        # train stage
        self.total_train_loss = 0.
        self.average_train_loss = 0.
        self.train_metric = {}
        self.train_metric_collection = MetricCollection({
            'confusionMatrix': ConfusionMatrix(num_classes=num_classes, task=task).to(device),
            'Accuracy': Accuracy(num_classes=num_classes, task=task).to(device),
            'Precision': Precision(num_classes=num_classes, task=task, average=average).to(device),
            'Recall': Recall(num_classes=num_classes, task=task, average=average).to(device),
            'Specificity': Specificity(num_classes=num_classes, task=task, average=average).to(device),
            'F1': F1Score(num_classes=num_classes, task=task, average=average).to(device),
            'CohenKappa': CohenKappa(num_classes=num_classes, task=task).to(device)
        }).to(device)
        self.train_balance_accuracy = 0.
        self.compute_train_auc = AUROC(num_classes=num_classes, task=task).to(device)
        self.train_auc = 0.

        # test stage
        self.total_eval_loss = 0.
        self.average_eval_loss = 0.
        self.eval_metric = {}
        self.eval_metric_collection = MetricCollection({
            'confusionMatrix': ConfusionMatrix(num_classes=num_classes, task=task).to(device),
            'Accuracy': Accuracy(num_classes=num_classes, task=task).to(device),
            'Precision': Precision(num_classes=num_classes, task=task, average=average).to(device),
            'Recall': Recall(num_classes=num_classes, task=task, average=average).to(device),
            'Specificity': Specificity(num_classes=num_classes, task=task, average=average).to(device),
            'F1': F1Score(num_classes=num_classes, task=task, average=average).to(device),
            'CohenKappa': CohenKappa(num_classes=num_classes, task=task).to(device)
        }).to(device)
        self.eval_balance_accuracy = 0.
        self.compute_eval_auc = AUROC(num_classes=num_classes, task=task).to(device)
        self.eval_auc = 0.

        self.summary = SummaryWriter(log_dir=self.log_dir + 'summery')
        self.early_stopping = EarlyStopping(patience=50, verbose=True)
        self.log('exp:' + str(self._kwargs['name']) + '  seed -> ' + str(self._kwargs['seed']))

    def reset(self):
        self.total_train_loss = 0.
        self.average_train_loss = 0.
        self.train_metric = {}
        self.train_metric_collection.reset()
        self.compute_train_auc.reset()
        self.train_auc = 0.
        self.train_balance_accuracy = 0.

        self.total_eval_loss = 0.
        self.average_eval_loss = 0.
        self.eval_metric = {}
        self.eval_metric_collection.reset()
        self.compute_eval_auc.reset()
        self.eval_auc = 0.
        self.eval_balance_accuracy = 0.

    def log(self, info: str):
        print(info)
        with open (f'{self.log_file}', 'a') as f:
            f.write(info)

    def train_update(self, loss, prob, prediction, label):
        if self.task=='binary':
            prob_positive = prob[:, 1]
        else:
            prob_positive = prob
        self.total_train_loss += loss.item()
        self.train_metric_collection.forward(prediction, label)
        self.compute_train_auc.update(prob_positive, label)

    def eval_update(self, loss, prob, prediction, label):
        if self.task=='binary':
            prob_positive = prob[:, 1]
        else:
            prob_positive = prob
        self.total_eval_loss += loss.item()
        self.eval_metric_collection.forward(prediction, label)
        self.compute_eval_auc.update(prob_positive, label)

    def compute_result(self, epoch, train_dataset_length, eval_dataset_length):
        self.average_train_loss = self.total_train_loss / train_dataset_length
        self.average_eval_loss = self.total_eval_loss / eval_dataset_length
        self.train_metric = self.train_metric_collection.compute()
        self.eval_metric = self.eval_metric_collection.compute()
        self.train_auc = self.compute_train_auc.compute()
        self.eval_auc = self.compute_eval_auc.compute()
        self.train_balance_accuracy = (self.train_metric['Recall'] + self.train_metric['Specificity']) / 2.0
        self.eval_balance_accuracy = (self.eval_metric['Recall'] + self.eval_metric['Specificity']) / 2.0

        # self.summary.add_scalar('train_loss', self.average_train_loss, epoch)
        self.summary.add_scalar('val_loss', self.average_eval_loss, epoch)
        self.summary.add_scalar('eval_accuracy', self.eval_metric['Accuracy'], epoch)
        self.summary.add_scalar('eval_precision', self.eval_metric['Precision'], epoch)
        self.summary.add_scalar('eval_recall', self.eval_metric['Recall'], epoch)
        self.summary.add_scalar('eval_spe', self.eval_metric['Specificity'], epoch)
        self.summary.add_scalar('eval_balance_accuracy', self.eval_balance_accuracy, epoch)
        self.summary.add_scalar('eval_f1', self.eval_metric['F1'], epoch)
        self.summary.add_scalar('eval_auc', self.eval_auc, epoch)
        # self.summary.add_scalar('train_cohen_kappa', self.train_metric['CohenKappa'], epoch)
        self.summary.add_scalar('eval_cohen_kappa', self.eval_metric['CohenKappa'], epoch)

    def print_result(self, e, epochs):
        train_output_result = (f"Epoch [{e}/{epochs}]:, train_loss={self.average_train_loss:.3f}, \n"
                           f"train_confusionMatrix:\n{self.train_metric['confusionMatrix']}\n"
                           f"train_accuracy={self.train_metric['Accuracy']}, \n"
                           f"train_recall={self.train_metric['Recall']}, \n"
                           f"train_precision={self.train_metric['Precision']}, \n"
                           f"train_specificity={self.train_metric['Specificity']}, \n"
                           f"train_cohen_kappa={self.train_metric['CohenKappa']}, \n"
                           f"train_balance_acc={self.train_balance_accuracy},\n "
                           f"train_f1_score={self.train_metric['F1']},\n "
                           f"train_auc={self.train_auc}\n")
        eval_output_result = (f"Epoch [{e}/{epochs}]:, eval_loss={self.average_eval_loss:.3f}, \n"
                           f"eval_confusionMatrix:\n{self.eval_metric['confusionMatrix']}\n"
                           f"eval_accuracy={self.eval_metric['Accuracy']}, \n"
                           f"eval_recall={self.eval_metric['Recall']}, \n"
                           f"eval_precision={self.eval_metric['Precision']}, \n"
                           f"eval_specificity={self.eval_metric['Specificity']}, \n"
                           f"eval_cohen_kappa={self.eval_metric['CohenKappa']}, \n"
                           f"eval_balance_acc={self.eval_balance_accuracy},\n "
                           f"eval_f1_score={self.eval_metric['F1']},\n "
                           f"eval_auc={self.eval_auc}\n\n")
        self.log(train_output_result)
        self.log(eval_output_result)

    def get_best(self, epoch):
        # if self.eval_metric['Accuracy'] > self.best_dicts['Accuracy']:
        self.best_dicts['epoch'] = epoch
        self.best_dicts['confusionMatrix'] = self.eval_metric['confusionMatrix']
        self.best_dicts['Accuracy'] = self.eval_metric['Accuracy']
        self.best_dicts['Recall'] = self.eval_metric['Recall']
        self.best_dicts['Precision'] = self.eval_metric['Precision']
        self.best_dicts['Specificity'] = self.eval_metric['Specificity']
        self.best_dicts['BalanceAccuracy'] = (self.eval_metric['Specificity'] + self.eval_metric['Recall']) /2.0
        self.best_dicts['F1'] = self.eval_metric['F1']
        self.best_dicts['CohenKappa'] = self.eval_metric['CohenKappa']
        self.best_dicts['AuRoc'] = self.eval_auc

    def execute(self, e, epoch, train_dataset_length, eval_dataset_length, fold, model=None):
        self.compute_result(epoch, train_dataset_length, eval_dataset_length)
        self.print_result(e, epoch)
        # if self.eval_metric['Accuracy'] > self.best_dicts['Accuracy']:
        if (self.eval_metric['Specificity'] + self.eval_metric['Recall']) /2.0 > self.best_dicts['BalanceAccuracy']:
            # abs(total_precision - total_recall) <= abs(self.best_dicts['p'] - self.best_dicts['recall']):
            self.get_best(e)
            model_save_path = self.log_dir + f'{str(self._kwargs["name"])}_best_model_fold{fold}.pth'
            torch.save(model.state_dict(), model_save_path)
            self.log(f"Best model saved to {model_save_path}\n\n")
        self.early_stopping(self.eval_metric['Accuracy'])
        return self.early_stopping.early_stop

    def finish(self, fold):
        best_result = (f"Fold {fold} Best Epoch: {self.best_dicts['epoch']}\n"
                   f"Best confusionMatrix : {self.best_dicts['confusionMatrix']}\n"
                   f"Best accuracy : {self.best_dicts['Accuracy']}, \n"
                   f"Best recall : {self.best_dicts['Recall']}, \n"
                   f"Best precision : {self.best_dicts['Precision']}, \n"
                   f"Best specificity : {self.best_dicts['Specificity']}, \n"
                   f"Best cohen_kappa : {self.best_dicts['CohenKappa']}, \n"
                   f"Best balance_acc : {self.best_dicts['BalanceAccuracy']},\n "
                   f"Best f1_score : {self.best_dicts['F1']},\n "
                   f"Best AUC : {self.best_dicts['AuRoc']}\n\n")
        self.log(best_result)