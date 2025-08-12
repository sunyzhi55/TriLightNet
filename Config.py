import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='AD Contrast Experiment')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--drop_ratio', default=0,
                        help='Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1')
    parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--cli_dir', type=str, default='./csv/ADNI_Clinical.csv', help='pet input path')
    parser.add_argument('--model', type=str, default='TriLightNet')
    parser.add_argument("--seed", default=42, type=int, help="seed given by LinkStart.py on cross Val")

    parser.add_argument("--n_splits", default=5, type=int, help="0~4")
    parser.add_argument("--data_parallel", default=0, type=int, help="test on server")


    # data_dir
    # /home/publicdata/ADNI/ADNI1
    # /mntcephfs/med_dataset/huxiangyang/ADNI/freesurfer/ADNI1
    # /data3/wangchangmiao/ADNI/freesurfer/ADNI1
    parser.add_argument('--mri_dir', type=str, default='/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/MRI', help='mri input path')
    parser.add_argument('--pet_dir', type=str, default='/data3/wangchangmiao/shenxy/ADNI/ADNI1_2/PET', help='pet input path')
    parser.add_argument('--csv_file', type=str, default='./csv/ADNI1_2_pmci_smci_train.csv', help='label input path')


    parser.add_argument('--best_result_model_path', type=str,default='best_result.pth', help='the best result model path')
    parser.add_argument('--device', type=str, default='cuda:4', help='device to run')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--class_num', type=int, default=2, help='the number of class')
    parser.add_argument('--m', type=float, default=0.999, help='ema momentum decay for prototype update scheme')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs to train")
    parser.add_argument('--print_freq', type=int, default=1,help='frequency of showing training results on console')
    parser.add_argument('--save_epoch_freq', type=int, default=20,help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='initial lambda decay value')
    parser.add_argument('--niter', type=int, default=50, help='iter of cosine Scheduler')
    parser.add_argument('--lr_policy', type=str, default='cosine', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--interpolation_lambda', type=float, default=20.0, help='interpolation strength')
    parser.add_argument('--logs', type=str, default='./logs.txt', help='logs to record the result')
    args = parser.parse_args()

    return args
