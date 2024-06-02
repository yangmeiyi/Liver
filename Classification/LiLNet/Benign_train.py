from __future__ import print_function
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import argparse
import shutil
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from resnet50_our import Our_resnet50
from dataloader.fnh_hem_cyst_dataloader import CancerSeT_CSV
from collections import OrderedDict
from utils.variables import *
from process import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Datasets
parser.add_argument('-d', '--dataset', default='B', type=str)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--num_classes', default=3, type=int, metavar='N',
                    help='number of classification')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=512, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 15],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--save', default=True, action='store_true', help='save model')
parser.add_argument('--pretrain', default=True, action='store_true', help='load pretrained param form Imagenet')
parser.add_argument('--upload', default=True, action='store_true', help='load param form medical')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet', help='model architecture')
parser.add_argument('--depth', type=int, default=50, help='Model depth.')
parser.add_argument('--block-name', type=str, default='Bottleneck',
                    help='the building block for ResNet: BasicBlock, Bottleneck')

# Ours
parser.add_argument('--beta', type=float, default=1., help='Ratio for the second to last.')
parser.add_argument('--self-distillation', action='store_true', default=True,
                    help='Utilizing self-distillation in ours or not.')
parser.add_argument('--fc_mode', type=str, default='2fcsd', help='')
parser.add_argument('--mask_mode', type=str, default='mask_mode', help='')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

# Device options
parser.add_argument('--gpu-id', default='0,1,2,3,4,5,6,7', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


args = parser.parse_args()


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Save Arguments
state = {k: v for k, v in args._get_kwargs()}
if not os.path.isdir(os.path.join(args.checkpoint)):
    mkdir_p(os.path.join(args.checkpoint))
with open(os.path.join(args.checkpoint, 'arguments.txt'), 'w') as f:
    for key in state.keys():
        f.writelines(str(key) + ': ' + str(state[key]) + '\n')

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


def save_acc_checkpoint(state, epoch, checkpoint="./", filename='B_our_sd_acc.pth.tar'):
    save_filename = "epoch_" + str(epoch) + "_" + filename
    filepath = os.path.join(checkpoint, save_filename)
    torch.save(state, filepath)


def main():
    global best_acc_SI, best_acc_AG, best_acc_SD
    best_acc_SI = 0.
    best_acc_AG = 0.
    best_acc_SD = 0.

    # Data
    print('==> Preparing Dataset %s For %d Classes' % (args.dataset, args.num_classes))
    print('    Dataset: %s\n    Num Classes: %d' % (args.dataset, args.num_classes))
    PATH = "/home/dkd/Data_4TDISK/dataset_ct_ap_pvp_crop_224/"  
    Liver_loader_train = CancerSeT_CSV(PATH, 'train')
    Liver_loader_test = CancerSeT_CSV(PATH, 'test')
    Liver_loader_val_hn = CancerSeT_CSV(PATH, 'val_hn')


    train_loader = torch.utils.data.DataLoader(Liver_loader_train, batch_size=args.train_batch, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(Liver_loader_test, batch_size=args.test_batch, shuffle=False)
    val_loader = torch.utils.data.DataLoader(Liver_loader_val_hn, batch_size=args.test_batch, shuffle=False)

    # Model
    print("==> Creating Model '{}'".format(args.arch+str(args.depth)))
    model = Our_resnet50(
        num_classes=args.num_classes * 5,
        block_name=args.block_name,
        self_distillation=args.self_distillation
    )

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()


    if up_load:
        checkpoint = torch.load(r"./B.pth.tar")
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # if pretrained:
    #     # replace_param(model=model)
    #     dict_trained = torch.load("../resnet50-19c8e357.pth")  # map_location=torch.device('cpu')
    #     dict_new = model.state_dict()
    #     # 1. filter out unnecessary keys
    #     for k in dict_trained.keys():
    #         if k in dict_new.keys() and not k.startswith('fc'):
    #             dict_new[k] = dict_trained[k]
    #
    #     # 2. overwrite entries in the existing state dict
    #     model.load_state_dict(dict_new)



    cudnn.benchmark = True
    model_params = sum(p.numel() for p in model.parameters()) / 1000000.0
    print('Total params: %.2fM' % model_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    # Train and val
    best_signal_acc_test = 0
    best_person_auc_test = 0
    best_person_acc_test = 0


    for epoch in range(args.start_epoch, args.epochs): #
        adjust_learning_rate(optimizer, epoch)
        print('\nDataset: {0} | Model: {1} | Params: {2:.2f}M | Beta: {3:.1f} | SD: {4}'.
              format(args.dataset, args.arch+str(args.depth), model_params, args.beta, args.self_distillation))
        print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        if Need_train
            train_process(train_loader, model, criterion, optimizer, use_cuda, args.beta, args.self_distillation)
        for data_loader in [test_loader, val_loader]:  # 
            test_loss, test_acc, acc_statistic, auc_statistic = test_process(data_loader, model, criterion, use_cuda, args.beta, args.self_distillation)
            if save_model and data_loader == test_loader:
                best_person_auc_test = max(best_person_auc_test, auc_statistic)
                best_signal_acc_test = max(best_signal_acc_test, test_acc)
                best_person_acc_test = max(best_person_acc_test, acc_statistic)
                test_acc_is_best = acc_statistic >= best_person_acc_test
                test_auc_is_best = auc_statistic >= best_person_auc_test
                save_acc_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'auc': best_person_auc_test,
                    'acc': best_person_acc_test,
                    'optimizer': optimizer.state_dict(),
                }, epoch)       
        print(" best_person_auc: {}".format(best_person_auc_test) + " best_signal_acc: {}".format(best_signal_acc_test) + " best_person_acc: {}\n".format(best_person_acc_test))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def get_total_time(start_time, end_time):
    total_seconds = end_time - start_time
    day = total_seconds // 86400
    hour = (total_seconds % 86400) // 3600
    minute = ((total_seconds % 86400) % 3600) // 60
    second = ((total_seconds % 86400) % 3600) % 60
    print('Total time: {}day {}hour {}min {:.2f}s'.format(day, hour, minute, second))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    get_total_time(start, end)
