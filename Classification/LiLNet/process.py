from __future__ import print_function

import time

import torch
import torch.nn.parallel
import torch.nn.functional as F
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np
from utils import *




def train_process(data_loader, model, criterion, optimizer, use_cuda, beta, self_distillation=False):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    top_SI = AverageMeter()
    top_AG = AverageMeter()
    if self_distillation:
        top_SD = AverageMeter()

    end = time.time()


    for batch_idx, data_img in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = data_img["img"].float()
        targets = data_img["label"].float()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda().long()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        joint_targets = torch.stack([targets * 5 + i
                                     for i in range(5)], 1).view(-1)



        if self_distillation:
            outputs3, outputs2, outputs_SD = model(inputs)
        else:
            outputs3, outputs2 = model(inputs)

        outputs_SI = outputs3[::5, ::5]

        outputs_AG = 0.
        for j in range(5):
            outputs_AG = outputs_AG + (outputs3[j::5, j::5]
                                       + outputs2[j::5, j::5]) / 10
        # measure loss
        if self_distillation:
            T = 3.00
            SD_loss = F.kl_div(F.log_softmax(outputs_SD / T, dim=1),
                               F.softmax(outputs_AG.detach() / T, dim=1), reduction='batchmean')
            loss = criterion(outputs3, joint_targets) + beta * criterion(outputs2, joint_targets) \
                   + (SD_loss.mul(T ** 2) + criterion(outputs_SD, targets) if self_distillation else 0.)
        else:
            loss = criterion(outputs3, joint_targets) + beta * criterion(outputs2, joint_targets) # + criterion(outputs_SI, targets)

        # print(outputs3, joint_targets)
        # exit()

        # measure accuracy
        prec_SI = accuracy(outputs_SI.data, targets.data, topk=(1,))
        prec_AG = accuracy(outputs_AG.data, targets.data, topk=(1,))

        top_SI.update(prec_SI[0], inputs.size(0))
        top_AG.update(prec_AG[0], inputs.size(0))

        if self_distillation:
            prec_SD = accuracy(outputs_SD.data, targets.data, topk=(1,))
            top_SD.update(prec_SD[0], inputs.size(0))

        # record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # finish
        if batch_idx % 50 == 0:
            print('[' + '{:5}'.format(batch_idx * len(data_img)) + '/' + '{:5}'.format(len(data_loader.dataset)) +
                  ' (' + '{:3.0f}'.format(100 * batch_idx / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()) + ' AG:' + '{:6.2f}'.format(prec_AG[0].item()))
    return losses.avg, top_SI.avg


def test_process(data_loader, model, criterion, use_cuda, beta, self_distillation=False):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    top_SI = AverageMeter()
    top_AG = AverageMeter()
    if self_distillation:
        top_SD = AverageMeter()

    end = time.time()
    with torch.no_grad():
        total_samples = len(data_loader.dataset)
        correct_samples = 0
        total_loss = 0

        people_id = []
        pred_list = []
        neg_pred_list = []
        pred_list_new = []
        labels = []
        paths_list = []
        for batch_idx, data_img in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = data_img["img"].float()
            targets = data_img["label"].float()
            paths = data_img["image_path"]

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda().long()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            joint_targets = torch.stack([targets * 5 + i
                                         for i in range(5)], 1).view(-1)

            # targets clips num
            if self_distillation:
                outputs3, outputs2, outputs_SD = model(inputs)
            else:
                outputs3, outputs2 = model(inputs)

            outputs_SI = outputs3[::5, ::5]

            outputs_AG = 0.
            for j in range(5):
                outputs_AG = outputs_AG + (outputs3[j::5, j::5]
                                           + outputs2[j::5, j::5]) / 10
            # measure loss
            if self_distillation:
                T = 3.00
                SD_loss = F.kl_div(F.log_softmax(outputs_SD / T, dim=1),
                                   F.softmax(outputs_AG.detach() / T, dim=1), reduction='batchmean')
                loss = criterion(outputs3, joint_targets) + beta * criterion(outputs2, joint_targets) \
                       + (SD_loss.mul(T ** 2) + criterion(outputs_SD, targets) if self_distillation else 0.)
            else:
                loss = criterion(outputs3, joint_targets) + beta * criterion(outputs2, joint_targets)
            # measure accuracy

            prec_SI = accuracy(outputs_SI.data, targets.data, topk=(1,))
            prec_AG = accuracy(outputs_AG.data, targets.data, topk=(1,))

            top_SI.update(prec_SI[0], inputs.size(0))
            top_AG.update(prec_AG[0], inputs.size(0))

            if self_distillation:
                prec_SD = accuracy(outputs_SD.data, targets.data, topk=(1,))
                top_SD.update(prec_SD[0], inputs.size(0))

            # measure person accuracy using AG
            _, pred = torch.max(outputs_AG, dim=1)
            correct_samples += pred.eq(targets).sum()
            people_id.extend(data_img['id'])
            pred_list.extend(outputs_AG.detach().cpu().numpy())  # detel .tolist()
            # pred_list_new.extend(pred.detach().cpu().numpy())
            paths_list.extend(paths)
            labels.extend(targets.detach().cpu().numpy().tolist())

            # record loss
            losses.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    acc = 100.0 * correct_samples / total_samples

    # multi classification
    df = pd.DataFrame({'people_id': people_id, 'preds': pred_list, 'labels': labels, 'path': paths_list})
    df_result_save(df)
    df = pd.DataFrame({'people_id': people_id, 'preds': pred_list, 'labels': labels})
    df = df.groupby('people_id')[['labels', 'preds']]
    person_preds, person_label, person_preds_label, acc_statistic = ACC_3Clas_statistic(df)
    auc_statistic = AUC_3Clas_statistic(person_preds, person_label)
    Confusion_Mat_3Clas_statistic(person_label, person_preds_label)

    print('Average test loss: ' + '{:.4f}'.format(loss) +'  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +'{:4.2f}'.format(acc) + '%)' + 'statis acc: ' + '{:4.2f}'.format(acc_statistic))

    return loss, acc, acc_statistic, auc_statistic

def ACC_error_statistic(df):
    error_name = []
    error_path = []
    error_pred = []
    error_label = []
    # person_label = []
    person_preds = []
    # person_preds_label = []
    df_group = df.groupby('people_id')[['labels', 'preds', 'path']]
    for name, id_group in df_group:
        pred_pro = np.mean(id_group["preds"].values, axis=0)
        person_preds.append(list(pred_pro))
        preds_pro = float(np.argmax([pred_pro]))
        # person_preds_label.append(preds_pro)
        labels = float(id_group["labels"].mean())
        # person_label.append(labels)
        if int(preds_pro) != int(labels):
            if labels == 1 and preds_pro == 2:
                print(name)
            for index, row in id_group.iterrows():
                pred = float(np.argmax(row["preds"]))
                label = row["labels"]
                if int(pred) != int(label):
                    error_name.append(name)
                    error_path.append(row["path"])
                    error_pred.append(pred)
                    error_label.append(label)
    # assert len(error_name) == len(error_path) == len(error_label) == len(error_pred)
    # dataframe = pd.DataFrame({"patient": error_name, "path": error_path, "pred": error_pred, "label": error_label})
    # dataframe.to_csv("sy_fnh_hem_cyst_error.csv", index=False, sep=",")

def df_result_save(df):
    person_preds = []
    person_label = []
    name_list = []
    df_group = df.groupby('people_id')[['labels', 'preds']]
    for name, id_group in df_group:
        pred_pro = np.mean(id_group["preds"].values, axis=0)
        preds_pro = float(np.argmax([pred_pro]))
        person_preds.append(preds_pro)
        labels = float(id_group["labels"].mean())
        person_label.append(labels)
        name_list.append(name)
    assert len(name_list) == len(person_label) == len(person_preds)
    dataframe = pd.DataFrame({"patient": name_list, "pred": person_preds, "label": person_label})
    dataframe.to_csv("doctor_BM.csv", index=False, sep=",")

def Auc2Class(df):
    def threshold(ytrue, ypred):
        fpr, tpr, thresholds = metrics.roc_curve(ytrue, ypred)
        y = tpr - fpr
        youden_index = np.argmax(y)
        optimal_threshold = thresholds[youden_index]
        point = [fpr[youden_index], tpr[youden_index]]
        print(optimal_threshold)
        return optimal_threshold, point, fpr, tpr

    single_threshold, single_point, single_fpr, single_tpr = threshold(df['labels'], df['neg_preds'])

    auc_single = metrics.roc_auc_score(df['labels'], df['neg_preds'])
    df['single'] = (df['neg_preds'] >= 0.5).astype(int)
    acc_single = (df['labels'] == df['single']).mean()

    df = df.groupby('people_id')[['labels', 'neg_preds']].mean()
    statistic_threshold, statistic_point, statistic_fpr, statistic_tpr = threshold(df['labels'], df['neg_preds'])
    df['outputs'] = (df['neg_preds'] >= 0.5).astype(int)
    auc_statis = metrics.roc_auc_score(df['labels'], df['neg_preds'])

    acc_statistic = (df['labels'] == df['outputs']).mean()
    df_sensitivity = df.loc[df["labels"] == 1]
    statistic_sensitivity = (df_sensitivity['labels'] == df_sensitivity['outputs']).mean()
    df_specificity = df.loc[df["labels"] == 0]
    statistic_specificity = (df_specificity['labels'] == df_specificity['outputs']).mean()


    return acc_single, acc_statistic, auc_single, auc_statis, single_threshold, statistic_threshold, \
           single_fpr, single_tpr, single_point, statistic_sensitivity, statistic_specificity, statistic_point


def ACC_3Clas_statistic(df):
    id_count = 0
    person_label = []
    person_preds = []
    person_preds_label = []
    for name, id_group in df:
        preds_pro = np.mean(id_group["preds"].values, axis=0)
        person_preds.append(list(preds_pro))
        preds_pro = float(np.argmax([preds_pro]))
        person_preds_label.append(preds_pro)
        labels = float(id_group["labels"].mean())
        person_label.append(labels)
        if int(preds_pro) == int(labels):
            id_count += 1
        # else:
        #     print("predicted error: ", name)
    acc_statistic = id_count / len(df)
    return person_preds, person_label, person_preds_label, acc_statistic

def macro_auc(y_true, person_preds):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        roc_auc[i] = metrics.roc_auc_score(y_true[:, i], person_preds[:, i])
        print("class {} ".format(i) + 'statis auc ' + '{:.4f}'.format(roc_auc[i]))
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], person_preds[:, i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(2):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 2
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    # np.save('/home/dkd/Code/HCC_ICC/tu/result/B_our_fpr_hn.npy', fpr["macro"])
    # np.save('/home/dkd/Code/HCC_ICC/tu/result/B_our_tpr_hn.npy', tpr["macro"])
    roc_auc["macro"] = metrics.roc_auc_score(y_true, person_preds, average="macro")
    # plt.figure()
    # plt.plot(fpr["macro"], tpr["macro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    # plt.show()
    return roc_auc["macro"]

def micro_auc(y_true, person_preds):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        roc_auc[i] = metrics.roc_auc_score(y_true[:, i], person_preds[:, i])
        print("class {} ".format(i) + 'statis auc ' + '{:.4f}'.format(roc_auc[i]))
    print("macro_auc: ", roc_auc["micro"])
    print("weigthed_auc: ", metrics.roc_auc_score(y_true, person_preds, average="weighted"))
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), person_preds.ravel())
    # np.save('/home/dkd/Code/HCC_ICC/tu/result/our_fpr_1.npy', fpr["micro"])
    # np.save('/home/dkd/Code/HCC_ICC/tu/result/our_tpr_1.npy', tpr["micro"])
    # plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    # plt.show()
    return roc_auc["micro"]

def list_onehot(actions: list, n: int):
    result = []
    for action in actions:
        result.append([int(k == action) for k in range(n)])
    return result


def AUC_3Clas_statistic(person_preds, person_label):
    # y_true = label_binarize(person_label, classes=[0, 1, 2]) # using for three classification
    y_true = list_onehot(person_label, 2 ) # # using for two classification
    y_true = np.array(y_true)
    person_preds = np.array(person_preds)
    roc_auc = macro_auc(y_true, person_preds)
    # roc_auc = micro_auc(y_true, person_preds)
    print("macro_auc: ", roc_auc)
    return roc_auc

def Confusion_Mat_3Clas_statistic(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print("weighted_f1: ", f1_score(y_true, y_pred, average='weighted'))
    print("macro_f1: ", f1_score(y_true, y_pred, average='macro'))
    print("weighted_recall: ", recall_score(y_true, y_pred, labels=[0., 1.], average='weighted'))
    print("macro_recall: ", recall_score(y_true, y_pred, labels=[0., 1.], average='macro'))
    print("weighted_precision: ", precision_score(y_true, y_pred, labels=[0., 1.], average='weighted'))
    print("macro_precision: ", precision_score(y_true, y_pred, labels=[0., 1.], average='macro'))
    confusion_data = confusion_matrix(y_true, y_pred, labels=[0., 1.])
    print("confusion matrix: \n", confusion_data)
    # plt.matshow(confusion_data, cmap=plt.cm.Reds)
