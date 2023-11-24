import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import math
from sklearn.metrics import f1_score

def auroc(prob, label):
    y_true = label.data.numpy().flatten()
    y_scores = prob.data.numpy().flatten()
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auroc_score = auc(fpr, tpr)
    return auroc_score, fpr, tpr


def AUC(prob, label):
    # y_true = label.data.numpy().flatten()
    # y_scores = prob.data.numpy().flatten()
    fpr, tpr, thresholds = roc_curve(label, prob)
    auroc_score = auc(fpr, tpr)
    return auroc_score

def auprc(prob,label):
    y_true=label.data.numpy().flatten()
    y_scores=prob.data.numpy().flatten()
    precision,recall,thresholds=precision_recall_curve(y_true,y_scores)
    auprc_score=auc(recall,precision)
    return auprc_score,precision,recall

def AUPR(prob,label):
    precision,recall,thresholds=precision_recall_curve(prob,label)
    auprc_score=auc(recall,precision)
    return auprc_score

def MyRealAndPrediction(Real,prediction):
    RealAndPrediction = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter])
        RealAndPrediction.append(pair)
        counter = counter + 1
    return RealAndPrediction

def MyRealAndPredictionProb(Real,prediction):
    RealAndPredictionProb = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter])
        RealAndPredictionProb.append(pair)
        counter = counter + 1
    return RealAndPredictionProb

def Indicator(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    y_real=y_real.data.numpy().flatten()
    y_predict=y_predict.data.numpy().flatten()
    x=[]
    y=[]
    z=[]
    for ele in y_real:
        ele=int(ele)
        x.append(ele)

    for ele in y_predict:
        ele=float(ele)
        z.append(ele)
        if ele >0.5:
            y.append(1)
        else:
            y.append(0)

    RealAndPrediction = MyRealAndPrediction(x, y)
    RealAndPredictionProb = MyRealAndPredictionProb(x, z)

    CM = confusion_matrix(x, y)
    print(CM)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    if (TP+FN)!=0:
        Sen = TP / (TP + FN)
    else:
        Sen=np.inf
    if (TN + FP)!=0:
        Spec = TN / (TN + FP)
    else:
        Spec=np.inf
    if (TP + FP)!=0:
        Prec = TP / (TP + FP)
    else:
        Prec=np.inf
    if math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))!=0:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        MCC=np.inf
    f1 = f1_score(y, x)
    Auc=AUC(torch.tensor(y_predict),torch.tensor(x))
    # Aupr=AUPR(torch.tensor(y_predict),torch.tensor(x))
    # print('MCC:', round(MCC, 4))
    # print('AUC:', Auc)
    # print('F1:', round(f1, 4))
    # print('Acc:', round(Acc, 4))
    # print('Sen:', round(Sen, 4))
    # print('Spec:', round(Spec, 4))
    # print('Prec:', round(Prec, 4))

    Result = []
    Result.append(round(MCC, 4))
    Result.append(round(Auc, 4))
    Result.append(0.0)
    Result.append(round(f1, 4))
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))

    return Result,RealAndPrediction,RealAndPredictionProb

def normalized_input(input):
    a,b,c=input.size()
    input=input.detach().numpy()
    new_input=np.zeros((a,b,c))
    i=0
    while i<a:
        j=0
        while j<b:
            k=0
            while k<c:
                new_input[i][j][k]=(input[i][j][k]-min(input[i][j]))/(max(input[i][j])-min(input[i][j]))
                k+=1
            j+=1
        i+=1
    return torch.from_numpy(new_input).float()

