
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import linear_model, datasets
from sklearn.svm import SVC
import numpy as np
from sklearn import tree
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
import copy
import os
import math
import random


## define the ml model

def etmain(X_train_std, Y_train, X_test_std, Y_test):
    model = ExtraTreesClassifier()
    model.fit(X_train_std, Y_train)
    predict = model.predict(X_test_std)
    lrpredpro = model.predict_proba(X_test_std)
    groundtruth = Y_test
    predictprob = lrpredpro
    return groundtruth, predict, predictprob , model

def svmmain(X_train_std, Y_train, X_test_std, Y_test):
    model = SVC(probability=True, kernel='rbf')
    model.fit(X_train_std, Y_train, sample_weight=None)
    predict = model.predict(X_test_std)
    predictprob =model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob  , model

def knnmain(X_train_std, Y_train, X_test_std, Y_test):
    model = KNeighborsClassifier(n_neighbors=7) 
    model.fit(X_train_std, Y_train)
    predict = model.predict(X_test_std)
    predictprob =model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob  , model

def xgmain(X_train_std, Y_train, X_test_std, Y_test):
    model = XGBClassifier()
    model.fit(X_train_std, Y_train)
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob  , model

def dtmain(X_train_std, Y_train, X_test_std, Y_test):
    model = tree.DecisionTreeClassifier(criterion = 'gini', splitter = "best", min_samples_split = 2)
    model.fit(X_train_std, Y_train)
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob  , model

def rfmain(X_train_std, Y_train, X_test_std, Y_test):
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train_std, Y_train)
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob , model

def gbdtmain(X_train_std, Y_train, X_test_std, Y_test):
    model = GradientBoostingClassifier()
    model.fit(X_train_std, Y_train)
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob , model



## evaluate the ml model


def evaluate(BaselineName,dataset):
    X_train_std, Y_train, X_test_std, Y_test=dataset
    if BaselineName == 'ExtraTrees':
        groundtruth, predict, predictprob , model = etmain (X_train_std, Y_train, X_test_std, Y_test)
    elif BaselineName == 'KNeighbors':
        groundtruth, predict, predictprob , model= knnmain(X_train_std, Y_train, X_test_std, Y_test)
    elif BaselineName =='XGBoost':
        groundtruth, predict, predictprob , model= xgmain(X_train_std, Y_train, X_test_std, Y_test)
    elif BaselineName =='DecisionTree':
        groundtruth, predict, predictprob , model = dtmain(X_train_std, Y_train, X_test_std, Y_test)
    elif BaselineName =='RandomForest':
        groundtruth, predict, predictprob , model = rfmain(X_train_std, Y_train, X_test_std, Y_test)
    elif BaselineName =='GradientBoosting':
        groundtruth, predict, predictprob , model= gbdtmain(X_train_std, Y_train, X_test_std, Y_test)
    else:
        return


    acc = metrics.accuracy_score(groundtruth, predict)
    precision = metrics.precision_score(groundtruth, predict, zero_division=1 )
    recall = metrics.recall_score(groundtruth, predict)
    f1 = metrics.f1_score(groundtruth, predict)
    tn, fp, fn, tp = metrics.confusion_matrix(groundtruth, predict).ravel()
    ppv = tp/(tp+fp+1.4E-45)
    npv = tn/(fn+tn+1.4E-45)
    mcc=metrics.matthews_corrcoef(groundtruth, predict)
    item={'BaselineName':BaselineName,'Accuracy':acc,'Precision':precision,'MCC':mcc,'PPV':ppv,'NPV':npv,'Recall':recall,'F1':f1,'TP':tp,'FP':fp,'TN':tn,'FN':fn}
    return groundtruth, predict, predictprob,item  , model



## plot roc
def ROC_plot(Y_test,y_score,filename):
    y_label=[]
    for i in range(len(Y_test)):
        y_label+=[[0,0]]
    for i in range(len(Y_test)):
        y_label[i][int(Y_test.values[i])]=1
    y_label=np.array(y_label)
    n_classes = 2

    # calculate ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    lw=2
    plt.figure(figsize=(8,8))
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i+1, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(filename[filename.rfind('/')+1:filename.rfind('.')])
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.show()
    


## plot confusion matrix
def plot_matrix(y_true, y_pred,filename):
    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm,annot=True,fmt='g',xticklabels=['Declined', 'Approved'],yticklabels=['Declined', 'Approved'],annot_kws={"fontsize":20})
    #xticklabels、yticklabels
    ax.set_xlabel('Predict',size=20) 
    ax.set_ylabel('GroundTruth',size=20) 
    plt.xticks(fontsize=15) 
    plt.yticks(fontsize=15) 
    plt.gcf().set_size_inches(8, 6)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=15)
    plt.savefig(filename)
    plt.show()