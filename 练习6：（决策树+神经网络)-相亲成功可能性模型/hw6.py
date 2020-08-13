# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:24:10 2020

@author: Away
"""
'''
1、背景介绍：

一家婚恋网站公司希望根据已注册用户的历史相亲数据,建立新用户相亲成功可能性的预测模型，数据存放在“date_data2.csv”中。

2、主要变量说明如下：

#income-月均收入（元）

#attractive-由婚恋网站评定出的个人魅力值,分值从0-100。

#assets-资产(万元)

#edueduclass-教育等级:1=小学,2=初中;3=高中,4=本科,5=硕士及以上

#Dated-是否相亲成功:1代表成功

3、作业安排：

3.1 基础知识：

      1）比较逻辑回归、决策树、神经网络的算法差异性比较。

3.2 案例解答步骤如下：

     1）使用决策树、神经网络建立相亲成功预测模型并通过调节超参数进行模型调优，比较两个模型的优劣。

      2)对income,attractive,assets进行分箱(5分箱)处理，用分箱后的数据建模，并比较与1）步骤中模型的表现是否有差异。
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(r"C:\Users\Away\Desktop\笔记\数据分析\数据科学实践")
#%%
data0=pd.read_csv('date_data2.csv')#源数据data0
data0.head()
#%%
#1、决策树（省略了变量筛选之类的）
## 1.训练集测试集划分
import sklearn.model_selection as cross_validation

target = data0['Dated']  # 选取目标变量
data_x=data0.iloc[:,:4]  # 选取自变量,有4个 0,1,2,3列
#data_x=data0.loc[:,:'edueduclass']  # 选取自变量,有4个

#建立训练集，测试集，random_state:随机抽样种子
train_data, test_data, train_target, test_target = cross_validation.train_test_split(data_x,target, test_size=0.4, train_size=0.6 ,random_state=12345) # 划分训练集和测试集

#%%
## 1.2决策树建模
import sklearn.tree as tree
#初始随便设置一个深度和划分
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=5) # 当前支持计算信息增益和GINI
clf.fit(train_data, train_target)  #  使用训练数据建模
#mytest=clf.predict_proba(train_data)
# 查看模型预测结果
# predict_proba返回的是一个 n 行 k 列的数组， 第 i 行 第 j 列上的数值是模型预测 第 i 个预测样本为某个标签的概率，并且每一行的概率和为1
train_est = clf.predict(train_data)  #  用模型预测训练集的结果
train_est_p=clf.predict_proba(train_data)[:,1]  #用模型预测训练集的概率
test_est=clf.predict(test_data)  #  用模型预测测试集的结果
test_est_p=clf.predict_proba(test_data)[:,1]  #  用模型预测测试集的概率 （为1的概率。成功的概率）
compare=pd.DataFrame({'test_target':test_target,'test_est':test_est,'test_est_p':test_est_p}).T # 查看测试集预测结果与真实结果对比
#%%
## 1.3评估
import sklearn.metrics as metrics

print(metrics.confusion_matrix(test_target, test_est,labels=[0,1]))  # 混淆矩阵
print(metrics.classification_report(test_target, test_est))  # 计算评估指标
print(pd.DataFrame(list(zip(data_x.columns, clf.feature_importances_))))  # 变量重要性指标
'''
              precision    recall  f1-score   support

           0       0.95      0.83      0.89        24
           1       0.79      0.94      0.86        16

    accuracy                           0.88        40
   macro avg       0.87      0.89      0.87        40
weighted avg       0.89      0.88      0.88        40

             0         1
0       income  0.000000
1   attractive  0.069195
2       assets  0.817452 资产最重要。。
3  edueduclass  0.113353
'''
#%%

# 根据roc曲线看出是否有多度拟合的情况
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)
plt.figure(figsize=[6,6])
plt.plot(fpr_test, tpr_test,color='blue')#test:蓝色
plt.plot(fpr_train, tpr_train, color='red')#train:红色
plt.title('ROC curve')
plt.show()
#严重过拟合。。
#%%
## 1.4 调参模型优化
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
#参数字典
param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[2,3,4,5],#模型逐渐变复杂
    'min_samples_split':[2,4,6,8,10,12]#枝杈最小样本越大，模型越简单
}
clf = tree.DecisionTreeClassifier()#这个也可以作为网格参数去选择
#网格搜素调参数
clfcv = GridSearchCV(estimator=clf, param_grid=param_grid, 
                   scoring='roc_auc', cv=4)#cv=4,每一种组合再进行4组交叉验证。总共有7*7*4个决策树，评分函数roc
clfcv.fit(train_data, train_target)
#%%
clfcv.best_params_#查看最好的参数
'''
Out[17]: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 2}
'''
#%%
# 查看模型预测结果
train_est = clfcv.predict(train_data)  #  用模型预测训练集的结果
train_est_p=clfcv.predict_proba(train_data)[:,1]  #用模型预测训练集的概率
test_est=clfcv.predict(test_data)  #  用模型预测测试集的结果
test_est_p=clfcv.predict_proba(test_data)[:,1]  #  用模型预测测试集的概率
#%%
# 1.5模型评估
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)
plt.figure(figsize=[6,6])
plt.plot(fpr_test, tpr_test, color='blue')#蓝色的是test
plt.plot(fpr_train, tpr_train, color='red')#红色的是train
plt.title('ROC curve')
plt.show()
#有改进，但是还是有过拟合现象。
#%%
print('AUC = %6.4f' %metrics.auc(fpr_test, tpr_test)) # AUC = 0.9440
#%%%
#################################################
#2、神经网络
## 2.1 标准化（神经网络必须先将变量进行极差标准化）

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)
#%%
#2.2.调参
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

param_grid = {
    'hidden_layer_sizes':[(15, ), (5, 5),(3, 6),(4, 8),(5, 6)],
    'activation':['logistic', 'tanh', 'relu'], 
    'alpha':[0.001, 0.01, 0.1, 0.2, 0.4, 1, 10]
}
mlp = MLPClassifier(max_iter=1000)
gcv = GridSearchCV(estimator=mlp, param_grid=param_grid, 
                   scoring='roc_auc', cv=4, n_jobs=-1)
gcv.fit(scaled_train_data, train_target)
#%%
gcv.best_params_
'''
{'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (5, 5)}
'''
#%%
#2.3 评估
train_est2 = gcv.predict(scaled_train_data)  #  用模型预测训练集的结果
train_est_p2=gcv.predict_proba(scaled_train_data)[:,1]  #用模型预测训练集的概率
test_est2=gcv.predict(scaled_test_data)  #  用模型预测测试集的结果
test_est_p2=gcv.predict_proba(scaled_test_data)[:,1]  #  用模型预测测试集的概率
#%%
fpr_test2, tpr_test2, th_test2 = metrics.roc_curve(test_target, test_est_p2)
fpr_train2, tpr_train2, th_train2 = metrics.roc_curve(train_target, train_est_p2)
plt.figure(figsize=[6,6])
plt.plot(fpr_test2, tpr_test2, color='blue')#蓝色的是test
plt.plot(fpr_train2, tpr_train2, color='red')#红色的是train
plt.title('ROC curve')
plt.show()
#%%
print('AUC = %6.4f' %metrics.auc(fpr_test2, tpr_test2)) # AUC = 0.9323 奇怪决策树的效果还好点

#############################################################
### 对连续变量进行分箱处理之后建模 




























