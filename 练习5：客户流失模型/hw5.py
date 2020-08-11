# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:32:08 2020

@author: Away
"""
'''
主要变量说明如下：
#subscriberID="个人客户的ID"
#churn="是否流失：1=流失";
#Age="年龄"
#incomeCode="用户居住区域平均收入的代码"
#duration="在网时长"
#peakMinAv="统计期间内最高单月通话时长"
#peakMinDiff="统计期间结束月份与开始月份相比通话时长增加数量"
#posTrend="该用户通话时长是否呈现出上升态势：是=1"
#negTrend="该用户通话时长是否呈现出下降态势：是=1"
#nrProm="电话公司营销的数量"
#prom="最近一个月是否被营销过：是=1"
#curPlan="统计时间开始时套餐类型：1=最高通过200分钟；2=300分钟；3=350分钟；4=500分钟"
#avPlan="统计期间内平均套餐类型"
#planChange="统计期间是否更换过套餐：1=是"
#posPlanChange="统计期间是否提高套餐：1=是"
#negPlanChange="统计期间是否降低套餐：1=是"
#call_10086="拨打10086的次数"


步骤如下：

（一）	两变量分析：检验该用户通话时长是否呈现出上升态势(posTrend)对流失(churn) 是否有预测价值

（二）	首先将原始数据拆分为训练和测试数据集，使用训练数据集建立在网时长对流失的逻辑回归，使用测试数据集制作混淆矩阵（阈值为0.5），提供准确性、召回率指标，提供ROC曲线和AUC。

（三）	使用向前逐步法从其它备选变量中选择变量，构建基于AIC的最优模型，绘制ROC曲线，同时检验模型的膨胀系数。
'''
#%%
import os
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
os.chdir(r'C:\Users\Away\Desktop\笔记\数据分析\’）
data0 = pd.read_csv('telecom_churn.csv', skipinitialspace=True)
data0.head()
#%%
#（一）	两变量分析：检验该用户通话时长是否呈现出上升态势(posTrend)对流失(churn) 是否有预测价值
# 两个分类变量的相关关系，列联表分析（卡方检验）
cross_table = pd.crosstab(data0.posTrend,data0.churn, margins=True)#margins 显示总数
#%%
def percConvert(ser):
    return ser/float(ser[-1])

cross_table.apply(percConvert, axis=1)#做行百分比
'''
churn      0.0   1.0   All
posTrend                  
0.0        829   990  1819
1.0       1100   544  1644
All       1929  1534  3463
'''

# In[8]:

print('''chisq = %6.4f 
p-value = %6.4f
dof = %i 
expected_freq = %s'''  %stats.chi2_contingency(cross_table.iloc[:2, :2]))


'''
期望频次：如果没关系p(x=0,y=0)=p(x=0)*p(y=0)
chisq = 158.4433 
p-value = 0.0000  有关系
dof = 1 
expected_freq = [[1013.24025411  805.75974589]
 [ 915.75974589  728.24025411]]
'''

#%%
#（二）	首先将原始数据拆分为训练和测试数据集，使用训练数据集建立在网时长对流失的逻辑回归，使用测试数据集制作混淆矩阵（阈值为0.5），提供准确性、召回率指标，提供ROC曲线和AUC。
# X:在网时长（duration）连续变量  Y：流失 分类变量
data0.plot(x='duration', y='churn', kind='scatter')
#%%
#采样训练集，测试集
train = data0.sample(frac=0.7, random_state=1234).copy()
test = data0[~ data0.index.isin(train.index)].copy()
print(' 训练集样本量: %i \n 测试集样本量: %i' %(len(train), len(test)))
'''
训练集样本量: 2424 
测试集样本量: 1039
'''
#%%
#逻辑回归 y~x,数据集,family(分布族)：伯努利（01分布），连接函数：logit
lg = smf.glm('churn ~ duration', data=train, 
             family=sm.families.Binomial(sm.families.links.logit)).fit()
lg.summary()

# 预测

# In[19]:

train['proba'] = lg.predict(train)
test['proba'] = lg.predict(test)
test['proba'].head(10)

#%%
#模型评估
#1、定阈值
test['prediction'] = (test['proba'] > 0.5).astype('int')#>0.5认为会流失
#2、混淆矩阵
pd.crosstab(test.churn, test.prediction, margins=True)
#3、准确率
acc = sum(test['prediction'] == test['churn']) /np.float(len(test))
print('The accurancy is %.2f' %acc)#The accurancy is 0.77
#%%
## ix方法已经被移除了用loc替换
for i in np.arange(0.1, 0.9, 0.05):
    prediction = (test['proba'] > i).astype('int')
    confusion_matrix = pd.crosstab(prediction,test.churn,
                                   margins = True)
    precision = confusion_matrix.loc[0, 0] /confusion_matrix.loc['All', 0]#命中率（A/A+C）
    recall = confusion_matrix.loc[0, 0] / confusion_matrix.loc[0, 'All']#召回率(覆盖率)A/A+B
    Specificity = confusion_matrix.loc[1, 1] /confusion_matrix.loc[1,'All']#特异度(负例覆盖率)D/C+D
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('threshold: %s, precision: %.2f, recall:%.2f ,Specificity:%.2f , f1_score:%.2f'%(i, precision, recall, Specificity,f1_score))
    #%%
    
# - 绘制ROC曲线

import sklearn.metrics as metrics

fpr_test, tpr_test, th_test = metrics.roc_curve(test.churn, test.proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train.churn, train.proba)

plt.figure(figsize=[3, 3])
plt.plot(fpr_test, tpr_test, 'b--')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()


# In[28]:
#AUC
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))#AUC = 0.8790 蛮高的e
#%%
#（三）	使用向前逐步法从其它备选变量中选择变量，构建基于AIC的最优模型，绘制ROC曲线，同时检验模型的膨胀系数。
# AIC越小，模型越好
def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')# 正无穷
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response,' + '.join(selected + [candidate]))
            aic = smf.glm(
                formula=formula, data=data, 
                family=sm.families.Binomial(sm.families.links.logit)
            ).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate=aic_with_candidates.pop()
        if current_score > best_new_score: 
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print ('aic is {},continuing!'.format(current_score))
        else:        
            print ('forward selection over!')
            break
            
    formula = "{} ~ {} ".format(response,' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = smf.glm(
        formula=formula, data=data, 
        family=sm.families.Binomial(sm.families.links.logit)
    ).fit()
    return(model)
#%%
candidates = ['churn','duration','AGE','edu_class','posTrend','negTrend','nrProm','prom','curPlan','avgplan','planChange','incomeCode','feton','peakMinAv','peakMinDiff','call_10086']
data_for_select = train[candidates]
lg_m1 = forward_select(data=data_for_select, response='churn')
lg_m1.summary()
#%%
#检查膨胀系数
def vif(df, col_i):
    from statsmodels.formula.api import ols
    
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)
#%%
exog = train[candidates].drop(['churn'], axis=1)
for i in exog.columns:
    print(i, '\t', vif(df=exog, col_i=i))
#posTrend,negTrend;curPlan,avgplan;nrProm,prom有明显的共线性问题,剔除其中两个后重新建模.剔除，curplan,posTrend,prom
'''
duration         1.1649188214231456
AGE      1.0604059554415832
edu_class        1.0919374065809322
posTrend         10.87998721692619
negTrend         10.799093191856452
nrProm   10.594010492273254
prom     10.642709479318954
curPlan          228.06562536008082
avgplan          224.90961280080845
planChange       3.8781006983584954
incomeCode       1.0331700826612906
feton    1.032150079222362
peakMinAv        1.2373194257375613
peakMinDiff      1.758824465225615
call_10086       1.027704090678157
'''
#%%
candidates = ['churn','duration','AGE','edu_class','negTrend','nrProm','avgplan','planChange','incomeCode','feton','peakMinAv','peakMinDiff','call_10086']
data_for_select = train[candidates]
lg_m2 = forward_select(data=data_for_select, response='churn')
lg_m2.summary()
#%%
# - 绘制ROC曲线
train['proba2'] = lg_m2.predict(train)
test['proba2'] = lg_m2.predict(test)

import sklearn.metrics as metrics

fpr_test2, tpr_test2, th_test2 = metrics.roc_curve(test.churn, test.proba2)
fpr_train2, tpr_train2, th_train2 = metrics.roc_curve(train.churn, train.proba2)

plt.figure(figsize=[3, 3])
plt.plot(fpr_test2, tpr_test2, 'b--')
plt.plot(fpr_train2, tpr_train2, 'r-')
plt.title('ROC curve')
plt.show()


# In[28]:
#AUC
print('AUC = %.4f' %metrics.auc(fpr_test2, tpr_test2))#AUC = 0.9061 有提升

#%%

#4）使用岭回归和Laso算法重建第三步中的模型，使用交叉验证法确定惩罚参数(C值)。并比较步骤四中Laso算法得到的模型和第三步得到的模型的差异
    






























