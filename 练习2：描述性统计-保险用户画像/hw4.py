# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 23:26:39 2020

@author: Away
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
#%%
os.chdir(r'C:\Users\Away\Desktop\笔记\数据分析\HW4')
data = pd.read_csv("auto_ins.csv",encoding='gbk')

#%% 1、首先对loss重新编码为1/0，有数值为1，命名为loss_flag
data['loss_flag']=data.Loss.map(lambda x:1 if x>0 else 0)
#%% 2、对loss_flag分布情况进行描述分析

#分析:单一分类变量，计算出频次，用简单条形图表现即可
loss_flag = data['loss_flag']
loss_flag.value_counts()
#loss_flag.value_counts().plot(kind='bar')
loss_flag.value_counts().plot(kind='pie',autopct='%.2f%%')#接近30%的人有出险记录
#%%% 分析是否出险和年龄、驾龄、性别、婚姻状态等变量之间的关系(提示:使用分类盒须图,堆叠柱形图)
#对于分类变量性别，婚姻状况，用堆叠图。连续变量，年龄，驾龄，用盒须图
#这里用到了子图
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,2,1)
ax2 = fig1.add_subplot(1,2,2)

#出险与年龄堆叠图
pd.crosstab(data.Gender,data.loss_flag).plot(kind='bar',stacked=True,ax=ax1)
#驾龄
pd.crosstab(data.Marital,data.loss_flag).plot(kind='bar',stacked=True,ax=ax2)
fig2 = plt.figure()
ax3 = fig2.add_subplot(1,2,1)
ax4 = fig2.add_subplot(1,2,2)
#年龄
sns.boxplot(x = 'loss_flag', y = 'Age', data = data,ax=ax3)#跟年龄没啥关系
#驾龄
sns.boxplot(x = 'loss_flag', y = 'exp', data = data,ax=ax4)#出险的人驾龄比较短（选择驾龄为一个变量）

#%%
from stack2dim import *#stack2dim是自定义的，直接用.可以画出标准化的堆叠图
stack2dim(data, i="Gender", j="loss_flag")#女性略高（选择性别为一个变量）
stack2dim(data, i="Marital", j="loss_flag")#未婚的略高（虽然未婚的数据量较小）

#%% 接下来就是根据选择出来的变量建模
















