# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:18:32 2020

@author: Away
"""


# coding: utf-8

# # 第6讲 统计推断基础
# - 数据说明：本数据是二手房房价信息
# -变量说明如下：
# dist-所在区--多分类
# roomnum-室的数量--多分类
# halls-厅的数量--多分类
# AREA-房屋面积--连续
# floor-楼层--多分类
# subway-是否临近地铁--二分类
# school-是否学区房--二分类
# price-平米单价--Y
# -分析思路：
# 在对房价的影响因素进行模型研究之前，首先对各变量进行描述性分析，以初步判断房价的影响因素，进而建立房价预测模型

#%%
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import statsmodels.api as sm
from numpy import corrcoef,array
#from IPython.display import HTML, display
from statsmodels.formula.api import ols
import os
os.chdir(r"C:\Users\Away\Desktop\笔记\数据分析")
# In[1]:
import pandas as pd
mdata = pd.read_csv(r'sndHsPr.csv')
mdata.head()
#%%
#1、描述性统计数据预处理阶段
describe=mdata.describe()#价格基本情况
# count     16210.000000
# mean      61151.810919
# std       22293.358147
# min       18348.000000
# 25%       42812.250000
# 50%       57473.000000
# 75%       76099.750000
# max      149871.000000
print(mdata.shape[0])#数据量：16210  这里为什么要统计数据量呢?因为做检验时数据量不能过大，一般不超过5000，否则p值会失效
#%%
## 1、描述性统计
#数据简单处理,重点关注价格日期等字段
#查看各个字段基本情况
data0 = mdata
describe=data0.describe(include="all").T
#print(data0.dtypes)
#价格转换为万元
data0.price = data0.price/10000
#type_dict = {'借':'out','贷':'income'}
#card_t3['type1'] = card_t3.t_type.map(type_dict)
#将区域名转换为中文
dist_dict = {
        'chaoyang' : "朝阳",
        'dongcheng' : "东城",
        'fengtai' :  "丰台",
        'haidian' : "海淀",
        'shijingshan' : "石景山",
        'xicheng': "西城"
        }

data0['dist']=data0.dist.map(dist_dict)
#%%
matplotlib.rcParams['axes.unicode_minus']=False#解决保存图像时负号'-'显示为方块的问题
plt.rcParams['font.sans-serif'] = ['SimHei']#指定默认字体 ，解决不能显示中文字体的问题#
#%%
## 1.1因变量y
#因变量图形（主要看看形态）
data0.price.plot(kind='hist',bins=20,color='lightblue')#右偏严重
plt.xlabel("单位面积房价（万元/平方米）")
plt.ylabel("频数")
print(data0.price.agg(['mean','median','std']))  #查看price的均值、中位数和标准差等更多信息
print(data0.price.quantile([0.25,0.5,0.75]))
pd.concat([(data0[data0.price==min(data0.price)]),(data0[data0.price==max(data0.price)])])#查看房价最高和最低的两条观测
#%%
##1.2、自变量描述（除了房屋面积都是分类变量）
for i in range(7):
    if i!=3:
        print(data0.columns.values[i],":")
        print(data0[data0.columns.values[i]].agg(['value_counts']).T)
        print('==================================')
    else:
        continue
print('AREA:')
print(data0.AREA.agg(['min','mean','max','median','std']).T)
#%%
#1.2.1 dist 区域 多分类变量
#频次统计
data0.dist.value_counts().plot(kind='bar')
#%%
#不同城区的房价
data0.price.groupby(data0.dist).mean().sort_values(ascending=True).plot(kind='barh')#能看出来和城区还是有关系的
#%%
data1 = data0[['dist','price']]
sns.boxplot(x='dist',y='price',data=data1)
#dat1.boxplot(by='dist',patch_artist=True)
plt.ylabel("单位面积房价(万元/平方米)")
plt.xlabel("城区")
plt.title("城区对房价的分组箱线图")#城区还是有明显影响的
#%%
#1.2.2 roomnum 卧室数-roomnum 多分类
data2=data0[['roomnum','price']]
data2.price.groupby(data2.roomnum).mean().plot(kind='bar')
#data2.boxplot(by='roomnum',patch_artist=True)# patch_artist	是否填充箱体的颜色；
plt.figure()
sns.boxplot(x='roomnum',y='price',data=data2)#没啥关系

#%%
#1.2.3厅数 halls
data3=data0[['halls','price']]
data3.price.groupby(data2.roomnum).mean().plot(kind='bar')
plt.figure()
sns.boxplot(x='halls',y='price',data=data3)#厅数越多价格越低，但是影响不大
#%%
#1.2.4楼层 floor
data4=data0[['floor','price']]
sns.boxplot(x='floor',y='price',data=data4)
plt.xlabel("楼层")#差异不大
#%%
#1.2.5 学校 分类变量
data5=data0[['school','price']]
sns.boxplot(x='school',y='price',data=data5)
plt.xlabel("学校")#明显有关
#%%
#1.2.6 地铁 分类变量
data6=data0[['subway','price']]
sns.boxplot(x='subway',y='price',data=data6)
plt.xlabel("地铁")#明显有关
#%%
# 画地铁和学区的标准化堆叠图(暂时不知道有啥用) 解答：后面建模时考虑自变量交互项有用，差异还是挺大的，说明有一定相关性（直觉来说学校周围应该交通便利）
from stack2dim import *
stack2dim(data0, i="subway", j="school")
#%%
# 1.2.7 AREA 连续变量
dataA=data0[['AREA','price']]
plt.scatter(dataA.AREA,dataA.price,marker='.')#发散型，右偏-对Y取对数
# area,price都是连续变量，进行相关分析
xg1=dataA[['price','AREA']].corr(method='pearson')#返回一个dataframe#  r=-0.074 中度负相关。
# 注意：只有在两两变量之间r<0.3时才考虑不要。建模和逻辑回归时，小于0.3也要保留

#%%
#下面这种写法默认用pearson相关系数，并返回相关系数矩阵
data1=array(dataA['price'])
data2=array(dataA['AREA'])
datB=array([data1,data2])
xg2=corrcoef(datB)#Array
#%%
#对Y取对数
dataA['price_ln'] = np.log(dataA['price'])
plt.figure(figsize=(8,8))
plt.scatter(dataA.AREA,dataA.price_ln,marker='.')
plt.ylabel("单位面积房价（取对数后）")
plt.xlabel("面积（平方米）")#小户型贵
#求AREA_ln和price_ln的相关系数
xg3=dataA[['price_ln','AREA']].corr(method='pearson')#|r|=0.058

#%%
##？？有把房屋面积也取对数的操作，为何？猜测因为发现仅仅将y取对数之后相关性降低了而且图形还是右偏的，考虑对x取对数
# In[58]:
#房屋面积和单位面积房价（取对数后）的散点图
dataA['price_ln'] = np.log(dataA['price'])  #对price取对数
dataA['AREA_ln'] = np.log(dataA['AREA'])  #对price取对数
plt.figure(figsize=(8,8))
plt.scatter(dataA.AREA_ln,dataA.price_ln,marker='.')
plt.ylabel("单位面积房价（取对数后）")
plt.xlabel("面积（平方米）")
#求AREA_ln和price_ln的相关系数矩阵
data1=array(dataA['price_ln'])
data2=array(dataA['AREA_ln'])
datB=array([data1,data2])
corrcoef(datB)# 0.09,高度相关，且x,y两个方向都是正态分布
##############################################################################
## 2 建模
# 目前结论：明显影响:区，地铁，学区 不明显：客厅，楼层 基本不影响：卧室（roomnum）
# 描述性统计结束，提出我们的假设（学区房贵。。），后面就开始假设检验分析，验证结论
#%%
# 做假设检验之前先进行抽样确定阈值。按区分层，每层抽400个，总样本2400.阈值5%
def get_sample(df, sampling="simple_random", k=1, stratified_col=None):
    """
    对输入的 dataframe 进行抽样的函数
    参数:
        - df: 输入的数据框 pandas.dataframe 对象
        - sampling:抽样方法 str
            可选值有 ["simple_random", "stratified", "systematic"]
            按顺序分别为: 简单随机抽样、分层抽样、系统抽样
        - k: 抽样个数或抽样比例 int or float
            (int, 则必须大于0; float, 则必须在区间(0,1)中)
            如果 0 < k < 1 , 则 k 表示抽样对于总体的比例
            如果 k >= 1 , 则 k 表示抽样的个数；当为分层抽样时，代表每层的样本量
        - stratified_col: 需要分层的列名的列表 list
            只有在分层抽样时才生效
    返回值:
        pandas.dataframe 对象, 抽样结果
    """
    import random
    import pandas as pd
    from functools import reduce
    import numpy as np
    import math
    
    len_df = len(df)
    if k <= 0:
        raise AssertionError("k不能为负数")
    elif k >= 1:
        assert isinstance(k, int), "选择抽样个数时, k必须为正整数"
        sample_by_n=True
        if sampling is "stratified":
            alln=k*df.groupby(by=stratified_col)[stratified_col[0]].count().count() # 有问题的
            #alln=k*df[stratified_col].value_counts().count() 
            if alln >= len_df:
                raise AssertionError("请确认k乘以层数不能超过总样本量")
    else:
        sample_by_n=False
        if sampling in ("simple_random", "systematic"):
            k = math.ceil(len_df * k)
        
    #print(k)
    if sampling is "simple_random":
        print("使用简单随机抽样")
        idx = random.sample(range(len_df), k)
        res_df = df.iloc[idx,:].copy()
        return res_df
    elif sampling is "systematic":
        print("使用系统抽样")
        step = len_df // k+1          #step=len_df//k-1
        start = 0                  #start=0
        idx = range(len_df)[start::step]  #idx=range(len_df+1)[start::step]
        res_df = df.iloc[idx,:].copy()
        #print("k=%d,step=%d,idx=%d"%(k,step,len(idx)))
        return res_df
    elif sampling is "stratified":
        assert stratified_col is not None, "请传入包含需要分层的列名的列表"
        assert all(np.in1d(stratified_col, df.columns)), "请检查输入的列名"
        
        grouped = df.groupby(by=stratified_col)[stratified_col[0]].count()
        if sample_by_n==True:
            group_k = grouped.map(lambda x:k)
        else:
            group_k = grouped.map(lambda x: math.ceil(x * k))
        
        res_df = df.head(0)
        for df_idx in group_k.index:
            df1=df
            if len(stratified_col)==1:
                df1=df1[df1[stratified_col[0]]==df_idx]
            else:
                for i in range(len(df_idx)):
                    df1=df1[df1[stratified_col[i]]==df_idx[i]]
            idx = random.sample(range(len(df1)), group_k[df_idx])
            group_df = df1.iloc[idx,:].copy()
            res_df = res_df.append(group_df)
        return res_df
    else:
        raise AssertionError("sampling is illegal")

#%%
# 根据城区分层采样
data_new = get_sample(data0,sampling='stratified',k=400,stratified_col=['dist'])
#2400个样本，阈值确定原则如下：
"""大致原则如下（自然科学取值偏小、社会科学取值偏大）：
n<100 alfa取值[0.05,0.2]之间
100<n<500 alfa取值[0.01,0.1]之间
500<n<3000 alfa取值[0.001,0.05]之间
"""
 #%%
#逐个检查分类变量的解释力度，方差分析
import statsmodels.api as sm
from statsmodels.formula.api import ols
print("dist的P值为:%.4f" %sm.stats.anova_lm(ols('price ~ C(dist)',data=data_new).fit())._values[0][4])
print("roomnum的P值为:%.4f" %sm.stats.anova_lm(ols('price ~ C(roomnum)',data=data_new).fit())._values[0][4])#明显高于0.001->不显著->独立
print("halls的P值为:%.4f" %sm.stats.anova_lm(ols('price ~ C(halls)',data=data_new).fit())._values[0][4])#高于0.001->边际显著->暂时考虑
print("floor的P值为:%.4f" %sm.stats.anova_lm(ols('price ~ C(floor)',data=data_new).fit())._values[0][4])#高于0.001->边际显著->暂时考虑
print("subway的P值为:%.4f" %sm.stats.anova_lm(ols('price ~ C(subway)',data=data_new).fit())._values[0][4])
print("school的P值为:%.4f" %sm.stats.anova_lm(ols('price ~ C(school)',data=data_new).fit())._values[0][4])
'''
dist的P值为:0.0000
roomnum的P值为:0.1014 #去掉roomnum，其他保留
halls的P值为:0.0002
floor的P值为:0.0013
subway的P值为:0.0000
school的P值为:0.0000
'''
#%%
# 厅数和楼层的影响不太显著。对于厅数可以做因子化处理，变成二分变量（'有厅','无厅'）
data_new['hall_new'] = data_new.halls
data_new.hall_new[data_new.hall_new>0]='有厅'
data_new.hall_new[data_new.hall_new==0]='无厅'
#%%
# 3 线性回归模型
lm1 = ols("price ~ C(dist)+school+subway+C(floor)+C(hall_new)++AREA", data=data_new).fit()#这里也可以自己设计基准创建哑变量
lm1_summary = lm1.summary()
lm1_summary  #回归结果展示1 R方=0.599 开始忘了加hall_new 发现加了对模型没啥改进作用
#%%
#初始线性回归模型残差分析
data_new['pred1']=lm1.predict(data_new)#模型预测结果
data_new['resid1']=lm1.resid#取残差
data_new.plot('pred1','resid1',kind='scatter')  #模型诊断图，存在异方差现象，对因变量取对数
#预测值增加，残差呈现喇叭口发散状况。考虑取对数
#%%
# 3.1 改进取对数后再次建模
data_new['price_ln'] = np.log(data_new['price'])
data_new['AREA_ln'] = np.log(data_new['AREA'])
lm2 = ols("price_ln ~ C(dist)+school+subway+C(floor)+AREA_ln", data=data_new).fit()
lm2_summary = lm2.summary()
lm2_summary  #回归结果展示2 R方=0.614 有一丢丢提升
#%%
#残差分析
data_new['pred2']=lm2.predict(data_new)#模型预测结果
data_new['resid2']=lm2.resid#取残差
data_new.plot('pred2','resid2',kind='scatter')  #模型诊断图，存在异方差现象，对因变量取对数
#%%
################################################
# 加上交互项建模，一般考虑比较显著的自变量交互
'''
交互项的理解：在不同的城区是否是学区房对于房价影响是不同的。
'''
#round(x,2) 四舍五入，保留两位小数

# print('石景山非学区房\t',round(data0[(data0['dist']=='石景山')&(data0['school']==0)]['price'].mean(),2),'万元/平方米\t',
#      '石景山学区房\t',round(data0[(data0['dist']=='石景山')&(data0['school']==1)]['price'].mean(),2),'万元/平方米')
# print('-------------------------------------------------------------------------')
#%%
schools=['石景山','丰台','朝阳','东城','海淀','西城']
for i in schools:
    print(i+'非学区房\t',round(data_new[(data_new['dist']==i)&(data_new['school']==0)]['price'].mean(),2),'万元/平方米\t',i+'学区房\t',round(data_new[(data_new['dist']==i)&(data_new['school']==1)]['price'].mean(),2),'万元/平方米')
'''
石景山非学区房  4.08 万元/平方米     石景山学区房  3.19 万元/平方米
丰台非学区房   4.24 万元/平方米     丰台学区房   4.69 万元/平方米
朝阳非学区房   5.06 万元/平方米     朝阳学区房   5.7 万元/平方米
东城非学区房   6.5 万元/平方米      东城学区房   7.84 万元/平方米
海淀非学区房   5.96 万元/平方米     海淀学区房   7.59 万元/平方米
西城非学区房   7.77 万元/平方米     西城学区房   9.09 万元/平方米

就石景山的学区房比非学区房便宜
'''
#%%
#画图看看差异
df = pd.DataFrame()
dist = ['石景山','丰台','朝阳','东城','海淀','西城']
Noschool=[]
school=[]
for i in dist:
    Noschool.append(data_new[(data_new['dist']==i)&(data_new['school']==0)]['price'].mean())
    school.append(data_new[(data_new['dist']==i)&(data_new['school']==1)]['price'].mean())
df['dist']=pd.Series(dist)
df['Noschool']=pd.Series(Noschool)
df['school']=pd.Series(school)
df

df1=df['Noschool'].T.values
df2=df['school'].T.values
plt.figure(figsize=(10,6))
x1=range(0,len(df))
x2=[i+0.3 for i in x1]#往右偏移一点
plt.bar(x1,df1,color='b',width=0.3,alpha=0.6,label='非学区房')
plt.bar(x2,df2,color='r',width=0.3,alpha=0.6,label='学区房')
plt.xlabel('城区')
plt.ylabel('单位面积价格')
plt.title('分城区、是否学区的房屋价格')
plt.legend(loc='upper left')
plt.xticks(range(0,6),dist)#X刻度
plt.show()
#%%
#感觉上面画法有点繁琐
df = pd.DataFrame()
dist = ['石景山','丰台','朝阳','东城','海淀','西城']
Noschool=[]
school=[]
for i in dist:
    Noschool.append(data_new[(data_new['dist']==i)&(data_new['school']==0)]['price'].mean())
    school.append(data_new[(data_new['dist']==i)&(data_new['school']==1)]['price'].mean())
df['dist']=pd.Series(dist)
df['Noschool']=pd.Series(Noschool)
df['school']=pd.Series(school)
df.plot(kind='bar')
plt.xlabel('城区')
plt.ylabel('单位面积价格')
plt.title('分城区、是否学区的房屋价格')
plt.legend(loc='upper left')
plt.xticks(range(0,6),dist)#X刻度

#%%

# #%%
# school=['石景山','丰台','朝阳','东城','海淀','西城']
# for i in school:
#     data0[data0.dist==i][['school','price']].boxplot(by='school',patch_artist=True)
#     plt.xlabel(i+'学区房')
#%%
# 3.2 加上交互项对数模型
lm3 = ols("price_ln ~ C(dist)*school+subway+C(floor)+AREA_ln", data=data_new).fit()
lm3_summary = lm3.summary()
lm3_summary  #回归结果展示 R方=0.618
#%%
#试试其他交互项
lm4 = ols("price_ln ~ C(dist)+school*subway+C(floor)+AREA_ln", data=data_new).fit()
lm4_summary = lm4.summary()
lm4_summary #0.615
#%%
lm5 = ols("price_ln ~ C(dist)*subway+school+C(floor)+AREA_ln", data=data_new).fit()
lm5_summary = lm5.summary()
lm5_summary #0.621 奇了怪了，考虑地区和地铁交互作用得到的结果是最好的
#%%
lm6 = ols("price_ln ~ C(dist)*subway*school+C(floor)+AREA_ln", data=data_new).fit()
lm6_summary = lm6.summary()
lm6_summary # 0.621
#%%##################################################################################
#选一个好的模型来做预测
x_new1=data_new.head(1)
x_new1
#%%
x_new1['dist']='东城'
x_new1['roomnum']=2
x_new1['AREA_ln']=np.log(80)
x_new1['subway']=1
x_new1['school']=1
x_new1['floor']='middle'
x_new1['hall_new']="有厅"

#%%
#预测值
print("单位面积房价：",round(math.exp(lm5.predict(x_new1)),2),"万元/平方米")
print("总价：",round(math.exp(lm5.predict(x_new1))*70,2),"万元")
                   









































