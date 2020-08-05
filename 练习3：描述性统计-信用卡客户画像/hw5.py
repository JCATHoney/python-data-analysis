# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 00:49:21 2020

@author: Away
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import time

#%%
os.chdir(r'C:\Users\Away\Desktop\笔记\数据分析\HW5')
#%%
#先读取三张表
card= pd.read_csv('card.csv',encoding='gbk')
disp= pd.read_csv('disp.csv',encoding='gbk')#权限分配表
clients= pd.read_csv('clients.csv',encoding='gbk')

#%%
import sqlite3 # sqlite3相当于轻量版，更多功能可使用SQLAlchemy
con = sqlite3.connect(':memory:') # 数据库连接
# sale.to_sql('sale', con) # 将DataFrame注册成可用sql查询的表
# newTable = pd.read_sql_query("select year, market, sale, profit from sale", con) # 也可使用read_sql
# newTable.head()
card.to_sql('card', con)
disp.to_sql('disp', con)
clients.to_sql('clients', con)

#%%

sql = '''select a.*,c.sex,c.birth_date
      from card as a left join disp as b on a.disp_id=b.disp_id
      left join clients as c on b.client_id=c.client_id
      where b.type='所有者'
      '''
card_t=pd.read_sql(sql, con)

#%%
#不同类型卡的持卡人的性别对比，如下图所示 两个分类变量,用标准化堆叠图
#pd.crosstab(card_t.type,card_t.sex).plot(kind = 'bar',stacked=True)
from stack2dim import *#stackdim是自定义的，直接用.可以画出标准化的堆叠图
stack2dim(card_t, i="type", j="sex")
#%%2、不同类型卡的持卡人在办卡时的平均年龄对比
#首先将字符型装换为日期
card_t['age']=(pd.to_datetime(card_t['issued'])-pd.to_datetime(card_t['birth_date'])).map(lambda x:x.days/365)
#持卡类型，分类变量；平均年龄，连续变量。用盒须图
sns.boxplot(x = 'type', y = 'age', data = card_t)
card_t.to_sql('card_t', con)
#%%
#问题3.4涉及取数窗口
#3、不同类型卡的持卡人在办卡前一年内的平均帐户余额对比
# 需要字段：卡的类型，持卡人，账户余额 涉及到三个表：card,disp,trans
trans=pd.read_csv(r"trans.csv",encoding="gbk")
trans.to_sql('trans', con)
#%%
#每张卡的交易记录，
# 卡id,发卡时间，卡类型，借贷类型，交易量，账户余额，交易时间
car_sql='''
select a.card_id,a.issued,a.type,c.type as t_type,c.amount,c.balance,c.date as t_date
  from card as a
  left join disp as b on a.disp_id=b.disp_id
  left join trans as c on b.account_id=c.account_id
  where b.type="所有者"
  order by a.card_id,c.date
'''
card_t2=pd.read_sql(car_sql, con)
#%%
#字符格式转换成日期
card_t2['issued']=pd.to_datetime(card_t2['issued'])
card_t2['t_date']=pd.to_datetime(card_t2['t_date'])

# In[9]:
# ## 将对账户余额进行清洗
#部分金额有逗号分隔
import datetime
card_t2['balance2'] = card_t2['balance'].map(
    lambda x: int(''.join(x[1:].split(','))))
card_t2['amount2'] = card_t2['amount'].map(
    lambda x: int(''.join(x[1:].split(','))))#join方法：Example: '.'.join(['ab', 'pq', 'rs']) -> 'ab.pq.rs'用‘，’f分隔的两部分连起来
card_t2.head()
#%%
#取前一年的数据
#发卡时间之前1年的交易记录
card_t3 = card_t2[card_t2.issued>card_t2.t_date][
    card_t2.issued<card_t2.t_date+datetime.timedelta(days=365)]

#card_t3["avg_balance"] = card_t3.groupby('card_id')['balance2'].mean()
card_t4=card_t3.groupby(['type','card_id'])['balance2'].agg([('avg_balance','mean')])
card_t4.to_sql('card_t4', con)
#%%
card_t5=card_t4.reset_index()
#card_t5=pd.read_sql('select * from card_t4', con)
sns.boxplot(x = 'type', y = 'avg_balance', data = card_t5)
#%%
#4、不同类型卡的持卡人在办卡前一年内的平均收入对比
type_dict = {'借':'out','贷':'income'}
card_t3['type1'] = card_t3.t_type.map(type_dict)
# card_t6= card_t3.groupby(['type','card_id','type1'])[['amount2']].sum()#这里不是求总收入吗?改动如下结果看起来没问题
card_t6= card_t3.groupby(['type','card_id','type1'])['amount2'].agg([('avg_ammount','mean')])
card_t6.head()
card_t6.to_sql('card_t6', con)
#%%
card_t7=card_t6.reset_index()
#card_t7=pd.read_sql('select * from card_t6', con)
card_t7.to_sql('card_t7', con)
card_t8=pd.read_sql('select * from card_t7 where type1="income"', con)
# In[13]:
sns.boxplot(x = 'type', y = 'avg_ammount', data = card_t8)#

#%%
card_t9=pd.read_sql('select * from card_t7 where type1="out"', con)
# In[13]:
sns.boxplot(x = 'type', y = 'avg_ammount', data = card_t9)
#%%

