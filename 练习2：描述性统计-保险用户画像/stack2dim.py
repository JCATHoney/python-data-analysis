# 如遇中文显示问题可加入以下代码
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def stack2dim(raw, i, j, rotation=0, location='upper right'):
    '''
    此函数是为了画两个维度标准化的堆积柱状图
    raw为pandas的DataFrame数据框
    i、j为两个分类变量的变量名称，要求带引号，比如"school"
    rotation：水平标签旋转角度，默认水平方向，如标签过长，可设置一定角度，比如设置rotation = 40
    location：分类标签的位置，如果被主体图形挡住，可更改为'upper left'
    '''
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import math

    data_raw = pd.crosstab(raw[i], raw[j])
    data = data_raw.div(data_raw.sum(1), axis=0)  # 交叉表转换成比率，为得到标准化堆积柱状图

    # 计算x坐标，及bar宽度
    createVar = locals()
    x = [0]  # 每个bar的中心x轴坐标
    width = []  # bar的宽度
    k = 0
    for n in range(len(data)):
        # 根据频数计算每一列bar的宽度
        createVar['width' + str(n)] = list(data_raw.sum(axis=1))[n] / sum(data_raw.sum(axis=1))
        width.append(createVar['width' + str(n)])
        if n == 0:
            continue
        else:
            k += createVar['width' + str(n - 1)] / 2 + createVar['width' + str(n)] / 2 + 0.05
            x.append(k)

    # 以下是通过频率交叉表矩阵生成一串对应堆积图每一块位置数据的数组，再把数组转化为矩阵
    y_mat = []
    n = 0
    y_level = len(data.columns)
    for p in range(data.shape[0]):
        for q in range(data.shape[1]):
            n += 1
            y_mat.append(data.iloc[p, q])
            if n == data.shape[0] * data.shape[1]:
                break
            elif n % y_level != 0:
                y_mat.extend([0] * (len(data) - 1))
            elif n % y_level == 0:
                y_mat.extend([0] * len(data))

    y_mat = np.array(y_mat).reshape(-1, len(data))
    y_mat = pd.DataFrame(y_mat)  # bar图中的y变量矩阵，每一行是一个y变量

    # 通过x，y_mat中的每一行y，依次绘制每一块堆积图中的每一块图

    from matplotlib import cm
    cm_subsection = [level for level in range(y_level)]
    colors = [cm.Pastel1(color) for color in cm_subsection]

    bottom = [0] * y_mat.shape[1]
    createVar = locals()
    for row in range(len(y_mat)):
        createVar['a' + str(row)] = y_mat.iloc[row, :]
        color = colors[row % y_level]

        if row % y_level == 0:
            bottom = bottom = [0] * y_mat.shape[1]
            if math.floor(row / y_level) == 0:
                label = data.columns.name + ': ' + str(data.columns[row])
                plt.bar(x, createVar['a' + str(row)],
                        width=width[math.floor(row / y_level)], label=label, color=color)
            else:
                plt.bar(x, createVar['a' + str(row)],
                        width=width[math.floor(row / y_level)], color=color)
        else:
            if math.floor(row / y_level) == 0:
                label = data.columns.name + ': ' + str(data.columns[row])
                plt.bar(x, createVar['a' + str(row)], bottom=bottom,
                        width=width[math.floor(row / y_level)], label=label, color=color)
            else:
                plt.bar(x, createVar['a' + str(row)], bottom=bottom,
                        width=width[math.floor(row / y_level)], color=color)

        bottom += createVar['a' + str(row)]

    plt.title(j + ' vs ' + i)
    group_labels = [str(name) for name in data.index]
    plt.xticks(x, group_labels, rotation=rotation)
    plt.ylabel(j)
    plt.legend(shadow=True, loc=location)
    plt.show()