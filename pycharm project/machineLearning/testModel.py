import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

# 选取随机数种子
np.random.seed(seed=10)
# 读取数据
inputs = pd.read_csv('../data/featuredata/total.csv', header=None)
outputs = pd.read_csv('../data/processdata/tag.csv', header=None)
# 将数据顺序打乱
train = np.array(pd.concat([inputs, outputs], axis=1))
np.random.shuffle(train)

# 分成输入和结果
inputs = pd.DataFrame(train).iloc[:, :-1]
outputs = pd.DataFrame(train).iloc[:, -1]
# 加载模型和特征权重
rf = joblib.load('../web/model/rf2.model')
weight = np.loadtxt('./result/data1.txt')


# 特征选择函数
def select(feature):
    select_list = []
    for i in range(len(weight)):
        if weight[i] == 1:
            if i == 0:
                select_list = feature[:, i]
            else:
                select_list = np.vstack((select_list, feature[:, i]))
    return select_list


# 规范化输入和结果
inputs = select(np.array(inputs)).transpose()
outputs = np.array(outputs).ravel()
# 进行十折交叉验证
recall_score = cross_val_score(rf, inputs, outputs, cv=10)
# 输出正确率
print('Accuracy = ', recall_score.mean())

