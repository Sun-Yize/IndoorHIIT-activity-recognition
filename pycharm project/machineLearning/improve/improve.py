from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


# 选取随机数种子
np.random.seed(seed=10)
# 读取数据
inputs = pd.read_csv('../../data/featuredata/total.csv', header=None)
outputs = pd.read_csv('../../data/processdata/tag.csv', header=None)
# 将数据顺序打乱
train = np.array(pd.concat([inputs, outputs], axis=1))
np.random.shuffle(train)
# 分成输入和结果
inputs = pd.DataFrame(train).iloc[:, :-1]
outputs = pd.DataFrame(train).iloc[:, -1]


# 将数据进行标准化
inputs = StandardScaler().fit_transform(inputs)
# 生成随机森林模型
rfc = RandomForestClassifier(n_estimators=100)
model = rfc.fit(inputs, outputs)
# 进行十折交叉验证
recall_score1 = cross_val_score(model, inputs, outputs, cv=10)
print('recall_score1 = ', recall_score1.mean())


# 根据已经生成的模型，对特征进行挑选
sfm = SelectFromModel(model, threshold=0.008, prefit=True)
# 根据挑选的特征，重新生成输入
inputs = sfm.transform(inputs)
# 进行十折交叉验证
recall_score2 = cross_val_score(rfc, inputs, outputs, cv=10)
print('recall_score2 = ', recall_score2.mean())
