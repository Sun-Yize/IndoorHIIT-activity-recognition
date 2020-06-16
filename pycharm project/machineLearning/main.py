from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import joblib


print('Generating Model')
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

# 训练初始随机森林模型，树的数量为100棵
rfc = RandomForestClassifier(n_estimators=100)
model = rfc.fit(inputs, outputs)

# 根据已经生成的模型，对特征进行挑选
sfm = SelectFromModel(model, threshold=0.008, prefit=True)
# 将挑选过特征写入文档
feature_idx = sfm.get_support()
selectList = np.zeros(len(feature_idx))
for i in range(len(feature_idx)):
    if str(feature_idx[i]) == 'True':
        selectList[i] = 1
    else:
        selectList[i] = 0
np.savetxt('./result/data1.txt', selectList, fmt='%f', delimiter=',')
# 根据挑选的特征，重新生成输入
inputs = sfm.transform(inputs)

# 根据新的特征训练模型
model = rfc.fit(inputs, outputs)
# 进行十折交叉验证
recall_score = cross_val_score(model, inputs, outputs, cv=10)
# 保存训练好的模型
joblib.dump(model, '../web/model/rf2.model')

# 输出测试结果
print('Accuracy = ', recall_score.mean())
print('Finish')
