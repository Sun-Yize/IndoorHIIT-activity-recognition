from flask import Flask, request
import sys
import numpy as np
import joblib
import feature
from scipy import signal
from detecta import detect_peaks
app = Flask(__name__)

# 分别读取ios和安卓的模型
weight1 = np.loadtxt('./model/data1.txt')
weight2 = np.loadtxt('./model/data2.txt')
RF1 = joblib.load('model/rf1.model')
RF2 = joblib.load('model/rf2.model')
# 特征提取文件
feature_class = feature.getFeature()
name = ['accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz']
# 计数和识别使用的滤波器
b1, a1 = signal.butter(2, [0.001, 0.03],  'bandpass')
b2, a2 = signal.butter(2, [0.001, 0.06],  'bandpass')
b3, a3 = signal.butter(8, 0.2, 'lowpass')
# 初始化全局变量
accTotal = hiitTotal = np.array([])
count = 0


# 特征权重函数
def select(feature_list, flag):
    select_list = []
    for i in range(len(globals()['weight'+flag])):
        if globals()['weight'+flag][i] == 1:
            select_list = np.append(select_list, feature_list[i])
    return select_list


# 动作识别函数
@app.route('/', methods=['post'])
def identify():
    global accTotal, count, hiitTotal
    inputs = []
    data = request.get_json()
    # 逐个轴提取特征
    for i in range(6):
        temp = np.array(list(map(float, data[name[i]].split(','))))
        temp = signal.filtfilt(b3, a3, temp)
        fea = feature_class.get_all_feature(temp)
        inputs = np.append(inputs, fea)
    # 检测是否处于静止
    if (inputs[1]-inputs[0]) > 0.05 and inputs[0] < 0.7:
        inputs = select(inputs, str(data['type'])).reshape(1, -1)
        result = globals()['RF'+str(data['type'])].predict(inputs)
        result = result[0]
    else:
        result = 0
    # 将识别结果保存，用于计数
    if count % 2 == 0:
        accTotal = np.append(accTotal, np.array(list(map(float, data['accy'].split(',')))))
    hiitTotal = np.append(hiitTotal, result)
    if count == 0:
        hiitTotal = np.append(hiitTotal, hiitTotal)
    count += 1
    print(str(result))
    return str(result)


# 计数函数
@app.route('/count', methods=['post'])
def countnum():
    global accTotal, count, hiitTotal
    numlist = {'1': 50, '2': 17}
    data = request.get_json()
    actNum = np.array([])
    countTotal = np.zeros(4)
    # 计数滤波
    filtedData = signal.filtfilt(globals()['b'+str(data['type'])], globals()['a'+str(data['type'])], accTotal)
    # 检测波峰
    numpeak = detect_peaks(filtedData)
    # 计算各个动作的个数
    for i1 in range(len(numpeak)):
        actNum = np.append(actNum, hiitTotal[(numpeak[i1] - 1) // numlist[str(data['type'])]])
    for i2 in range(4):
        countTotal[i2] = sum(actNum == i2+1)
    print('count = ', countTotal)
    return str(countTotal)


# 清除函数，重置全局变量
@app.route('/clear', methods=['post'])
def clear():
    global accTotal, count, hiitTotal
    accTotal = hiitTotal = np.array([])
    count = 0
    return '0'


if __name__ == '__main__':
    context = (sys.path[0] + '/Nginx/1_www.inifyy.cn_bundle.crt', sys.path[0] + '/Nginx/2_www.inifyy.cn.key')
    app.run(debug=1, host='172.17.0.3', port=8080, ssl_context=context)
