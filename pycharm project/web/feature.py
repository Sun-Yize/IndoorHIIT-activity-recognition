import numpy as np
from scipy import stats
from scipy.fftpack import fft


class getFeature():
    def get_all_feature(self, inputs):
        inputs = np.array(inputs)
        # 最小值
        min = np.min(inputs)
        # 最大值
        max = np.max(inputs)
        # 均值
        mean = np.mean(inputs)
        # 四分位数范围
        iqr = stats.iqr(inputs)
        # 能量度量
        energy = self.energy(inputs)
        # FFT变换
        process = np.abs(fft(inputs)) / len(inputs) / 2
        # 频域偏度系数
        wskew = stats.skew(process)
        # 频域峰度系数
        wkurtosis = stats.kurtosis(process)
        # 将所有特征合并为数组
        array = [min, max, mean, iqr, energy, wskew, wkurtosis]
        return array

    # 计算能量度量
    def energy(self, inputs):
        return np.dot(inputs, np.transpose(inputs))/len(inputs)
