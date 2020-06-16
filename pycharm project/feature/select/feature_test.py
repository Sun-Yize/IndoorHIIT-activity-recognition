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
        # 中值
        median = np.median(inputs)
        # 中值绝对偏差
        mad = stats.median_absolute_deviation(inputs)
        # 标准差
        std = np.std(inputs, ddof=1)
        # 偏度
        skew = stats.skew(inputs)
        # 峰度
        kurtosis = stats.kurtosis(inputs)
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
        array = [min, max, mean, median, mad, std, skew, kurtosis, iqr, energy, wskew, wkurtosis]
        return array

    # 计算能量度量
    def energy(self, inputs):
        return np.dot(inputs, np.transpose(inputs))/len(inputs)
