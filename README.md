# IndoorHIIT动作识别项目说明
> 山东大学（威海）<br>
> 18 数据科学 孙易泽

+ 本项目为通过微信小程序进行动作的识别，项目选取了徒手侧平举、前后交叉小跳、开合跳、半蹲四个动作，在测试者左手手持手机的情况下，利用微信小程序实时采集手机的六轴数据，并用随机森林模型和波峰检测法，对测试者做出的动作进行实时的识别和计数。

+ b站视频链接：https://www.bilibili.com/video/BV1oT4y1L7a8

+ 文档链接：https://nbviewer.jupyter.org/gist/Sun-Yize-SDUWH/9bee3f6c4533e768c8ca726084ba7f1e

+ 以下说明，为项目文件中各个文件夹的相关说明

### python project

+ data文件夹：训练所用数据，处理之后的数据
+ process文件夹：预处理数据代码，包括信号处理与窗口切割数据
+ feature文件夹：特征提取以及特征选取相关代码
+ machineLearning文件夹：各个算法测试比对，算法的优化与提升
+ numcount文件夹：动作计数相关代码测试
+ web文件夹：服务器部署代码
+ IndoorHIIT.ipynb：python全部工程说明文档，可在工程中直接查看，或访问以下网址：
  + https://nbviewer.jupyter.org/gist/Sun-Yize-SDUWH/9bee3f6c4533e768c8ca726084ba7f1e



### 微信小程序

+ 小程序已发布，二维码如下：

<img src='https://inifyy.cn:4100' width='500px'>

+ 完整页面文件以及js逻辑层在miniprogram文件夹中



### IndoorHIIT数据

+ 包含了本项目所收集的全部数据，已经将数据中安卓数据与苹果数据提前分开
