# U-net
基于U-net的医学影像分割 / pytorch实现
## Main Code
 * `main.py`<br>
   使用Unet模型对ISBI Challenge 2012的数据集（详见dataset）进行训练，并对结果进行预测。
 * `unet.py`<br>
    Unet模型的pytorch实现
 * `DataHelper.py`<br>
   此代码中包含函数功能有：<br>
   1、对训练数据、测试数据的读取与矩阵化处理（在'main.py'被转成tensor）<br>
   2、对预测结果的二值化处理
 
## Operating environment
   1、Ubuntu 16.04<br>
   2、python 3.6<br>
   3、cuda 9.0<br>
   4、pytorch<br>
