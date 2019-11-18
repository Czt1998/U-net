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
   本次实验的环境如下：<br>
 * Ubuntu 16.04<br>
 * python 3.6<br>
 * cuda 9.0<br>
 * pytorch<br><br>
   需要的第三方模块如下：<br>
 * numpy<br>
 * PIL<br>
 * tqdm<br>
 * skimage
 
## Operation instructions
   配置环境之后运行main.py

## Sample
<div align=center><img src="https://github.com/Czt1998/U-net/blob/master/dataset/test/0.png" /></div>

<div align=center><img src="https://github.com/Czt1998/U-net/blob/master/dataset/test/0_predict.png"/></div>

## Reference
   本次实验参考github有<br>
   [https://github.com/JavisPeng/u_net_liver](https://github.com/JavisPeng/u_net_liver)<br>
   [https://github.com/zhixuhao/unet](https://github.com/zhixuhao/unet)<br>
   也欢迎查看我的[博客](https://blog.csdn.net/ykben/article/details/103118619)
