# SSD-tensorflow
基于SSD的车辆检测  
1.数据为百度阿波罗自动驾驶检测数据集包含图像及四个检测点坐标,也可使用Labelme等标注工具自行标注,原始数据位于demo1文件夹.  
(1)数据预处理  
    首先运行txt_xml实现txt到xml数据转化(SSD使用VOC数据集),之后仿照VOC数据集格式对自己的数据集进行转化,文件夹VOC2007分别建立Annotations,ImageSets,JPEGImages三个文件夹并存入相应数据.之后将VOC格式的数据转化为tfrecord形式作为网络输入.  
(2)设置数据类别,并进行相应代码和网络调整.     
2.下载预训练好的vgg300网络进行训练,训练一小时模型可大致框住目标.结果在notebook文件夹demo1_test可看出.    
![image](https://github.com/dotah88/SSD-tensorflow/blob/master/image/%E4%B8%8B%E8%BD%BD.png)


