# -tensorflow-
如何使用tensorflow训练图片

训练环境（ubuntu 16.04）：
CPU：Intel(R) Core(TM) i5-4460  CPU @ 3.20GHz
GPU：
内存：28G

下面是使用GPU进行训练的步骤。

搭建训练环境

安装VNC服务端
>sudo apt-get install xfce4 xfce4-goodies vnc4server
>vnc4server :1

>vi ~/.vnc/xstartup     ;修改此文件为以下三行
#!/bin/bash
xrdb $HOME/.Xresources
startxfce4 &

>vnc4server -kill :1     ;删除掉服务
>vnc4server :1           ;启动服务

注意在第一次打开时，选择默认选项。

安装gcc-4.9
>sudo apt-get install gcc-4.9
>sudo apt-get install g++-4.9
>sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 20
>sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 20
>sudo update-alternatives --config gcc
>sudo update-alternatives --config g++

安装cuda和cuDNN
需要这两个文件：
cuda-repo-ubuntu1604-8-0-rc_8.0.27-1_amd64.deb
cudnn-8.0-linux-x64-v6.0.tar
下载地址：
http://developer.download.nvidia.com/compute/cuda/8.0/direct/cuda-repo-ubuntu1604-8-0-rc_8.0.27-1_amd64.deb
https://developer.nvidia.com/rdp/cudnn-download

将下载好的文件上传到ubuntu中/data目录下。

>sudo dpkg -i /data/cuda-repo-ubuntu1604-8-0-rc_8.0.27-1_amd64.deb
>sudo apt-get update
>sudo apt-get install cuda
>sudo apt-get install nvidia-cuda-toolkit
>sudo apt-get install libcupti-dev

下面这两句，每次开新的命令行窗口都需要执行：
>export PATH=/usr/local/cuda-8.0/bin:$PATH
>export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

执行下面这句应该有类似的输出，如果没有，则重启一下系统。
>cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  361.77  Sun Jul 17 21:18:18 PDT 2016
GCC version:  gcc version 4.8.4 (Ubuntu 4.8.4-2ubuntu1~14.04.3)

执行下面这句应该有类似输出，如果没有，则重启一下系统。
>nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Wed_May__4_21:01:56_CDT_2016
Cuda compilation tools, release 8.0, V8.0.26

>sudo tar -xvf /data/cudnn-8.0-linux-x64-v6.0.tar -C /usr/local
>sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

安装cmake和Dlib的依赖库
>sudo apt-get install cmake
>sudo apt-get install libopenblas-dev liblapack-dev

测试cuDNN是否安装成功
>wget http://dlib.net/files/dlib-19.7.zip
>unzip dlib-19.7.zip
>cd dlib-19.7/examples
>mkdir build
>cd build
>cmake ../../dlib/cmake_utils/test_for_cuda      ;注意，运行这句的时候，如果提醒有东西没有安装，要你用apt-get安装时。都需要安装。然后删除掉build目录，重新从"cd dlib-19.7/examples"那一步开始。
>cmake --build .

安装Tensorflow
>sudo apt-get install python-pip python-dev
>pip install tensorflow-gpu

;下面是用来验证是否安装成功，进入python命令行，一行一行输入接下来的4行运行。
>python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

安装Tensorflow物体检测接口
>pip install pillow
>pip install lxml
>pip install IPython==5.0
;如果安装的过程中，提示pip需要升级，则升级（pip install --upgrade pip）
>sudo pip install jupyter
>sudo pip install matplotlib
>cd ~
>wget https://github.com/tensorflow/models/archive/master.zip
>unzip master.zip -d ~/tensorflow
>mv ~/tensorflow/models-master ~/tensorflow/models
>cd ~/tensorflow/models/research/
>sudo apt install protobuf-compiler
>protoc object_detection/protos/*.proto --python_out=.
>echo "export PYTHONPATH=$PYTHONPATH:~/tensorflow/models/research:~/tensorflow/models/research/slim">> ~/.bashrc
;退出系统重新登录，然后执行下面两行检查是否安装成功
>cd ~/tensorflow/models/research/
>python object_detection/builders/model_builder_test.py

准备数据集

下面的操作在Windows上进行：
使用Dlib的工具imglab进行标注。
然后，合并所有的标注数据到一个单独的xml文件，命名为：underground_corridor.xml。

下面的操作在Ubuntu 16.04上进行：
将create_imglab_tf_record.py复制到~/tensorflow/models/research/object_detection/
将underground_corridor_label_map.pbtxt复制到~/tensorflow/models/research/object_detection/data/
将标注好的underground_corridor.xml和图片目录，一起复制到~/tensorflow/models/research/
用下面的命令将Dlib格式的标注数据，转换成TensorFlow的tfrecord格式：

>python object_detection/create_imglab_tf_record.py --input_xml_path underground_corridor.xml --image_dir ./ --output_dir ./ --label_map_path underground_corridor_label_map.pbtxt
运行完成后，会生成from_imglab_train.record和from_imglab_val.record。一个用于训练，一个用于评估，将from_imglab_train.record改为underground_corridor_train.record，将from_imglab_val.record改为underground_corridor_val.record。
注意，underground_corridor_label_map.pbtxt这个文件内存放了ID与物体种类的对应关系。TensorFlow需要每个种类都有一个ID，所以，之后添加了其它分类，请手动更新这个文件，ID从1开始计数。

用准备好的数据集进行训练
下载Google预先在其它数据集上训练好的模型
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
我这里下载的是ssd_mobilenet_v1_coco模型

解压模型文件以及修改配置文件
>mkdir ~/underground_corridor
>mkdir ~/underground_corridor/data
>tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz -C ~/underground_corridor/data
>cp ~/underground_corridor/data/faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.index ~/underground_corridor/data/model.ckpt.index
>cp ~/underground_corridor/data/faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.meta ~/underground_corridor/data/model.ckpt.meta
>cp ~/underground_corridor/data/faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.data-00000-of-00001 ~/underground_corridor/data/model.ckpt.data-00000-of-00001
>cp underground_corridor_train.record ~/underground_corridor/data/underground_corridor_train.record
>cp underground_corridor_val.record ~/underground_corridor/data/underground_corridor_val.record
>cp object_detection/data/underground_corridor_label_map.pbtxt ~/underground_corridor/data/underground_corridor_label_map.pbtxt
>cp object_detection/samples/configs/faster_rcnn_resnet101_pets.config 

修改~/underground_corridor/data/faster_rcnn_resnet101_underground_corridors.config
1、修改路径，将PATH_TO_BE_CONFIGURED替换成~/underground_corridor/data（这里需要绝对路径，改成你的绝对路径）；
2、修改num_classes：label数目；
3、修改num_examples：评估图片的数量；
4、搜索所有的pet_，替换成underground_corridor_。

现在你的 ~/underground_corridor/data/目录应该看起来是这个样子:
>ls -1 ~/underground_corridor/data/
ssd_mobilenet_v1_coco_11_06_2017
faster_rcnn_resnet101_underground_corridor.config
model.ckpt.data-00000-of-00001
model.ckpt.index
model.ckpt.meta
underground_corridor_label_map.pbtxt
underground_corridor_train.record
underground_corridor_val.record

下面的步骤都在图形界面上运行（也是前面安装vnc的目的所在），即不要在ssh登录上去的纯命令行内运行：
开始训练
>mkdir ~/underground_corridor/train
>cd ~/tensorflow/models/research
>python object_detection/train.py --logtostderr --pipeline_config_path=~/underground_corridor/data/faster_rcnn_resnet101_underground_corridors.config --train_dir=~/underground_corridor/train

运行评估程序
>mkdir ~/underground_corridor/eval
>cd ~/tensorflow/models/research
>python object_detection/eval.py --logtostderr --pipeline_config_path=~/underground_corridor/data/faster_rcnn_resnet101_underground_corridors.config --checkpoint_dir=~/underground_corridor/train --eval_dir=~/underground_corridor/eval

运行看板程序
>tensorboard --logdir=~/underground_corridor
现在，在其它电脑上可以用 http://<ip>:6006 打开网页，实时监控训练进程。
 
导出Tensorflow训练好的模型
 当训练完你的模型之后,应该将其导出到一个原型Tensorflow图。通常需要检查三个文件：
 - model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001,
 - model.ckpt-${CHECKPOINT_NUMBER}.index
 - model.ckpt-${CHECKPOINT_NUMBER}.meta
 这三份文件放在~/underground_corridor/train，该目录示例如下：
> ls -1 ~/underground_corridor/train/
-rw-rw-r-- 1 gzdev gzdev       517 11月 18 09:20 checkpoint
-rw-rw-r-- 1 gzdev gzdev 493291692 11月 18 09:20 events.out.tfevents.1510912234.gzdev
-rw-rw-r-- 1 gzdev gzdev  16592471 11月 17 17:50 graph.pbtxt
-rw-rw-r-- 1 gzdev gzdev 438764272 11月 18 08:40 model.ckpt-191343.data-00000-of-00001
-rw-rw-r-- 1 gzdev gzdev     40511 11月 18 08:40 model.ckpt-191343.index
-rw-rw-r-- 1 gzdev gzdev   8691234 11月 18 08:40 model.ckpt-191343.meta
-rw-rw-r-- 1 gzdev gzdev 438764272 11月 18 08:50 model.ckpt-193543.data-00000-of-00001
-rw-rw-r-- 1 gzdev gzdev     40511 11月 18 08:50 model.ckpt-193543.index
-rw-rw-r-- 1 gzdev gzdev   8691234 11月 18 08:50 model.ckpt-193543.meta
-rw-rw-r-- 1 gzdev gzdev 438764272 11月 18 09:00 model.ckpt-195734.data-00000-of-00001
-rw-rw-r-- 1 gzdev gzdev     40511 11月 18 09:00 model.ckpt-195734.index
-rw-rw-r-- 1 gzdev gzdev   8691234 11月 18 09:00 model.ckpt-195734.meta
-rw-rw-r-- 1 gzdev gzdev 438764272 11月 18 09:10 model.ckpt-197873.data-00000-of-00001
-rw-rw-r-- 1 gzdev gzdev     40511 11月 18 09:10 model.ckpt-197873.index
-rw-rw-r-- 1 gzdev gzdev   8691234 11月 18 09:10 model.ckpt-197873.meta
-rw-rw-r-- 1 gzdev gzdev 438764272 11月 18 09:20 model.ckpt-200000.data-00000-of-00001
-rw-rw-r-- 1 gzdev gzdev     40511 11月 18 09:20 model.ckpt-200000.index
-rw-rw-r-- 1 gzdev gzdev   8691234 11月 18 09:20 model.ckpt-200000.meta
-rw-rw-r-- 1 gzdev gzdev      3936 11月 17 17:50 pipeline.config
开始导出
$ cd ~/tensorflow/models/research
$ python object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path ~/underground_corridor/data/faster_rcnn_resnet101_underground_corridors.config \
--trained_checkpoint_prefix ~/underground_corridor/train/model.ckpt-200000 \
--output_directory ~/underground_corridor/out/

导出的out目录示例如下：
> ls -1 ~/underground_corridor/out/
-rw-rw-r-- 1 gzdev gzdev        77 11月 20 11:13 checkpoint
-rw-rw-r-- 1 gzdev gzdev 190446830 11月 20 11:13 frozen_inference_graph.pb
-rw-rw-r-- 1 gzdev gzdev 249567356 11月 20 11:13 model.ckpt.data-00000-of-00001
-rw-rw-r-- 1 gzdev gzdev     25713 11月 20 11:13 model.ckpt.index
-rw-rw-r-- 1 gzdev gzdev   2636006 11月 20 11:13 model.ckpt.meta
drwxr-xr-x 3 gzdev gzdev      4096 11月 19 17:38 saved_model/
其中frozen_inference_graph.pb为我们的目标文件。

使用Tensorflow导出的模型识别图片
一下操作可在ssh中操作
$ sudo apt-get install protobuf-compiler -y
$ cd ~/tensorflow/models/research
$ protoc object_detection/protos/*.proto --python_out=.
$ cd ~
$ git clone https://github.com/qdraw/tensorflow-object-detection-tutorial.git 
$ cd tensorflow-object-detection-tutorial
$ chmod +x install.opencv.ubuntu.sh
$ ./install.opencv.ubuntu.sh
$ cp ~/underground_corridor/out/frozen_inference_graph.pb ~/tensorflow-object-detection-tutorial/ssd_mobilenet_v1_coco_11_06_2017
$cp ~/underground_corridor/train/underground_corridor_label_map.pbtxt ~/tensorflow-object-detection-tutorial/data

下面命令行在vnc中开始识别
$ python image_object_detection.py 

  
Reference
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md
https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
