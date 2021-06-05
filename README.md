# 欢迎使用Yolo_for_Pathological examination

@(作者)[slugger]

以下是Yolo_for_Pathological examination的详细配置及使用指导（**带*为可选步骤**）：
 >前情提要，该项目已针对假体、关节炎、内生软骨瘤、动脉瘤样骨囊肿等16种样例进行了290张小样本预训练

-------------------

[TOC]

##1.数据集准备



在数据准备阶段，需要将数据集文件夹整理为如下结构。并执行下一步操作，将xml转为txt进行存储摆放。

### 1.1数据文件存储摆放格式
``` python
bone
├── images
│   ├── train        # 训练集图片，这里我只列举几张示例
│   │   ├── 000050.jpg
│   │   ├── 000051.jpg
│   │   └── 000052.jpg
│   └── val          # 验证集图片
│       ├── 001800.jpg
│       ├── 001801.jpg
│       └── 001802.jpg
└── labels               
    ├── train       # 训练集的标签文件
    │   ├── 000050.txt
    │   ├── 000051.txt
    │   └── 000052.txt
    └── val        # 验证集的标签文件
        ├── 001800.txt
        ├── 001801.txt
        └── 001802.txt
```
### 1.2xml转为txt格式

预先画框标注好的数据集为图片和xml一一对应，需执行项目主目录下的**yolo.py**，将xml转为txt文件

代码的该部分需做相应修改：
1）画框标注的类别序号
2）xml文件路径和目标转为txt文件的路径
``` python
# 把voc的xml标签文件转化为yolo的txt标签文件
    # 1、类别
    classes1 = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         '10', '11', '12', '13', '14', '15' ]
    # 2、voc格式的xml标签文件路径
    xml_files1 = r'E:\work\yolov5out\bone\labels\xml'
    # 3、转化为yolo格式的txt标签文件存储路径
    save_txt_files1 = r'E:\work\yolov5out\bone\labels\train'
```



###1.3 数据集准备提示
>训练集train及验证集val比例建议为9：1或8：2。在数据集文件存储摆放格式处理时，务必使**images/train**目录下的图片和**labels/train**目录下的txt文件一一对应；**images/val**目录下的图片和**labels/val**目录下的txt文件一一对应.




## 2环境配置

### 2.1Python版本
**Yolo_for_Pathological examination**需要`python3.7以上`

### *2.2GPU准备
如只用CPU进行训练，将十分缓慢，建议进行GPU训练，需要安装对应`cuda`及`cudnn`，安装指导链接：https://blog.csdn.net/u011788214/article/details/117124772

### 2.3运行环境
cd到项目主目录下，执行如下进行项目依赖安装
>pip install -U -r requirements.txt

### 2.4修改数据和模型配置文件
配置文件为：`./data/coco128.yaml`，该文件中内容为：
``` python
Annotations   coco.yaml        hat_hair_beard.yaml  JPEGImages
coco128.yaml  get_coco2017.sh  ImageSets            VOC2007
(yolov5) shl@zfcv:~/shl/yolov5/data$ cat coco128.yaml
# COCO 2017 dataset http://cocodataset.org - first 128 training images
# Download command:  python -c "from yolov5.utils.google_utils import gdrive_download; gdrive_download('1n_oKgR81BJtqk75b00eAjdv03qVCQn2f','coco128.zip')"
# Train command: python train.py --data ./data/coco128.yaml
# Dataset should be placed next to yolov5 folder:
#   /parent_folder
#     /coco128
#     /yolov5


# train and val datasets (image directory or *.txt file with image paths)
train: ../coco128/images/train2017/
val: ../coco128/images/train2017/

# number of classes
nc: 80

# class names
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'](yolov5) shl@zfcv:~/shl/yolov5/data$

```
需要对其中如下三部分做修改
**1**.`训练集和验证集图片的路径`
``` python
train: ../coco128/images/train2017/
val: ../coco128/images/train2017/
```
**train**及**val**分别修改为主路径下数据集`./bone/images/train`及`./bone/images/val`，此处路径建议选择绝对路径（即存储数据集的完整路径）

**2**.修改`类别数nc`
 >`nc:16`#取决于分类类别总数
 
 **3**.修改类别列表，把类别修改为自己的类别
``` python
names = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         '10', '11', '12', '13', '14', '15' ]
```
 
### 2.5修改模型配置文件
修改模型配置文件，位置为主目录下的`models/yolov5s.yaml`模型的配置文件
`yolov5s.yaml`配置文件中原内容为：
``` python
# parameters
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# yolov5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 1-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
   [-1, 3, Bottleneck, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 4-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 6-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]], # 8-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 6, BottleneckCSP, [1024]],  # 10
  ]

# yolov5 head
head:
  [[-1, 3, BottleneckCSP, [1024, False]],  # 11
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1]],  # 12 (P5/32-large)

   [-2, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 1, Conv, [512, 1, 1]],
   [-1, 3, BottleneckCSP, [512, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1]],  # 17 (P4/16-medium)

   [-2, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 3, BottleneckCSP, [256, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1]],  # 22 (P3/8-small)

   [[], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]

```
因此，在`yolov5s.yaml`中只需要修改一处，把`nc`修改为自己的类别数即可
>nc:16


## 3.训练数据集

### 3.1直接训练：

训练命令：
>`python train.py --img 512 --batch 16 --epochs 50000 --data ./data/coco128.yaml --cfg ./models/yolov5s.yaml --weights ./weights/yolov5s.pt`


训练结束后，会在主目录下的runs/train/exp/weights文件夹下生成两个预训练的模型：

`best.pt`：保存的是中间一共比较好模型
`last.pt`：训练结束后保存的最后模型

### *3.2可视化训练：
开始之前，需要安装依赖
``` 
pip install wandb
```

训练命令：
>`python train.py --img 512 --batch 16 --epochs 50000 --data ./data/coco128.yaml --cfg ./models/yolov5s.yaml --weights ./weights/yolov5s.pt`

![Alt text](./035933dc-c1e5-4c49-962d-9e56e1adf0e4.png)


如果没有账号的话，就选择1在线创建；已有账号的话就选择2；不想使用wandb的话，就选择3。由于我已经在网页端注册过了，所以输入数字2

**wandb网站如无法访问，需要在本地运行，则访问https://docs.wandb.ai/guides/self-hosted/local进行本地启动操作**

终端提示需要wandb.ai的API Key，在浏览器中访问站点 https://wandb.ai/authorize，复制后贴到终端中**（复制到终端中的key是不显示的，复制粘贴后直接回车运行）**

训练结束后，会在主目录下的`runs/train/exp/weights`文件夹下生成两个预训练的模型：

`best.pt`：保存的是中间一共比较好模型
`last.pt`：训练结束后保存的最后模型

###  3.3进行模型测试

对文件夹内的所有图片进行测试



**(xxxxx/xx为存放测试图片文件的路径，该处进行简单示意，weights文件夹就为主目录下的`runs/train/exp/weights`，记得针对项目实际路径对下列执行命令做相应修改)**
>`python detect.py --source xxxx/xxxxxx/xxxx --weights xxxxx/weights/best.pt`

之后遍会在主目录下的`runs/detect/exp`下生成检测结果



