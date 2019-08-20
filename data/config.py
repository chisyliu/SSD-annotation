# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("/home/home2/guojianhua/ssd.pytorch-master")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

#缩小物体时的均值当做幕布，写成超参数了
MEANS = (104, 117, 123)

#这里的类数是类的数目加上背景
# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1], #featuremap的长或者宽
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300], #特征图上的点到图片上点的缩放距离 可以发现 大概相乘大概是300
    'min_sizes': [30, 60, 111, 162, 213, 264],#特征图上的点在原图上的框的大小
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],#（2，3）表示有1/2，2/1，1/3，3/1
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
