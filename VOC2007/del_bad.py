import os, sys
import glob
from PIL import Image

# VEDAI 图像存储位置
src_img_dir = "/home/mt/learn/SSD-Tensorflow-master/demo1/train/images"
# VEDAI 图像的 ground truth 的 txt 文件存放位置
src_txt_dir = "/home/mt/learn/SSD-Tensorflow-master/demo1/train/labels"
count=[]
img_Lists = glob.glob(src_img_dir + '/*.jpg')
class_names=[]
spt_list=[]
empty=[]
count1=[]
img_basenames = [] # e.g. 100.jpg
for item in img_Lists:
    img_basenames.append(os.path.basename(item))

img_names = [] # e.g. 100
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_names.append(temp1)

ign_cou = 0
car_cou = 0
ped_cou = 0
cyc_cou = 0
tru_cou = 0
mot_cou = 0
tri_cou = 0
bar_cou = 0
van_cou = 0
tra_cou = 0
bus_cou = 0
ign_img_cou = 0
car_img_cou = 0
ped_img_cou = 0
cyc_img_cou = 0
tru_img_cou = 0
mot_img_cou = 0
tri_img_cou = 0
bar_img_cou = 0
van_img_cou = 0
tra_img_cou = 0
bus_img_cou =0
for img in img_names:
    im = Image.open((src_img_dir + '/' + img + '.jpg'))
    width, height = im.size
    # open the crospronding txt file
    gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
    gt1 = open(src_txt_dir + '/' + img + '.txt').read()
    for i in gt:
        spt = i.split(' ')
        if spt[0]=='ignore':
            ign_cou+=1
        if spt[0]=='car':
            car_cou+=1
        if spt[0]=='pedestrian':
            ped_cou+=1
        if spt[0]=='cyclist':
            cyc_cou+=1
        if spt[0]=='truck':
            tru_cou+=1
        if spt[0]=='motorcyclist':
            mot_cou+=1
        if spt[0]=='tricyclelist':
            tri_cou+=1
        if spt[0]=='barrowlist':
            bar_cou+=1
        if spt[0]=='van':
            van_cou+=1
        if spt[0]=='trafficcone':
            tra_cou+=1
        if spt[0]=='bus':
            bus_cou+=1
    if 'ignore' in gt1:
        ign_img_cou+=1
    if 'car' in gt1:
        car_img_cou+=1
    if 'pedestrian' in gt1:
        ped_img_cou+=1
    if 'cyclist' in gt1:
        cyc_img_cou+=1
    if 'truck' in gt1:
        tru_img_cou+=1
    if 'motorcyclist' in gt1:
        mot_img_cou+=1
    if 'tricyclelist' in gt1:
        tri_img_cou+=1
    if 'barrowlist' in gt1:
        bar_img_cou+=1
    if 'van' in gt1:
        van_img_cou+=1
    if 'trafficcone' in gt1:
        tra_img_cou+=1
    if 'bus' in gt1:
        bus_img_cou+=1


print(bus_img_cou)
print(bus_cou)
