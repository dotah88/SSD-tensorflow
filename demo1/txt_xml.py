import os, sys
import glob
from PIL import Image

# VEDAI 图像存储位置
src_img_dir = "/home/mt/learn/SSD-Tensorflow-master/demo1/train/images"
# VEDAI 图像的 ground truth 的 txt 文件存放位置
src_txt_dir = "/home/mt/learn/SSD-Tensorflow-master/demo1/train/labels"
src_xml_dir = "/home/mt/learn/SSD-Tensorflow-master/demo1/train/xml"
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

count_ignore = 0
count_car =0
count_ped = 0
count_cyc =0
count_tru =0
count_mot =0
count_tri = 0
count_bar = 0
count_van = 0
count_tra = 0
count_bus = 0
for img in img_names:
    im = Image.open((src_img_dir + '/' + img + '.jpg'))
    width, height = im.size

    # open the crospronding txt file
    gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
    #gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()

    # write in xml file
    #os.mknod(src_xml_dir + '/' + img + '.xml')
    # xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
    # xml_file.write('<annotation>\n')
    # xml_file.write('    <folder>VOC2007</folder>\n')
    # xml_file.write('    <filename>' + str(img) + '.png' + '</filename>\n')
    # xml_file.write('    <size>\n')
    # xml_file.write('        <width>' + str(width) + '</width>\n')
    # xml_file.write('        <height>' + str(height) + '</height>\n')
    # xml_file.write('        <depth>3</depth>\n')
    # xml_file.write('    </size>\n')

    # write the region of image on xml file
    if 'ignore' in gt:
        count_ignore += 1
    if 'car' in gt:
        count_car += 1
    if 'pedestrian' in gt:
        count_ped += 1
    if 'cyclist' in gt:
        count_cyc += 1
    if 'truck' in gt:
        count_tru += 1
    if 'motorcyclist' in gt:
        count_mot += 1
    if 'tricyclelist' in gt:
        count_tri += 1
    if 'barrowlist' in gt:
        count_bar += 1
    if 'van' in gt:
        count_van += 1
    if 'trafficcone' in gt:
        count_tra += 1
    if 'bus' in gt:
        count_bus += 1


    for img_each_label in gt:
        spt = img_each_label.split(' ') #这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
        spt_list.append(spt[0])
        # if 'ignore' in spt_list:
        #     count_ignore+=1
        #     count.append(count_ignore)
        # if 'car' in spt_list:
        #     count_car+=1
        #     count.append(count_car)
        # if 'pedestrian' in spt_list:
        #     count_ped+=1
        #     count.append(count_ped)
        # if 'cyclist' in spt_list:
        #     count_cyc+=1
        #     count.append(count_cyc)
        # if 'truck' in spt_list:
        #     count_tru+=1
        #     count.append(count_tru)
        # if 'motorcyclist' in spt_list:
        #     count_mot+=1
        #     count.append(count_mot)
        # if 'tricyclelist' in spt_list:
        #     count_tri+=1
        #     count.append(count_tri)
        # if 'barrowlist' in spt_list:
        #     count_bar+=1
        #     count.append(count_bar)
        # if 'van' in spt_list:
        #     count_van+=1
        # if 'trafficcone' in spt_list:
        #     count_tra+=1
        #     count.append(count_van)
        # if 'bus' in spt_list:
        #     count_bus+=1
        #     count.append(count_bus)

        # xml_file.write('    <object>\n')
        # xml_file.write('        <name>' + str(spt[0]) + '</name>\n')
        # xml_file.write('        <pose>Unspecified</pose>\n')
        # xml_file.write('        <truncated>0</truncated>\n')
        # xml_file.write('        <difficult>0</difficult>\n')
        # xml_file.write('        <bndbox>\n')
        # xml_file.write('            <xmin>' + str(spt[4]) + '</xmin>\n')
        # xml_file.write('            <ymin>' + str(spt[5]) + '</ymin>\n')
        # xml_file.write('            <xmax>' + str(spt[6]) + '</xmax>\n')
        # xml_file.write('            <ymax>' + str(spt[7]) + '</ymax>\n')
        # xml_file.write('        </bndbox>\n')
        # xml_file.write('    </object>\n')
    # xml_file.write('</annotation>')

for i in spt_list:
    if i not in empty:
        class_names.append(i)
print(set(class_names))
print("======================")
print(count_ignore,count_car,count_ped,count_cyc,count_tru,count_mot,count_tri,count_bar,count_van,count_tra,count_bus)
print("=========================")
print(len(spt_list))
print('======================================')
print(spt_list.count('ignore'),spt_list.count('car'),spt_list.count('pedestrian'),spt_list.count('cyclist'),spt_list.count('truck'),spt_list.count('motorcyclist'),spt_list.count('tricyclelist'),spt_list.count('barrowlist'),spt_list.count('van'),spt_list.count('trafficcone'),spt_list.count('bus'))
# TRAIN_STATISTICS = {
#     'none': (0, 0),
#     'ignore': (1, 1),
#     'car': (1, 1),
#     'pedestrian': (1, 1),
#     'cyclist': (1, 1),
#     'truck': (1, 1),
#     'motorcyclist': (1, 1),
#     'tricyclelist': (1, 1),
#     'barrowlist': (1, 1),
#     'van': (1, 1),
#     'trafficcone': (1, 1),
#     'bus': (1, 1),
#     'total': (11, 11),