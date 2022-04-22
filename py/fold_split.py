import numpy as np
import os,code
import cv2
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

import numpy as np
from skimage.io import imshow
import matplotlib.pyplot

import csv
import shutil
import code


lines = open('train.csv').readlines()
label_dict = {}
label_list = [[],[],[],[],[]]

for line in lines[1:]:
	name, label = line.strip().split(",")
	label_dict[name+".png"] = label
	label_list[int(label)].append(name + '.png')

####################################################

label_train_dic = {}
for line in lines:
    file_name, label = line.strip().split(",")
    label_train_dic[file_name+'.png'] =  label

####################################################

fold_list = [[],[],[],[],[]]

for class_num, class_list in enumerate(label_list):
	chunk = np.array_split(class_list, 5)
	for fold_num in range(5):
		for item in  chunk[fold_num]:
			fold_list[fold_num].append(item)

####################################################

#Make a CSV file having the information of image name, class, and fold
row_list = [['img_file', 'class', 'fold']]
for fold_num, in_fold_list in enumerate(fold_list):
	for item in in_fold_list:
		row_list.append([item, label_train_dic[item], fold_num])


with open('image_5folds.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	writer.writerows(row_list)

#Divide the images into five folders
for i in range(5):
	for image in fold_list[i]:
		shutil.copy('train_images_crop_224_224/1/'+image, 'cv_fold{}'.format(i))


