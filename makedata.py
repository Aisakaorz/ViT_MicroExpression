# -*- coding = utf-8 -*-
# @File : makedata.py
# @Software : PyCharm
import glob
import os
import shutil

image_list=glob.glob('data/NHFIER/*/*.png')
print(image_list)
file_dir='dataset'
if os.path.exists(file_dir):
    print('true')
    #os.rmdir(file_dir)
    shutil.rmtree(file_dir)#删除再建立
    os.makedirs(file_dir)
else:
    os.makedirs(file_dir)

from sklearn.model_selection import train_test_split

trainval_files, val_files = train_test_split(image_list, test_size=0.3, random_state=42)

train_dir='train'
val_dir='val'
train_root=os.path.join(file_dir,train_dir)
val_root=os.path.join(file_dir,val_dir)
for file in trainval_files:
    file_class=file.replace("\\","/").split('/')[-2]
    file_name=file.replace("\\","/").split('/')[-1]
    file_class=os.path.join(train_root,file_class)
    if not os.path.isdir(file_class):
        os.makedirs(file_class)
    shutil.copy(file, file_class + '/' + file_name)

for file in val_files:
    file_class=file.replace("\\","/").split('/')[-2]
    file_name=file.replace("\\","/").split('/')[-1]
    file_class=os.path.join(val_root,file_class)
    if not os.path.isdir(file_class):
        os.makedirs(file_class)
    shutil.copy(file, file_class + '/' + file_name)
