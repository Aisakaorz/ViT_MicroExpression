import glob
import os
import shutil

from sklearn.model_selection import train_test_split

image_list = glob.glob('data/*/*/*/*.png')
file_dir = 'dataset'

if os.path.exists(file_dir):
    shutil.rmtree(file_dir)  # 删除再建立
    os.makedirs(file_dir)
else:
    os.makedirs(file_dir)

train_val_files, val_files = train_test_split(image_list, test_size=0.2, random_state=42)
train_dir = 'train'
val_dir = 'val'
train_root = os.path.join(file_dir, train_dir)
val_root = os.path.join(file_dir, val_dir)
for file in train_val_files:
    file_class = file.replace("\\", "/").split('/')[-3]
    file_name = file.replace("\\", "/").split('/')[-1]
    file_class = os.path.join(train_root, file_class)
    if not os.path.isdir(file_class):
        os.makedirs(file_class)
    shutil.copy(file, file_class + '/' + file_name)

for file in val_files:
    file_class = file.replace("\\", "/").split('/')[-3]
    file_name = file.replace("\\", "/").split('/')[-1]
    file_class = os.path.join(val_root, file_class)
    if not os.path.isdir(file_class):
        os.makedirs(file_class)
    shutil.copy(file, file_class + '/' + file_name)
