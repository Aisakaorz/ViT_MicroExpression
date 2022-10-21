import cv2
import dlib
import glob
import os.path
import shutil
import torch

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# 使用dlib分割人脸
def crop(files_list, files_dir):
    for file in tqdm(files_list):
        file_class = file.replace("\\", "/").split('/')[-3]  # 图片类别 (negative, neutral, positive)
        file_name = file.replace("\\", "/").split('/')[-1]  # 图片文件名
        if not os.path.isdir(files_dir + '/' + file_class):
            os.makedirs(files_dir + '/' + file_class)

        image = cv2.imread(file)

        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 使用特征提取器get_frontal_face_detector
        detector = dlib.get_frontal_face_detector()
        dots = detector(image_gray, 1)

        # for dot in dots:
        # # 将框画在原图上
        # # cv2.rectangle  参数1：图片， 参数2：左上角坐标， 参数2：左上角坐标， 参数3：右下角坐标， 参数4：颜色（R,G,B）， 参数2：粗细
        #     my_img = cv2.rectangle(image, (dot.left(), dot.top()), (dot.right(), dot.bottom()), (0, 0, 0), 1)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)

        # 人脸检测器
        predictor = dlib.shape_predictor(r'./shape_predictor_68_face_landmarks.dat')
        for dot in dots:
            shape = predictor(image, dot)
            # 将关键点绘制到人脸上
            # for i in range(68):
            #     cv2.putText(image, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_DUPLEX, 0.1, (0, 255, 0), 1, cv2.LINE_AA)
            #     cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255))
        # cv2.imshow("rotated", image)
        # cv2.waitKey(0)

        # 人脸对齐
        image = dlib.get_face_chip(image, shape, size=224)
        # cv2.imshow("68landmarks", image)
        # cv2.waitKey(0)

        cv2.imwrite(files_dir + '/' + file_class + '/' + file_name, image)


def copy(files_list, files_dir):
    for file in tqdm(files_list):
        file_class = file.replace("\\", "/").split('/')[-3]  # 图片类别 (negative, neutral, positive)
        file_name = file.replace("\\", "/").split('/')[-1]  # 图片文件名
        if not os.path.isdir(files_dir + '/' + file_class):
            os.makedirs(files_dir + '/' + file_class)
        shutil.copy(file, files_dir + '/' + file_class + '/' + file_name)


# 归一化 标准化
def get_mean_and_std(train_data):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in tqdm(train_loader):
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':

    image_list = glob.glob('../data_raw/*/*/*/*.png')
    file_dir = '../data_train'
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)  # 删除再建立
        os.makedirs(file_dir)
    else:
        os.makedirs(file_dir)

    train_files, val_files = train_test_split(image_list, test_size=0.2, random_state=46)
    train_dir = '../data_train/train'
    val_dir = '../data_train/val'

    # crop人脸并保存
    # crop(train_files, train_dir)
    # crop(val_files, val_dir)

    # 若不crop则使用以下代码
    copy(train_files, train_dir)
    copy(val_files, val_dir)

    # 归一化 标准化
    train_dataset = ImageFolder(root=r'../data_train', transform=transforms.ToTensor())
    with open('./mean_std.txt', 'w') as file:
        file.write(str(get_mean_and_std(train_dataset)))
    print(get_mean_and_std(train_dataset))
