import shutil
import os
from config.config import *

def data_split():
    images = []
    with open(path_images, 'r') as f1:
        for line in f1:
            images.append(list(line.strip('\n').split(',')))

    # 读取train_test_split.txt文件
    split = []
    with open(path_split, 'r') as f2:
        for line in f2:
            split.append(list(line.strip('\n').split(',')))

    classes = []
    with open(path_classes, 'r') as f3:
        for line in f3:
            classes.append(list(line.strip('\n').split(',')))

    # import pdb
    # pdb.set_trace()
    # 划分
    num = len(images)
    num_classes = len(classes)  # 图像的总个数
    for k in range(num):
        file_name = images[k][0].split(' ')[1].split('/')[0]
        if int(split[k][0][-1]) == 1:
            for i in range(num_classes):
                if classes[i][0].split(' ')[1] == file_name:
                    # import pdb
                    # pdb.set_trace()
                    if os.path.exists(train_path):
                        with open(train_path, "a") as fp:
                            fp.write(trian_save_path + images[k][0].split(' ')[1].split('/')[1] + ' ' +
                                     str(int(classes[i][0].split(' ')[0])-1) + "\n")
                        fp.close()
                    else:
                        if not os.path.exists(datadir_path):
                            os.makedirs(datadir_path)
                        with open(train_path, 'w+') as fp:
                            fp.write(trian_save_path + images[k][0].split(' ')[1].split('/')[1] + ' ' +
                                     str(int(classes[i][0].split(' ')[0])-1) + "\n")
                        fp.close()
            if os.path.isdir(trian_save_path):
                shutil.copy(ROOT_PATH + 'images/' + images[k][0].split(' ')[1],
                            trian_save_path + images[k][0].split(' ')[1].split('/')[1])
            else:
                os.makedirs(trian_save_path)
                shutil.copy(ROOT_PATH + 'images/' + images[k][0].split(' ')[1],
                            trian_save_path + images[k][0].split(' ')[1].split('/')[1])

            # print('%s处理完毕!' % images[k][0].split(' ')[1].split('/')[1])

        else:
            for i in range(num_classes):
                if classes[i][0].split(' ')[1] == file_name:
                    if os.path.exists(test_path):
                        with open(test_path, "a") as fp:
                            fp.write(test_save_path + images[k][0].split(' ')[1].split('/')[1] + ' ' +
                                     str(int(classes[i][0].split(' ')[0])-1) + "\n")
                        fp.close()
                    else:
                        if not os.path.exists(datadir_path):
                            os.makedirs(datadir_path)
                        with open(test_path, 'w+') as fp:
                            fp.write(test_save_path + images[k][0].split(' ')[1].split('/')[1] + ' ' +
                                     str(int(classes[i][0].split(' ')[0])-1) + "\n")
                        fp.close()
            if os.path.isdir(test_save_path):
                shutil.copy(ROOT_PATH + 'images/' + images[k][0].split(' ')[1],
                            test_save_path + images[k][0].split(' ')[1].split('/')[1])
            else:
                os.makedirs(test_save_path)
                shutil.copy(ROOT_PATH + 'images/' + images[k][0].split(' ')[1],
                            test_save_path + images[k][0].split(' ')[1].split('/')[1])
    print('处理完毕!')
