
# -*- coding:utf-8 -*-
import os
import random
import cv2


def get_500_identity(path_root, output_root):
    identity_total = os.listdir(path_root)
    #产生500个人
    identity_500 = []

    for i in range(600):
        training_id = random.randint(0, len(identity_total))
        identity_500.append(identity_total.pop(training_id))

    #print(identity_500)

    #每个人产生10张照片
    cout = 0
    identity_name = 1
    for identity in identity_500:
        #img_name = 1
        path_identity = os.path.join(path_root, identity)
        if not os.path.isdir(path_identity):
            continue
        imgs_total = os.listdir(path_identity)
        print('len of images: %d'%(len(imgs_total)))
        if len(imgs_total) < 10:
            continue
        cout += 1
        for i in range(10):
            img_id = random.randint(0, len(imgs_total)-1)
            img_path = os.path.join(path_identity, imgs_total.pop(img_id))
            img = cv2.imread(img_path)
            #cv2.imshow('test', img)
            #cv2.waitKey(1000)
            if not os.path.exists(os.path.join(output_root, str(identity_name))):
                os.makedirs(os.path.join(output_root, str(identity_name)))
            path_write = os.path.join(output_root, str(identity_name), str(i+1)+'.png')
            cv2.imwrite(path_write, img)
            print('Write %s'%path_write)
            #imgs_10.append(os.path.join(path_identity, imgs_total.pop(img_id)))
        identity_name += 1

        if cout == 500:
            print('Get 500 identity')
            break

        #得到了十张人脸


if __name__ == '__main__':
    CASIA_root = '/home/zx/CASIA-algin-112'
    output_root = '/home/zx/dongxijia/data_for_test_model_performance'
    get_500_identity(CASIA_root, output_root)