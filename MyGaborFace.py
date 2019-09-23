import math
import os
import argparse
import time

import cv2
import numpy as np
from BaseFace import BaseFace

class MyGaborFace(BaseFace):
    '''
    Using Gabor filter for face recogniztion
    cv2.getGaborKernel也实现了对应的代码
    '''
    #def __init__(self, name_dataset,dir_data, size_img, N_identity, N_training_img):
    #    self.init_features()




    def init_features(self):
        self.feaures_vectors = []

        #很多用list的方法，都可以numpy实现，而且更优化
        img_templete = np.zeros(shape=(self.height, self.width), dtype='float64')
        for i in range(self.N_total_training_img):
            img_templete = self.imgs_training[:, i].reshape(self.height, self.width)
            feature_vectors = self.extractFeatures(img_templete)
            self.feaures_vectors.append(feature_vectors)
        print('Init features down!')

    def cal_distance_features(self, fv1, fv2):
        #输入是features的输出k，是一个图像的10个gabor特征list
        normlist = []
        for i in range(len(fv1)):
            k = fv1[i]
            p = fv2[i]

            normlist.append((p-k)**2.0)
        sums = sum([i.sum() for i in normlist])
        #除以100000是为什么?
        return math.sqrt(sums)/100000

    def find_identity(self, path_img):
        img = cv2.imread(path_img, 0)
        fv_unkown_img = self.extractFeatures(img)
        distes = [self.cal_distance_features(fv_unkown_img, fv_i) for fv_i in self.feaures_vectors]
        #print('Face set contains %d faces'%len(distes))
        return int(distes.index(min(distes))/self.N_training_img)+1

    def verification(self):
        print('> Evaluation %s faces started' %self.name_dataset )
        results_file = os.path.join('results', '%s_results_t.txt' % self.name_dataset)

        if not os.path.exists('results'):
            os.makedirs('results')
        test_cout = self.N_testing_img * self.N_identity
        cout_correct = 0.0
        with open(results_file, 'w') as f:
            for id_face in range(1, self.N_identity + 1):
                # 把没有用于训练的图像拿来做验证
                for id_test in range(1, 11):
                    if id_test not in self.ids_training[id_face - 1]:
                        path_to_img = os.path.join(self.dir_identity, str(id_face), str(id_test) + self.type_img)
                        id_result = self.find_identity(path_to_img)
                        result = (id_result == id_face)

                        if result == True:
                            cout_correct += 1
                            f.write('image: %s\nresult: correct\n\n' % (path_to_img))
                        else:
                            f.write('image: %s\nresult: wrong, got %2d\n\n' % (path_to_img, id_result))

            print('> Evaluating %s faces ended' % self.name_dataset)
            self.accuary = float(100. * cout_correct / test_cout)
            print('Correct: ' + str(self.accuary) + '%')
            f.write('Correct: %.2f\n' % (self.accuary))


    def bulid_filters(self, w, h, num_theta, fi, sigma_x, sigma_y, psi):
        '''Get set of filters for GABOR'''
        #num_theta是将360分成多少个频率段，n个频率段的Gabor结果级联起来就是Gabor特征
        #fi是filter的数量吗, fi*num_thetha才是卷积核的数量
        filters = []
        #fi=(0.75, 1.5),fi是波长分之一，就是频率嘛
        #每个频率下的不同倾斜(反向)角度的滤波器,2个频率下的5中倾斜度，即一共10个滤波器

        for i in range(num_theta):
            theta = (i+1.0)/num_theta * np.pi
            for f_var in fi:
                kernel = self.get_Gabor_kernel(w, h, sigma_x, sigma_x, theta, f_var, psi)
                # 1.5倍和2倍有什么区别呢
                # kernel = 1.5*kernel/kernel.sum()

                filters.append(kernel)
        return filters


    def get_Gabor_kernel(self, w, h, sigma_x, sigma_y, theta, fi, psi):
        "getting gabor kernel with those values"
        #psi代表相位（角度）偏移
        #sigma_x = 2, sigma_y=1, 但是这个值和带宽有关，这里取得并不一定对
        kernel_size_x = w
        kernel_size_y = h
        #y, x是坐标矩阵
        (y, x) = np.meshgrid(np.arange(0, kernel_size_y), np.arange(0, kernel_size_x))

        #旋转是什么意思, 注意x, y都是矩阵形式
        x_theta = x * np.cos(theta) + y*np.sin(theta)
        y_theta = -x*np.sin(theta) + y*np.cos(theta)

        #计算Gabor滤波的实数部分, 返回的是一个矩阵
        Gabor_kernel_real = np.exp(-1.0*(x_theta ** 2.0/ sigma_x**2.0 + y_theta**2.0/sigma_y**2.0))*np.cos(2*np.pi*fi*x_theta+psi)
        return Gabor_kernel_real


    def extractFeatures(self, img):
        '''

        :param img:
        :return: n*gabor_kenel's feature list
        '''
        filters = self.bulid_filters(img.shape[0], img.shape[1], 5, (0.75, 1.5), 2, 1, np.pi/2)
        #滤波器也需要变换到频率域中
        #fft.fft2，2维傅里叶变换
        fft_filters = [np.fft.fft2(i) for i in filters]
        #图像也要从空间域变换到频率域中
        img_fft = np.fft.fft2(img)
        #频率域中相乘
        a = [img_fft*i_filter for i_filter in fft_filters]
        #从频率域中变换回空间域
        s = [np.fft.ifft2(i) for i in a]

        #取实数
        k = [i.real for i in s]
        #k = np.array(k).flatten()
        #k是特征的list,其实可以级联成一个numpy
        return k










def parse_args():
    parser = argparse.ArgumentParser(description='My Fisherface parameters')
    # general
    parser.add_argument('--name_dataset', default='att', help='name of dataset, including att and CASIA-500')
    parser.add_argument('--dir_data', default='/home/zx/dongxijia/Eigenfaces/Eigenfaces/att_faces', help='training and testing set directory')
    #parser.add_argument('--dir_data', default='/home/zx/dongxijia/data_for_test_model_performance',
    #                    help='training and testing set directory')
    parser.add_argument('--size_img', default='92*112', help='image size')
    parser.add_argument('--N_identity', default=40, help='Numbers of dataset person')
    parser.add_argument('--N_training_img', default=6, help='Number of person for training in a indentity, rest is test')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    accuarys = []
    since = time.time()
    for i in range(1, 10):
        #face_id = '/home/zx/dongxijia/Eigenfaces/Eigenfaces/att_faces/%d/3.pgm'%i
        #result_id = sample_Gabor.find_identity('/home/zx/dongxijia/Eigenfaces/Eigenfaces/att_faces/%d/5.pgm'%i)
        sample_Gabor = MyGaborFace(args.name_dataset, args.dir_data, args.size_img, args.N_identity,
                                   args.N_training_img)
        sample_Gabor.init_features()
        sample_Gabor.verification()
        #print('Real_id %s, result_id is %d' % (face_id, result_id))
        accuarys.append(sample_Gabor.accuary)
        average_time = (time.time() - since) / 10
    print('Average time: %d s' % average_time)
    print('Number of training images %d' % args.N_training_img)
    print('Mean accury is %.2f' % np.mean(accuarys))








