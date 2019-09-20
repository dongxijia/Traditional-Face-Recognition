import argparse

#系统库
import os
import sys
import shutil
import random
import time

#科学计算库
import cv2
import numpy as np

class Eigenface():

    #初始化模型参数
    def __init__(self, face_dataset, total_faces_n=40,img_size='92*112', energy=0.85):
        #energy应该是阈值的意思
        self.faces_dir = face_dataset
        self.energy = energy

        if self.faces_dir == 'att_faces':
            self.type_img = '.pgm'
        else:
            self.type_img = '.png'
        self.total_faces_n = total_faces_n
        self.training_faces_cout = 3
        self.test_faces_cout = 10-self.training_faces_cout
        #6*40, 40个人，每一个人取6张图片训练
        self.total_train_faces = total_faces_n * self.training_faces_cout
        self.img_hight, self.img_width = int(img_size.split('*')[0]), int(img_size.split('*')[1])
        self.training_ids_total = []
        cur_img = 0
        # 每一个人的训练图像集
        training_ids = []

        #用于训练的所有人脸列
        imgs_training_total = np.empty(shape=(self.img_width*self.img_hight, self.total_train_faces), dtype='float64')


        #每一个人face_id
        for face_id in range(1, self.total_faces_n+1):
            #在1到10中随机取9个，为什么是随机的呢
            training_ids = random.sample(range(1, 11), self.training_faces_cout)
            self.training_ids_total.append(training_ids)

            #每一张单张的人脸
            for training_id in training_ids:
                path_img = os.path.join(self.faces_dir, str(face_id), str(training_id)+self.type_img)

                print('> reading file: '+ path_img)

                img = cv2.imread(path_img, 0)#灰度图
                #把图像长方形拉成一列
                img_col = np.array(img, dtype='float64').flatten()
                #加到total training imgs里面去
                imgs_training_total[:, cur_img] = img_col[:]
                cur_img += 1

        #已经得到了完整的训练集, 然后求平均脸
        self.mean_img_col = np.sum(imgs_training_total, axis=1) / self.total_train_faces

        for col in range(0, self.total_train_faces):
            imgs_training_total[:, col] -= self.mean_img_col[:]

        #协方差矩阵C
        C = np.matrix(imgs_training_total.transpose()) * np.matrix(imgs_training_total)

        #求列平均
        C /= self.total_train_faces

        self.cal_eva_evc(C)
        # 再imgs_training_total×特征向量，才是真正的特征向量
        self.evectors = imgs_training_total * self.evectors
        #归一化
        norms = np.linalg.norm(self.evectors, axis=0)
        self.evectors = self.evectors / norms

        #人脸的权重向量W， Ω
        print('Cal the weight of face')
        self.W = self.evectors.transpose() * imgs_training_total


    def cal_eva_evc(self, C):
        # 求协方差矩阵的特征值和特征向量
        self.evalues, self.evectors = np.linalg.eig(C)
        #indices返回从大到小的索引值
        sort_indices = self.evalues.argsort()[::-1]
        self.evalues = self.evalues[sort_indices]
        self.evectors = self.evectors[:, sort_indices]


        evalues_sum = sum(self.evalues[:])
        evalues_cout = 0
        evalues_energy = 0.0#阈值?权重

        for evalue in self.evalues:
            evalues_cout += 1
            evalues_energy += evalue / evalues_sum

            #特征值大小的意义是什么？特征值越大，特征向量越能描述原矩阵吗？
            if evalues_energy >= self.energy:
                break

        #剪枝特征向量和特征值, 找到主成分，其实就是PCA的原理
        self.evalues = self.evalues[0:evalues_cout]
        self.evectors = self.evectors[:,0:evalues_cout]

    def search_in_dataset(self, path_to_img):
        img = cv2.imread(path_to_img, 0)
        #把人脸也拉成一列
        img_col = np.array(img, dtype='float64').flatten()
        #减去平均脸
        img_col -= self.mean_img_col
        img_col = np.reshape(img_col, (self.img_width*self.img_hight, 1))
        #将人脸投影到特征向量上

        S = self.evectors.transpose() * img_col

        diff = self.W - S
        #计算距离的模
        norms = np.linalg.norm(diff, axis=0)

        #找到距离最小的那一列
        closet_face_id = np.argmin(norms)
        #找到face_id
        return int(closet_face_id/self.training_faces_cout) +1

    def verification(self):
        #可用10折交叉验证，来评估模型的准确率
        #从训练集中分割出的每人4张人脸，与训练集是同分布的
        print('> Evaluation ATT faces started')
        results_file = os.path.join('results', 'casia500_results_t.txt')
        #新建文件夹
        if not os.path.exists('results'):
            os.makedirs(results_file)
        test_count = self.test_faces_cout*self.total_faces_n
        test_correct = 0
        with open(results_file, 'w') as f:
            for face_id in range(1, self.total_faces_n+1):
                #用没有用来测试的图像来做验证
                for test_id in range(1, 11):
                    if test_id not in self.training_ids_total[face_id-1]:
                        path_to_img = os.path.join(self.faces_dir, str(face_id), str(test_id)+self.type_img)
                        result_id = self.search_in_dataset(path_to_img)
                        result = (result_id == face_id)

                        if result == True:
                            test_correct += 1
                            f.write('image: %s\nresult: correct\n\n'%(path_to_img))
                        else:
                            f.write('image: %s\nresult: wrong, got %2d\n\n'%(path_to_img, result_id))

            print('> Evaluating ATT faces ended')
            self.accuracy = float(100. * test_correct/test_count)
            print('Correct: '+ str(self.accuracy) + '%')
            f.write('Correct: %.2f\n' %(self.accuracy))







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My implement of Eigenface face recognition')
    #parser.add_argument('--face_dataset', type=str, default='att_faces',
    #                    help='train image root')
    parser.add_argument('--total_faces_n', type=int, default=500,
                        help='the number of identity')
    parser.add_argument('--img_size', type=str, default='112*112',
                        help='the size of image')

    parser.add_argument('--face_dataset', type=str, default='/home/zx/dongxijia/data_for_test_model_performance', help='train image root')
    #parser.add_argument('--test_dataset', type=str, default='celebrity_faces', help='test image root')

    args = parser.parse_args()
    #初始化模型
    accuarys = []
    N = 0
    since = time.time()
    for i in range(10):
        EigenFace = Eigenface(args.face_dataset, args.total_faces_n, args.img_size)
    #得到人脸权重向量
    #在训练集的同分布验证集上测试结果
        EigenFace.verification()
        accuarys.append(EigenFace.accuracy)
        N = EigenFace.training_faces_cout
    average_time = (time.time() - since)/10
    print('Average time: %d s'%average_time)
    print('Number of training images %d' % N)
    print('Mean accury is %.2f' % np.mean(accuarys))





