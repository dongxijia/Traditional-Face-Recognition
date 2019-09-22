import argparse
import random
import os
import time


#科学计算库
import cv2
import numpy as np

class Fisherface():

    def __init__(self, name_dataset,dir_data, size_img, N_identity, N_training_img):
        #注意命名事项，先说属性名词，再说形容词
        #FisherFace基于LDA，LDA对每个类别样本求均值，而PCA是对所有样本数据求均值,也是必须灰度图吗
        self.name_dataset = name_dataset
        self.dir_identity = dir_data
        self.height, self.width = size_img.split('*')[0], size_img.split('*')[1]
        self.S = int(self.width) * int(self.height)
        self.N_identity = N_identity
        self.N_training_img = N_training_img
        self.N_total_training_img = self.N_identity * self.N_training_img
        self.N_testing_img = 10 - self.N_training_img
        self.init_imgs()

        #将Y用于计算LDA
        self.cal_PCA()
        self.cal_LDA()

    
    def init_imgs(self):
        #Imgs_total  = np.empty(shape=(self.S, self.N_total_training_img),dtype='float64')
        if self.name_dataset == 'att':
            self.type_img = '.pgm'
        if self.name_dataset == 'CASIA-500':
            self.type_img = '.png'

        #初始化np，因为np的array不能改变形状的
        self.imgs_training = np.empty(shape=(self.S, self.N_total_training_img), dtype='float64')
        self.ids_training = []
        #idx_identity = 0
        index_img = 0

        for id_face in range(1, self.N_identity+1):
            #产生N个训练图像的list
            ids_img = random.sample(range(1, 11), self.N_training_img)
            self.ids_training.append(ids_img)

            for id_img in ids_img:
                path_img = os.path.join(self.dir_identity, str(id_face), str(id_img)+ self.type_img)

                print('> reading file: ' + path_img)
                img = cv2.imread(path_img, 0)
                vector_img = np.array(img, dtype='float64').flatten()
                #加到总的imgs_training中去
                self.imgs_training[:, index_img] = vector_img[:]
                index_img += 1

    def cal_PCA(self):
        #在进行LDA之前，需要PCA的处理结果
        self.mean_total = self.imgs_training.mean(axis=1)
        for col in range(0, self.N_total_training_img):
            self.imgs_training[:, col] -= self.mean_total[:]

        #计算协方差矩阵, 原本是List，要转成numpy的array，用了np.matrix
        C = np.matrix(self.imgs_training.transpose())*np.matrix(self.imgs_training)

        #求每列平均
        C /= self.N_total_training_img

        self.evalues, self.evectors = np.linalg.eig(C)
        '''
        #计算真正的特征向量, S*N = S*N X N*N，有N个人的嘛
        self.evectors = np.matrix(self.imgs_training) * np.matrix(self.evectors)

        #计算向量的模
        norm = np.linalg.norm(self.evectors, axis=0)
        #归一化
        self.evectors /= norm
        '''
        #对特征值和特征向量进行排序，求主成分
        sort_indices = self.evalues.argsort()[::-1]
        self.evectors = self.evectors[:, sort_indices]
        self.evalues = self.evalues[sort_indices]

        #m的意义是什么，m是阈值
        evalues_sum = sum(self.evalues[:])
        self.energy = 0.85
        self.cout_evalues = 0
        evalue_energy = 0.0

        for evalue in self.evalues:
            self.cout_evalues += 1
            evalue_energy += evalue / evalues_sum

            if evalue_energy >= self.energy:
                break

        self.evectors = self.evectors[:, 0:self.cout_evalues]

        # 计算真正的特征向量, S*N = S*N X N*N，有N个人的嘛
        self.pca_evectors = np.matrix(self.imgs_training) * np.matrix(self.evectors)

        # 计算向量的模
        norm = np.linalg.norm(self.pca_evectors, axis=0)
        # 归一化
        self.pca_evectors /= norm
        #计算权重矩阵W
        print('PCA is down')

        self.Y = np.matrix(self.pca_evectors.transpose())*np.matrix(self.imgs_training)


    def cal_LDA(self):
        #和PCA的计算方式很相似
        #axis是n，就是n维上被压缩成1
        #axis=0，输出矩阵是1行，求每一列的平均;axis=1，输出矩阵是一列，求每一行的平均
        mean_total = self.Y.mean(axis=1)

        #Sw是类内散列矩阵, Sb是类间散列矩阵
        Sb, Sw = np.zeros(shape=(self.cout_evalues, self.cout_evalues), dtype='complex128'), np.zeros(shape=(self.cout_evalues, self.cout_evalues), dtype='complex128')

        imgs_trained = 0
        #每次选一类人
        for id_face in range(1, self.N_identity+1):
            Yi = self.Y[:, imgs_trained:imgs_trained+self.N_training_img]
            #mean_class shape = [1*N*traning_img]
            mean_Class = Yi.mean(axis=1)
            Sw += np.dot((Yi-mean_Class), (Yi-mean_Class).transpose())
            Sb += self.N_training_img * np.dot((mean_Class-mean_total), (mean_Class-mean_total).transpose())
            imgs_trained += self.N_training_img

        #求特征值和特征向量
        self.evalues, self.evectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)
        sort_indices = self.evalues.argsort()[::-1]
        self.evalues = self.evalues[sort_indices]
        self.evectors = self.evectors[:, sort_indices]

        evalue_sum = sum(self.evalues[:])
        evalues_cout = 0
        evalues_energy = 0.0

        for evalue in self.evalues:
            evalues_cout += 1
            evalues_energy += evalue / evalue_sum

            if evalues_energy >= self.energy:
                break

        #得到主成分
        self.evalues = np.array(self.evalues[0:evalues_cout])
        #real返回实数部分
        self.lda_evectors = self.evectors[:, 0:evalues_cout].real

        #Wopt?W?
        self.Wopt = np.matrix(self.pca_evectors)*np.matrix(self.lda_evectors)

        #
        self.W = np.dot(self.Wopt.transpose(), self.imgs_training)
        print('LDA is down')

    def find_id_face(self, path_to_img):
        img = cv2.imread(path_to_img, 0)

        vector_img =np.array(img, dtype='float64').flatten()
        #先减去平均脸
        vector_img -= self.mean_total
        vector_img = np.reshape(vector_img, (self.S,1))

        #将人脸投影到特征向量上
        S = self.Wopt.transpose() * vector_img
        distance = self.W - S
        norms = np.linalg.norm(distance, axis=0)

        #找到距离最小的那一列
        closet_face_id = np.argmin(norms)
        return int(closet_face_id/self.N_training_img) + 1

    def verification(self):
        #使用5折或者10折交叉验证法
        print('> Evaluation %s faces started'%self.name_dataset)
        results_file = os.path.join('results', '%s_results_t.txt'%self.name_dataset)

        if not os.path.exists('results'):
            os.makedirs('results')
        test_cout = self.N_testing_img * self.N_identity
        cout_correct = 0.0
        with open(results_file, 'w') as f:
            for id_face in range(1, self.N_identity+1):
                #把没有用于训练的图像拿来做验证
                for id_test in range(1, 11):
                    if id_test not in self.ids_training[id_face-1]:
                        path_to_img = os.path.join(self.dir_identity, str(id_face),str(id_test)+self.type_img)
                        id_result = self.find_id_face(path_to_img)
                        result = (id_result == id_face)

                        if result == True:
                            cout_correct += 1
                            f.write('image: %s\nresult: correct\n\n'%(path_to_img))
                        else:
                            f.write('image: %s\nresult: wrong, got %2d\n\n'%(path_to_img, id_result))

            print('> Evaluating %s faces ended'%self.name_dataset)
            self.accuary = float(100. * cout_correct/test_cout)
            print('Correct: '+ str(self.accuary) + '%')
            f.write('Correct: %.2f\n'%(self.accuary))




def parse_args():
    parser = argparse.ArgumentParser(description='My Fisherface parameters')
    # general
    parser.add_argument('--name_dataset', default='CASIA-500', help='name of dataset, including att and CASIA-500')
    #parser.add_argument('--dir_data', default='/home/zx/dongxijia/Eigenfaces/Eigenfaces/att_faces', help='training and testing set directory')
    parser.add_argument('--dir_data', default='/home/zx/dongxijia/data_for_test_model_performance',
                        help='training and testing set directory')
    parser.add_argument('--size_img', default='112*112', help='image size')
    parser.add_argument('--N_identity', default=500, help='Numbers of dataset person')
    parser.add_argument('--N_training_img', default=9, help='Number of person for training in a indentity, rest is test')

    args = parser.parse_args()

    return args


if __name__ == '__main__':


    args = parse_args()
    accuarys = []
    since = time.time()
    for i in range(10):
        sample_fisherface = Fisherface(args.name_dataset, args.dir_data, args.size_img, args.N_identity, args.N_training_img)
        sample_fisherface.verification()
        accuarys.append(sample_fisherface.accuary)
    average_time = (time.time() - since) / 10
    print('Average time: %d s' % average_time)
    print('Number of training images %d'%args.N_training_img)
    print('Mean accury is %.2f'%np.mean(accuarys))
    #face_id = sample_fisherface.find_id_face('/home/zx/dongxijia/Eigenfaces/Eigenfaces/att_faces/1/1.pgm')
    #print('Seach_id %d' %face_id)

    
    
    
