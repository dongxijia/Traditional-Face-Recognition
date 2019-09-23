from BaseFace import BaseFace

import cv2
import numpy as np
from sklearn import svm
import time
#sklearn is the module of Hands-on Machine learning with scikit-learn and tensorflow

import os
import argparse

class MyHogSVM(BaseFace):
    "Face Recognition for face using HOG+SVM"
    def __init__(self, name_dataset, dir_data, size_img, N_identity, N_training_img):
        super(MyHogSVM, self).__init__(name_dataset, dir_data, size_img, N_identity, N_training_img)


    def extract_HOG_img(self, img):
        HOG_cells = self.init_cell_img(img)
        HOG_blocks = self.combine_cells_to_block(HOG_cells)

        return HOG_blocks



    def init_cell_img(self, img, width_size=8, binsize=9):
        '''

        :param width_size: width and height of one cell
        :param binsize: number of orient
        :return: HOG of cells in a image
        '''
        self.width_size_cell = width_size
        self.binsize_cell = binsize
        height_size = width_size
        N_cells_height, N_cells_width = self.height // height_size, self.width // width_size
        #np.zeros可以生成任意大小的ndarray
        HOG_cells = np.zeros((N_cells_height, N_cells_width, binsize), dtype=np.int32)

        #cal histogram of cells
        for i in range(N_cells_height):
            #标记要计算HOG的点
            row_cell = height_size*i
            for j in range(N_cells_width):
                col_cell = width_size*j
                HOG_cells[i, j] = self.cal_histogram_cell(img[row_cell:row_cell+N_cells_width, col_cell:col_cell+N_cells_height])

        return HOG_cells

    def combine_cells_to_block(self, HOG_cells, width_size=2, height_size=2):
        assert (width_size == height_size)
        #block和cell一样都是堆叠的
        N_blocks_height, N_blocks_width = self.width_size_cell - width_size + 1, self.width_size_cell - height_size + 1
        self.width_size_block = width_size
        HOG_blocks = np.zeros((N_blocks_height, N_blocks_width, width_size*width_size*self.binsize_cell), dtype=np.float32)
        for i in range(N_blocks_height):#height 也就是行，秉承着先行后列的习惯，就先height后width了
            for j in range(N_blocks_width):
                #collection cells in a block
                #把4个cell的HOG特征串联成新的block特征，LBP+h其实就是把图像分成了多个block
                HOG_block = HOG_cells[i:i+height_size, j:j+width_size].flatten().astype(np.float32)
                #L1归一化特征，这一步可能有点问题, 因为特征是向量，所有后续的所有东西都是向量或矩阵
                #对光照和阴影获得更好的效果
                HOG_block /= np.sqrt(np.sum(abs(HOG_block))+1e-6)#防止0的平方根
                HOG_blocks[i, j] = HOG_block

        #把blocks的HOG特征串联起来，就是整个图像的HOG特征
        self.size_HOG_feature = HOG_blocks.size
        #print('size of HOG for img is %d'%self.size_HOG_feature)
        return HOG_blocks.flatten()







    def cal_histogram_cell(self, img_cell):
        '''

        :param img_np: imread(path, 0), a cell of  gray image
        :return: the histogram of a cell
        '''
        img_np = img_cell
        #############cal value and angle of gradient###################
        dx = cv2.Sobel(img_np, cv2.CV_16S, 1, 0)
        dy = cv2.Sobel(img_np, cv2.CV_16S, 0, 1)
        # s is a small number avoiding /0
        s = 1e-3
        angle = np.int32(np.arctan(dy / (dx + s)) / np.pi * 180) + 90  # 为什么要加90

        # 将梯度转回uint8
        dy = cv2.convertScaleAbs(dy)
        dx = cv2.convertScaleAbs(dx)
        # 计算梯度大小，结合水平梯度和竖直梯度
        value_grd = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
        #imshow('test_gra', value_grd)

        ###########collecting histogram of gradient#####################
        #hist is a 9D vector
        hist = np.zeros((self.binsize_cell), dtype=np.int32)
        #180 or 360, 效果有什么不同
        step = 180 // self.binsize_cell
        seq_bins = angle // step
        seq_bins = seq_bins.flatten()
        value_grd = value_grd.flatten()
        for seq_i, val in zip(seq_bins.flatten(), value_grd.flatten()):
            hist[seq_i] += val
        return hist


    def get_train_data(self):
        path_img = '/home/zx/dongxijia/CASIA-100/100/9.png'
        test_img = cv2.imread(path_img, 0)
        self.extract_HOG_img(test_img)
        HOG_features = np.zeros(shape=(self.N_total_training_img, self.size_HOG_feature), dtype=np.float64)
        class_indentity = np.zeros(shape=(self.N_total_training_img), dtype=np.int32)
        i_img = 0
        for id_identity in range(1, len(self.ids_training)+1):
            for id_img in self.ids_training[id_identity-1]:
                class_indentity[i_img] = id_identity
                img = self.imgs_training[:, i_img].reshape(self.height, -1)
                #img = self.imgs_training[:(id_identity-1)*self.N_training_img+id_img-1].reshape(self.height, -1)
                HOG_features[i_img] = self.extract_HOG_img(img)
                i_img += 1
        print('Get train data down...')

        return HOG_features, class_indentity


    def train_svm(self, x_train, y_train):
        print('Start train...')
        self.lin_clf = svm.LinearSVC()
        self.lin_clf.fit(x_train, y_train)
        print('Train down...')

    def pridict_identify(self, path_img):
        img = cv2.imread(path_img,  0)
        HOG_feature = self.extract_HOG_img(img)
        id_result = self.lin_clf.predict([HOG_feature])
        #print(id_result)
        return id_result


    def verification(self):
        print('Evaluation start..')
        results_file = os.path.join('results', 'HOGSVM_%s_results_t.txt' % self.name_dataset)

        if not os.path.exists('results'):
            os.makedirs('results')
        test_cout = self.N_testing_img * self.N_identity
        #cout_correct = 0.0
        ture_label = np.zeros(shape=(test_cout), dtype=np.int32)
        test_features = np.zeros(shape=(test_cout, self.size_HOG_feature), dtype=np.float32)

        i_img = 0
        with open(results_file, 'w') as f:
            for id_face in range(1, self.N_identity+1):
                #把没有用于训练的图像拿来做验证
                for id_test in range(1, 11):
                    if id_test not in self.ids_training[id_face-1]:
                        path_to_img = os.path.join(self.dir_identity, str(id_face),str(id_test)+self.type_img)
                        img = cv2.imread(path_to_img, 0)
                        ture_label[i_img] = id_face
                        HOG_feature = self.extract_HOG_img(img)
                        test_features[i_img] = HOG_feature
                        i_img += 1

            predict_label = self.lin_clf.predict(test_features)
            correct = (ture_label == predict_label)
            self.accuary = np.sum(correct)*1.0/test_cout
            print('Correct: %f'%(self.accuary))
            f.write('Correct: %f'%(self.accuary))
        print('Evaluation end...')
















def parse_args():
    parser = argparse.ArgumentParser(description='My HOG+SVM parameters')
    # general
    parser.add_argument('--name_dataset', default='CASIA-100', help='name of dataset, including att and CASIA-500')
    #parser.add_argument('--dir_data', default='/home/zx/dongxijia/Eigenfaces/Eigenfaces/att_faces', help='training and testing set directory')
    parser.add_argument('--dir_data', default='/home/zx/dongxijia/CASIA-100',
                        help='training and testing set directory')
    parser.add_argument('--size_img', default='112*112', help='image size')
    parser.add_argument('--N_identity', default=100, help='Numbers of dataset person')
    parser.add_argument('--N_training_img', default=6, help='Number of person for training in a indentity, rest is test')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    accuarys = []
    since = time.time()
    for i in range(10):

        Sample_HOG = MyHogSVM(args.name_dataset, args.dir_data, args.size_img, args.N_identity,
                                       args.N_training_img)
        x_train, y_train = Sample_HOG.get_train_data()
        Sample_HOG.train_svm(x_train, y_train)
        Sample_HOG.verification()
        accuarys.append(Sample_HOG.accuary)

    average_time = (time.time() - since) / 10
    print('Average time: %d s' % average_time)
    print('Number of training images %d' % args.N_training_img)
    print('Mean accury is %.4f' % np.mean(accuarys))
    '''
    for i in range(1, 41):
        id_result = Sample_HOG.pridict_identify('/home/zx/dongxijia/Eigenfaces/Eigenfaces/att_faces/%d/6.pgm'%i)
        print('Ture id is %d, predict is %d'%(i, id_result[0]))
    '''
