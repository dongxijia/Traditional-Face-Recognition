from BaseFace import BaseFace

import cv2
import numpy as np
import sklearn
#sklearn is the module of Hands-on Machine learning with scikit-learn and tensorflow

import os
import argparse

class MyHogSVM(BaseFace):
    "Face Recognition for face using HOG+SVM"
    def __init__(self, name_dataset, dir_data, size_img, N_identity, N_training_img):
        super(MyHogSVM, self).__init__(name_dataset, dir_data, size_img, N_identity, N_training_img)


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
                HOG_block = HOG_cells[i:i+height_size, j:j+width_size].flatten.astype(np.float32)
                #归一化特征，这一步可能有点问题, 因为特征是向量，所有后续的所有东西都是向量或矩阵
                #对光照和阴影获得更好的效果
                HOG_block /= np.sqrt(np.sum(abs(HOG_block)+1e-6))#防止0的平方根
                HOG_blocks[i, j] = HOG_block

        #把blocks的HOG特征串联起来，就是整个图像的HOG特征
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
        value_grd = cv2.addWeighted(dx, 0.5, dy, 0.5)
        #imshow('test_gra', value_grd)

        ###########collecting histogram of gradient#####################
        #hist is a 9D vector
        hist = np.zeros((self.binsize_cell), dtype=np.int32)
        #180 or 360, 效果有什么不同
        step = 180 // self.binsize_cell
        seq_bins = angle // step
        for seq_i, val in zip(seq_bins.flatten(), value_grd.flatten):
            hist[seq_i] += val
        return hist


    def cal_hog_img(self, img_np):
        '''

        :param img_np:
        :return: HOG of img
        '''
        #计算水平和竖直方向的梯度,Sobel函数求完导数后会有负值，还有会大于255的值,要用16位有符号数。

        #可以用opencv查看梯度value的图像
        width_cell, height_cell = 8, 8







def parse_args():
    parser = argparse.ArgumentParser(description='My Fisherface parameters')
    # general
    parser.add_argument('--name_dataset', default='att', help='name of dataset, including att and CASIA-500')
    parser.add_argument('--dir_data', default='/home/zx/dongxijia/Eigenfaces/Eigenfaces/att_faces', help='training and testing set directory')
    #parser.add_argument('--dir_data', default='/home/zx/dongxijia/data_for_test_model_performance',
    #                    help='training and testing set directory')
    parser.add_argument('--size_img', default='92*112', help='image size')
    parser.add_argument('--N_identity', default=40, help='Numbers of dataset person')
    parser.add_argument('--N_training_img', default=5, help='Number of person for training in a indentity, rest is test')

    args = parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Sample_HOG = MyHogSVM(args.name_dataset, args.dir_data, args.size_img, args.N_identity, args.N_training_img)
