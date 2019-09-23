import os
import cv2

import numpy as np
import random

class BaseFace():
    '''
    Base class of Face
    '''
    def __init__(self,name_dataset,dir_data, size_img, N_identity, N_training_img):
        self.name_dataset = name_dataset
        self.dir_identity = dir_data
        self.width, self.height = int(size_img.split('*')[0]), int(size_img.split('*')[1])
        self.S = int(self.width) * int(self.height)
        self.N_identity = N_identity
        self.N_training_img = N_training_img
        self.N_total_training_img = self.N_identity * self.N_training_img
        self.N_testing_img = 10 - self.N_training_img
        self.init_imgs()


    def init_imgs(self):
        #Imgs_total  = np.empty(shape=(self.S, self.N_total_training_img),dtype='float64')
        if self.name_dataset == 'att':
            self.type_img = '.pgm'
        if self.name_dataset.split('-')[0] == 'CASIA':
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
