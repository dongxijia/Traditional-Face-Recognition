#coding=utf-8
import numpy as np
import cv2
import os

from functools import partial
import time
import progressbar
from multiprocessing import Pool


type_haar_list = ['two_vertical', 'two_horizontal', 'three_vertical', 'three_horizontal', 'four']
#(1, 2) means(width, height)
size_haar_list = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]

class VJ_detector():
    #'实现VJ人脸检测框架'
    def __init__(self):

        '''
        首先实现积分图
        然后计算Haar特征
        最后利用adaboost进行人脸检测
        '''
        #type = 0, 1, 2, 3, 4

        #self.type_haar = type_haar_list[type_haar]
        pass

    def train_Adboost(self):
        self.init_training_data()
        #using training data
        num_pos = len(self.trainging_faces)
        num_neg = len(self.trainging_no_faces)
        num_imgs_total = num_pos+num_neg
        h_img, w_img = self.trainging_faces[0].shape

        #设置Haar特征框大小
        min_feature_height = 8
        max_feature_height = 10
        min_feature_width = 8
        max_feature_width = 10

        #initial weights and labels for training set
        weights_pos = np.ones(num_pos) * 1.0 / (2* num_pos)
        weights_neg = np.ones(num_neg) * 1.0 / (2* num_neg)

        #这里的hstack，对一维来说，其实就是相加, what's the meaning
        weights = np.hstack((weights_pos, weights_neg))
        labels = np.hstack((np.ones(num_pos), np.ones(num_neg)*-1))

        #把图像转为积分图
        integral_img_faces = list(map(self.cal_integral_img, self.trainging_faces))
        integral_img_no_faces = list(map(self.cal_integral_img, self.trainging_no_faces))

        #integral_imgs_total = integral_img_faces+integral_img_no_faces

        #计算Haar特征, 分类器数量为2，也可以是Haar特征的数量
        num_classifiers = 2

        sample_vote = self.extract_haar_img(integral_img_faces[0], min_feature_width, max_feature_width, min_feature_height, max_feature_height)

        #特征数量是1469
        votes = np.zeros(shape=(num_imgs_total, len(sample_vote)))
        print(votes.shape)
        #bar = progressbar.ProgressBar()
        #use as many as cpus
        since = time.time()
        pool = Pool(processes=None)
        #results = []
        #如果想要顺序执行，chunksize=1，但是非常慢
        results_faces = list(pool.map(
            partial(self.extract_haar_img, min_feature_width=min_feature_width, max_feature_width=max_feature_width,
                    min_feature_height=min_feature_height, max_feature_height=max_feature_height), integral_img_faces
            ))
        #不需要顺序执行，只要分开执行正例和反例就行了
        #results = list(pool.map(partial(self.extract_haar_img, min_feature_width=min_feature_width, max_feature_width=max_feature_width, min_feature_height=min_feature_height, max_feature_height=max_feature_height), integral_imgs_total, chunksize=1))
        print('faces haar extract done...')
        #pool.close()
        #pool.join()
        #pool = Pool(processes=None)

        results_no_faces = list(pool.map(
            partial(self.extract_haar_img, min_feature_width=min_feature_width, max_feature_width=max_feature_width,
                    min_feature_height=min_feature_height, max_feature_height=max_feature_height), integral_img_no_faces
        ))
        print('non-faces haar extract done...')
        pool.close()
        pool.join()


        time_fly = time.time() - since
        print('Using time %s'%time_fly)
        votes = np.array(results_faces+results_no_faces)
        print(votes.shape)

        '''
        for i in bar(range(num_imgs_total)):
            if i % 500 == 0:
                print('Extract %d img'%i)

            #votes[i, :] = np.array(self.extract_haar_img(integral_imgs_total[i], min_feature_width, max_feature_width, min_feature_height, max_feature_height))
            #而我这种写法是不能多进程的,除非把不同的图像用于多进程
            #原来的多进程是把不同的features拆分开

            results.append(pool.apply_async(self.extract_haar_img, (
                integral_imgs_total[i], min_feature_width, max_feature_width, min_feature_height, max_feature_height)))

            #result = pool.map(self.extract_haar_img, (integral_imgs_total[i], min_feature_width, max_feature_width, min_feature_height, max_feature_height))
            #votes[i, :] = result.get()

        
        count = 0
        for res in results:
            votes[count, :] = res.get()
            count += 1
            if count % 100 == 0:
                print('Actuall Extract %d img'%count)
        '''

        print('Extract haar features done...')

        classifiers = []






    def init_training_data(self, path_train='/home/zx/dongxijia/Viola-Jones/data/trainset', path_test='/home/zx/dongxijia/Viola-Jones/data/testset'):

        def load_images(path):
            imgs = []
            for _file in os.listdir(path):
                if _file.endswith('.png'):
                    img_arr = cv2.imread(os.path.join(path,_file), 0).astype(np.float64)
                    #归一化
                    img_arr /= img_arr.max()
                    imgs.append(img_arr)
            return imgs

        self.trainging_faces = load_images(os.path.join(path_train, 'faces'))
        self.trainging_no_faces = load_images(os.path.join(path_train, 'non-faces'))
        self.test_faces = load_images(os.path.join(path_test, 'faces'))
        self.test_no_faces = load_images(os.path.join(path_test, 'non-faces'))

        print('Loading imgs done...')


    def init_haar_detector(self, type_haar, threshold=0, polarity=1, weight=1):
        self.type_haar = type_haar_list[type_haar]
        size_haar = size_haar_list[type_haar]
        self.width_haar, self.height_haar = size_haar[0], size_haar[1]
        self.threshold_haar = threshold
        self.polarity_haar = polarity
        self.weight_haar = weight

    def cal_feature_one_haar(self, integral_img, p_top_left, width, height):
        '''

        :param integral_img:
        :param p_top_left:
        :param width:
        :param height:
        :return: a type haar feature of a region
        '''
        p_bottom_right = (p_top_left[0]+width, p_top_left[1]+height)
        score = 0
        if self.type_haar == 'two_vertical':
            first = self.sum_region(integral_img, p_top_left, (p_top_left[0]+width, p_top_left[1]+height//2))
            second = self.sum_region(integral_img, (p_top_left[0], p_top_left[1]+height//2), p_bottom_right)
            score = first-second
        elif self.type_haar == 'two_horizontal':
            first = self.sum_region(integral_img, p_top_left, (p_top_left[0]+width//2, p_top_left[1]+height))
            second = self.sum_region(integral_img, (p_top_left[0]+width//2, p_top_left[1]), p_bottom_right)
            score = first - second
        elif self.type_haar == 'three_vertical':
            first = self.sum_region(integral_img, p_top_left, (p_top_left[0]+width, p_top_left[1]+height//3))
            second = self.sum_region(integral_img, (p_top_left[0], p_top_left[1]+height//3), (p_top_left[0]+width, p_top_left[1]+height//3*2))
            third = self.sum_region(integral_img, (p_top_left[0], p_top_left[1]+height//3*2), p_bottom_right)
            #third前面没有*2，会有什么影响呢
            score = first-second+third
        elif self.type_haar == 'three_horizontal':
            first = self.sum_region(integral_img, p_top_left, (p_top_left[0]+width//3, p_top_left[1]+height))
            second = self.sum_region(integral_img, (p_top_left[0]+width//3, p_top_left[1]), (p_top_left[0]+width//3*2, p_top_left[1]+height))
            third = self.sum_region(integral_img, (p_top_left[0]+width//3*2, p_top_left[1]), p_bottom_right)
            score = first-second+third
        elif self.type_haar == 'four':
            first = self.sum_region(integral_img, p_top_left, (p_top_left[0]+width//2, p_top_left[1]+height//2))
            second = self.sum_region(integral_img, (p_top_left[0]+width//2, p_top_left[1]), (p_top_left[0]+width//2, p_top_left[1]+height//2))
            third = self.sum_region(integral_img, (p_top_left[0], p_top_left[1]+height//2), (p_top_left[0]+width//2, p_top_left[1]+height))
            fourth = self.sum_region(integral_img, (p_top_left[0]+width//2, p_top_left[1]+height//2), p_bottom_right)
            score = first-second-third+fourth

        #vote

        return self.weight_haar*(1 if score < self.polarity_haar*self.threshold_haar else -1)

    def extract_haar_img(self, arr_img, min_feature_width=1, max_feature_width=-1, min_feature_height=1, max_feature_height=-1):
        '''

        :param arr_img:
        :param min_feature_width: limit the retangle of haar feature
        :param max_feature_width:
        :param min_feature_height:
        :param max_feature_height:
        :return:
        '''
        width_img, height_img = arr_img.shape[1], arr_img.shape[0]
        #积分直方图integral_img
        integral_img = self.cal_integral_img(arr_img)
        features = []
        #选择某个类型的Haar特征并初始化
        for i_type in range(len(type_haar_list)):
            self.init_haar_detector(i_type)
            #得到缩放比例
            scale_w = int(width_img / self.weight_haar)
            scale_h = int(height_img / self.height_haar)
            #确定某个Haar特征
            cout = 0
            if max_feature_width == -1:
                max_feature_width = self.height_haar*scale_h+1
            if max_feature_height == -1:
                max_feature_height = self.height_haar*scale_h+1
            left_w_haar = max(min_feature_width, self.width_haar)
            right_w_haar = min(max_feature_width, self.width_haar*scale_w+1)
            left_h_haar = max(min_feature_height, self.height_haar)
            right_h_haar = min(max_feature_height, self.height_haar*scale_h+1)
            #可以限制Haar特征矩形框的范围，以减少计算量
            for w_haar in range(left_w_haar, right_w_haar, self.width_haar):
                for h_haar in range(left_h_haar, right_h_haar, self.height_haar):
                    #对图像进行窗口滑动
                    
                    #注意这里要加1
                    for x in range(width_img-w_haar+1):
                        for y in range(height_img-h_haar+1):
                            #计算Haar特征的响应值,也就是分数
                            score_0 = self.cal_feature_one_haar(integral_img, (x, y), w_haar, h_haar)
                            #self.polarity_haar *= -1
                            #score_1 = self.cal_feature_one_haar(integral_img, (x, y), w_haar, h_haar)

                            features.append(score_0)
                            cout += 1
                            #features.append(score_1)
            #print('N of haar features of module (%d, %d) is %d'%(self.width_haar,self.height_haar, cout))
        #print('N of haar features of 5 module is %d'%(len(features)))
        return features





    def cal_integral_img(self, arr_img):
        '''
        row_sum(x,y)存放x行的像素点（包括(x, y)点），intetral_img(x+1, y+1)存放的是原图中(x, y)的y列的积分图
        :param arr_img: (H, w) array of img
        :return: the integral of img
        '''

        row_sum = np.zeros(shape=arr_img.shape)
        #额外多一行和多一列，来解决第一行和第一列的问题
        integral_img = np.zeros(shape=(arr_img.shape[0]+1, arr_img.shape[1]+1))
        for y in range(arr_img.shape[0]):
            for x in range(arr_img.shape[1]):
                row_sum[x, y] = row_sum[x-1, y] + arr_img[x, y]
                integral_img[x+1, y+1] = integral_img[x+1, y+1-1] + row_sum[x, y]

        return integral_img

    def sum_region(self, integral_img, p_top_left, p_bottom_right):
        '''

        :param integral_img:  array integral of img
        :param p_top_left: point(x, y) of top_left
        :param p_bottom_right: point(x, y) of bottom_right
        :return: sum of region where arrouded by two points
        '''

        top_left = (p_top_left[1], p_top_left[0])
        bottom_right = (p_bottom_right[1], p_bottom_right[0])
        if top_left == bottom_right:
            return integral_img[top_left]

        top_right = (bottom_right[0], top_left[1])
        bottom_left = (top_left[0], bottom_right[1])

        return integral_img[bottom_right]-integral_img[top_right] - integral_img[bottom_left]+integral_img[top_left]


if __name__ == '__main__':
    '''
    path_img = '/home/zx/dongxijia/Methods_faces/face_detector/1.png'
    img = cv2.imread(path_img, 0)
    img_20 = cv2.resize(img, (24, 24), interpolation=cv2.INTER_CUBIC)
    sample_vj = VJ_detector()
    sample_vj.extract_haar_img(img_20)
    '''

    sample_vj = VJ_detector()
    #sample_vj.init_training_data()
    sample_vj.train_Adboost()
