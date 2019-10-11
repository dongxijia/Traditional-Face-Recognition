#coding=utf-8
import numpy as np
import cv2
import os

from functools import partial
import time
import progressbar
from multiprocessing import Pool


def _cal_feature(integral_img, extractors):
    features = []
    for extractor in extractors:
        features.append(extractor.cal_feature_one_haar(integral_img))
    return np.array(features)


type_haar_list = ['two_vertical', 'two_horizontal', 'three_vertical', 'three_horizontal', 'four']
#(1, 2) means(width, height)
size_haar_list = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]

class Haar_extractor():
    def __init__(self, type_haar, p_top_left, width_haar, height_haar, threshold=0.11, polarity=-1, weight=1):
        #位置也是Haar特征提取器的属性，因为1469个特征提取器不仅是用来提取特征的，也是用来分类的,后面需要选取这个这些特征提取器
        #那么包括位置的特征提取器，不能在类方法中计算变换位置的Haar特征
        self.type_haar = type_haar_list[type_haar]
        self.p_top_left = p_top_left

        self.width_haar, self.height_haar = width_haar, height_haar
        self.threshold_haar = threshold
        self.polarity_haar = polarity
        self.weight_haar = weight

    def cal_feature_one_haar(self, integral_img):
        '''

        :param integral_img:
        :param p_top_left:
        :param width:
        :param height:
        :return: a type haar feature of a region
        '''
        p_top_left = self.p_top_left
        width = self.width_haar
        height = self.height_haar


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
        #结果是1或是-1，haar特征的触发结果，暗示分类结果
        #加权投票
        return self.weight_haar*(1 if score < self.polarity_haar*self.threshold_haar else -1)


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

    def train_Adboost(self, num_classifiers=2):
        #这只是一层的Adboost，用两个特征，后面有很多层Adboost，每一层Adboost使用的特征越来越多
        #2->10->25->25->50->50->50, 前7层，一共38层Adaboost
        #所以级联，可以用较少的特征，排除前面不是人脸的部分
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

        #这里的hstack，对一维来说，其实就是相加, what's the meaning,weight应该是特征的权重
        weights = np.hstack((weights_pos, weights_neg))
        labels = np.hstack((np.ones(num_pos), np.ones(num_neg)*-1))

        #把图像转为积分图
        integral_img_faces = list(map(self.cal_integral_img, self.trainging_faces))
        integral_img_no_faces = list(map(self.cal_integral_img, self.trainging_no_faces))

        #integral_imgs_total = integral_img_faces+integral_img_no_faces

        #计算Haar特征, 分类器数量为2，也可以是Haar特征的数量
        #num_classifiers = 2

        extractors = self.create_haar_extractor(self.trainging_faces[0], min_feature_width, max_feature_width,
                                               min_feature_height, max_feature_height)
        num_extractor = len(extractors)

        #特征数量是1469
        votes = np.zeros(shape=(num_imgs_total, num_extractor))
        print(votes.shape)
        bar = progressbar.ProgressBar()
        #use as many as cpus
        since = time.time()
        pool = Pool(processes=None)
        votes_faces = pool.map(partial(_cal_feature, extractors=extractors), integral_img_faces)

        #如果想要顺序执行，chunksize=1，但是非常慢
        #for i in bar(range(num_pos)):
            #这个用法是对同一张图，并行计算Haar特征，所以不会引起乱序的问题

            #votes[i, :] = np.array(list(pool.map(partial(_cal_feature, integral_img=integral_img_faces[i]), extractors)))

        #不需要顺序执行，只要分开执行正例和反例就行了
        print('faces haar extract done...')
        #pool.close()
        #pool.join()
        #pool = Pool(processes=None)
        #bar = progressbar.ProgressBar()

        votes_no_faces = pool.map(partial(_cal_feature, extractors=extractors), integral_img_no_faces)

        # for i in bar(range(num_neg)):
        #     votes[i+num_pos, :] = np.array(list(pool.map(partial(_cal_feature, integral_img=integral_img_no_faces[i]),
        #                                                  extractors)))

        print('non-faces haar extract done...')
        pool.close()
        pool.join()

        time_fly = time.time() - since
        print('Using time %s'%time_fly)
        votes = np.array(votes_faces + votes_no_faces)
        print(votes.shape)

        print('Extract haar features done...')


        #分类器就是特征提取器，所以需要将特征提取器设为对象，也就是类
        classifiers = []

        print('Selecting classifiers...')
        bar = progressbar.ProgressBar()
        num_features = votes.shape[1]
        #单独拿出来的原因是后面需要remove
        feature_indexes = list(range(num_features))

        for _ in bar(range(num_classifiers)):
            #只需要找到num_classifiers个分类特征
            classification_errors = np.zeros(shape=len(feature_indexes))

            #归一化权重，权重是图像的分类权重
            weights /= np.sum(weights) * 1.0

            #基于分类损失选择最终的分类器，也就是最佳的特征,从1469个特征里找到一个最佳特征
            for idx_f in range(len(feature_indexes)):
                #计算每一个特征下的分类误差
                idx_f_actual = feature_indexes[idx_f]

                error = sum(map(lambda idx_img: 1*weights[idx_img] if labels[idx_img] != votes[idx_img, idx_f_actual] else 0 ,
                                range(num_imgs_total)))
                classification_errors[idx_f] = error

            min_idx_error = np.argmin(classification_errors)
            best_error = classification_errors[min_idx_error]
            best_idx_feature = feature_indexes[min_idx_error]
            best_extractor = extractors[best_idx_feature]
            #权重更新公式从哪来的
            feature_weight = 0.5 * np.log((1- best_error) / best_error)
            best_extractor.weight_haar = feature_weight

            #将特征提取器加入分类器中
            classifiers.append(best_extractor)

            #update image weight，也就是训练集的权重重新分布,图像权重更新公式？
            weights = np.array(list(map(lambda idx_img:weights[idx_img]*np.sqrt((1-best_error)/best_error
                                                                                ) if labels[idx_img]!= votes[idx_img, best_idx_feature] else
                                        weights[idx_img]*np.sqrt(best_error/(1-best_error)),
                                    range(num_imgs_total))
                               ))

            feature_indexes.remove(best_idx_feature)

        print('Train done...')
        self.classifiers = classifiers

    def test(self):
        integral_img_faces = list(map(self.cal_integral_img, self.test_faces))
        integral_img_no_faces = list(map(self.cal_integral_img, self.test_no_faces))

        print('Start testing...')
        correct_faces = 0
        correct_no_faces = 0
        for face in integral_img_faces:
            correct_faces += 1 if sum(_cal_feature(face,self.classifiers)) >= 0 else 0

        for no_face in integral_img_no_faces:
            correct_no_faces += 1 if sum(_cal_feature(no_face, self.classifiers)) < 0 else 0

        print('N of faces is %d\n correct is %d'%(len(integral_img_faces), correct_faces))

        print('N of no-faces is %d\n correct is %d'%(len(integral_img_no_faces), correct_no_faces))




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

    def create_haar_extractor(self, arr_img, min_feature_width=1, max_feature_width=-1, min_feature_height=1,
                         max_feature_height=-1):
        '''

        :param arr_img:
        :param min_feature_width: limit the retangle of haar feature
        :param max_feature_width:
        :param min_feature_height:
        :param max_feature_height:
        :return:
        '''
        width_img, height_img = arr_img.shape[1], arr_img.shape[0]
        # 积分直方图integral_img
        #integral_img = self.cal_integral_img(arr_img)
        extractor = []
        # 选择某个类型的Haar特征并初始化
        for i_type in range(len(type_haar_list)):

            size_haar = size_haar_list[i_type]
            width_haar, height_haar = size_haar[0], size_haar[1]

            # 得到缩放比例
            scale_w = int(width_img / width_haar)
            scale_h = int(height_img / height_haar)
            # 确定某个Haar特征
            if max_feature_width == -1:
                max_feature_width = width_haar * scale_w + 1
            if max_feature_height == -1:
                max_feature_height = height_haar * scale_h + 1
            left_w_haar = max(min_feature_width, width_haar)
            right_w_haar = min(max_feature_width, width_haar * scale_w + 1)
            left_h_haar = max(min_feature_height, height_haar)
            right_h_haar = min(max_feature_height, height_haar * scale_h + 1)
            # 可以限制Haar特征矩形框的范围，以减少计算量
            for w_haar in range(left_w_haar, right_w_haar, width_haar):
                for h_haar in range(left_h_haar, right_h_haar, height_haar):
                    # 对图像进行窗口滑动

                    # 注意这里要加1
                    for x in range(width_img - w_haar + 1):
                        for y in range(height_img - h_haar + 1):
                            # 添加特征抽取器

                            extractor.append(Haar_extractor(i_type, (x, y), w_haar, h_haar))
                            #extractor.append(Haar_extractor(i_type, (x, y), w_haar, h_haar, polarity=-1))
                            #原文中改变极性，添加另一个特征抽取器

            # print('N of haar features of module (%d, %d) is %d'%(self.width_haar,self.height_haar, cout))
        print('N of haar feature extractor of 5 module is %d'%(len(extractor)))
        return extractor


if __name__ == '__main__':

    # path_img = '/home/zx/dongxijia/Methods_faces/face_detector/1.png'
    # img = cv2.imread(path_img, 0)
    # img_20 = cv2.resize(img, (24, 24), interpolation=cv2.INTER_CUBIC)
    # sample_vj = VJ_detector()
    # sample_vj.create_haar_extractor(img_20)


    sample_vj = VJ_detector()
    sample_vj.train_Adboost(num_classifiers=2)
    #这是层数为1的Cascade
    #classifiers = sample_vj.classifiers
    #print(classifiers)
    sample_vj.test()