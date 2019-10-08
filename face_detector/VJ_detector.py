import numpy as np
import cv2


type_haar_list = ['two_vertical', 'two_horizontal', 'three_vertical', 'three_horizontal', 'four']
#(1, 2) means(width, height)
size_haar_list = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]

class VJ_detector():
    '实现VJ人脸检测框架'
    def __init__(self):

        '''
        首先实现积分图
        然后计算Haar特征
        最后利用adaboost进行人脸检测
        '''
        #type = 0, 1, 2, 3, 4

        #self.type_haar = type_haar_list[type_haar]
        pass

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

    def extract_haar_img(self, arr_img):
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
            for w_haar in range(self.width_haar, self.width_haar*scale_w+1, self.width_haar):
                for h_haar in range(self.height_haar, self.height_haar*scale_h+1, self.height_haar):
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
            print('N of haar features of module (%d, %d) is %d'%(self.width_haar,self.height_haar, cout))
        print('N of haar features of 5 module is %d'%(len(features)))
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
    path_img = '/home/zx/dongxijia/Methods_faces/face_detector/1.png'
    img = cv2.imread(path_img, 0)
    img_20 = cv2.resize(img, (24, 24), interpolation=cv2.INTER_CUBIC)
    sample_vj = VJ_detector()
    sample_vj.extract_haar_img(img_20)
