import numpy as np

#for image read, 可用cv2代替
from scipy import ndimage

#for resize, 也可以用cv2代替
from scipy import misc

class SIFT():
    "Get the keypoints and descriptors using SIFT"
    def __init__(self):
        #s每一层金字塔的尺度数量
        self.s = 3
        self.s_pyr = self.s + 3
        self.N_DoG_s = self.s_pyr -1
        #N是金字塔的层数
        self.N_pyr_level = 4
        #每一层金字塔中的相邻尺度相差的比例因子k
        self.k = 2**(1.0 / self.s)

        #1.3约等于1.6/k, 1.6约等于2^0.67,就是一直×k
        self.kvec_level_1 = np.array([1.3, 1.6, 1.6*self.k, 1.6*(self.k**2), 1.6*(self.k**3), 1.6*(self.k**4)])
        #不同层的对应尺度之间相差了k**3倍
        self.kvec_level_2 = np.array(
            [1.6 * (self.k ** 2), 1.6 * (self.k ** 3), 1.6 * (self.k ** 4), 1.6 * (self.k ** 5), 1.6 * (self.k ** 6), 1.6 * (self.k ** 7)])
        self.kvec_level_3 = np.array([1.6 * (self.k ** 5), 1.6 * (self.k ** 6), 1.6 * (self.k ** 7), 1.6 * (self.k ** 8), 1.6 * (self.k ** 9), 1.6 * (self.k ** 10)])
        self.kvec_level_4 = np.array([1.6 * (self.k ** 8), 1.6 * (self.k ** 9), 1.6 * (self.k ** 10), 1.6 * (self.k ** 11), 1.6 * (self.k ** 12), 1.6 * (self.k ** 13)])

        #self.kvec_total
    def init_DoG_Gaussian_pyramids(self, path_img):
        X1_img = ndimage.imread(path_img, flatten=True)
        #X1_img = cv2.imread(path_img, 0)

        #---------------downsample images-----------------
        #不同的金字塔层，是不同的分辨率
        X2_img = misc.imresize(X1_img, 200, 'bilinear').astype(int)
        X1_img = misc.imresize(X2_img, 0.5, 'bilinear').astype(int)
        X05_img = misc.imresize(X1_img, 0.5, 'bilinear').astype(int)
        X025_img = misc.imresize(X05_img, 0.5, 'bilinear').astype(int)


        #---------------initialize Gaussian_ pyramids---------------
        #这里是有4层金字塔,shape的返回顺序是行和列，也就是高和宽,
        #这里的6是尺度个数，为了满足尺度变换的连续性，6 = s+3
        pyr_level_1 = np.zeros((X2_img.shape[0], X2_img.shape[1], self.s_pyr))
        pyr_level_2 = np.zeros((X1_img.shape[0], X1_img.shape[1], self.s_pyr))
        pyr_level_3 = np.zeros((X05_img.shape[0], X05_img.shape[1], self.s_pyr))
        pyr_level_4 = np.zeros((X025_img.shape[0], X025_img.shape[1], self.s_pyr))


        #---------------Constructing pyramids-------------
        print('Constructing Guassian pyramids...')

        for i in range(self.s_pyr):
            #为每一层高斯金字塔，中的不同尺度的图像赋值
            pyr_level_1[:, :, i] = ndimage.gaussian_filter(X2_img, self.kvec_level_1[i])
            pyr_level_2[:, :, i] = ndimage.gaussian_filter(X1_img, self.kvec_level_2[i])
            pyr_level_3[:, :, i] = ndimage.gaussian_filter(X05_img, self.kvec_level_3[i])
            pyr_level_4[:, :, i] = ndimage.gaussian_filter(X025_img, self.kvec_level_4[i])



        #--------------Constructing DoG pyramids------------
        print('Constructing DoG pyramids')
        #6个尺度个数，总共可以有5个尺度差分
        self.DoG_pyr_level_1 = np.zeros(shape=(X2_img.shape[0], X2_img.shape[1], self.N_DoG_s))
        self.DoG_pyr_level_2 = np.zeros(shape=(X1_img.shape[0], X1_img.shape[1], self.N_DoG_s))
        self.DoG_pyr_level_3 = np.zeros(shape=(X05_img.shape[0], X05_img.shape[1], self.N_DoG_s))
        self.DoG_pyr_level_4 = np.zeros(shape=(X025_img.shape[0], X025_img.shape[1], self.N_DoG_s))

        #赋值
        for i in range(0, self.N_DoG_s):
            self.DoG_pyr_level_1[:,:,i] = pyr_level_1[:, :, i+1] - pyr_level_1[:, :, i]
            self.DoG_pyr_level_2[:, :, i] = pyr_level_2[:, :, i + 1] - pyr_level_2[:, :, i]
            self.DoG_pyr_level_3[:, :, i] = pyr_level_3[:, :, i + 1] - pyr_level_3[:, :, i]
            self.DoG_pyr_level_4[:, :, i] = pyr_level_4[:, :, i + 1] - pyr_level_4[:, :, i]

    def find_extrema_points(self, threshold=1):
        print('First extrema detecting...')
        #5个查分尺度，除去边界的两个尺度，还剩3个尺度

    def find_extrema_points_in_octave(self, DoG_level_x, magrin):
        '''

        :param DoG_level_x: Dog_img
        :param magrin: margin of height and width
        :return:
        '''
        pass









