import numpy as np

#for image read, 可用cv2代替
from scipy import ndimage

#for resize, 也可以用cv2代替
from scipy import misc

from scipy import stats

class SIFT():
    "Get the keypoints and descriptors using SIFT"
    def __init__(self, s=3, N_pyr_level=4, threshold=5):
        #s每一层金字塔的尺度数量
        self.s = 3
        self.s_pyr = self.s + 3
        self.N_DoG_s = self.s_pyr -1
        #N_DoG_s是差分金字塔的尺度数量
        self.N_scales_extrema_p = self.N_DoG_s -2
        #N_scales_extrema_p是特征点所有尺度的数量

        self.N_pyr_level = 4
        #N_pyr_level是金字塔的层数
        #每一层金字塔中的相邻尺度相差的比例因子k
        self.k = 2**(1.0 / self.s)

        #用于筛选极值点的阈值
        self.threshold = 5

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
        self.X2_img = misc.imresize(X1_img, 200, 'bilinear').astype(int)
        self.X1_img = misc.imresize(self.X2_img, 0.5, 'bilinear').astype(int)
        self.X05_img = misc.imresize(self.X1_img, 0.5, 'bilinear').astype(int)
        self.X025_img = misc.imresize(self.X05_img, 0.5, 'bilinear').astype(int)


        #---------------initialize Gaussian_ pyramids---------------
        #这里是有4层金字塔,shape的返回顺序是行和列，也就是高和宽,
        #这里的6是尺度个数，为了满足尺度变换的连续性，6 = s+3
        self.pyr_level_1 = np.zeros((self.X2_img.shape[0], self.X2_img.shape[1], self.s_pyr))
        self.pyr_level_2 = np.zeros((self.X1_img.shape[0], self.X1_img.shape[1], self.s_pyr))
        self.pyr_level_3 = np.zeros((self.X05_img.shape[0], self.X05_img.shape[1], self.s_pyr))
        self.pyr_level_4 = np.zeros((self.X025_img.shape[0], self.X025_img.shape[1], self.s_pyr))


        #---------------Constructing pyramids-------------
        print('Constructing Guassian pyramids...')

        for i in range(self.s_pyr):
            #为每一层高斯金字塔，中的不同尺度的图像赋值
            self.pyr_level_1[:, :, i] = ndimage.gaussian_filter(self.X2_img, self.kvec_level_1[i])
            self.pyr_level_2[:, :, i] = ndimage.gaussian_filter(self.X1_img, self.kvec_level_2[i])
            self.pyr_level_3[:, :, i] = ndimage.gaussian_filter(self.X05_img, self.kvec_level_3[i])
            self.pyr_level_4[:, :, i] = ndimage.gaussian_filter(self.X025_img, self.kvec_level_4[i])



        #--------------Constructing DoG pyramids------------
        print('Constructing DoG pyramids')
        #6个尺度个数，总共可以有5个尺度差分
        self.DoG_pyr_level_1 = np.zeros(shape=(self.X2_img.shape[0], self.X2_img.shape[1], self.N_DoG_s))
        self.DoG_pyr_level_2 = np.zeros(shape=(self.X1_img.shape[0], self.X1_img.shape[1], self.N_DoG_s))
        self.DoG_pyr_level_3 = np.zeros(shape=(self.X05_img.shape[0], self.X05_img.shape[1], self.N_DoG_s))
        self.DoG_pyr_level_4 = np.zeros(shape=(self.X025_img.shape[0], self.X025_img.shape[1], self.N_DoG_s))

        #赋值
        for i in range(0, self.N_DoG_s):
            self.DoG_pyr_level_1[:,:,i] = self.pyr_level_1[:, :, i+1] - self.pyr_level_1[:, :, i]
            self.DoG_pyr_level_2[:, :, i] = self.pyr_level_2[:, :, i + 1] - self.pyr_level_2[:, :, i]
            self.DoG_pyr_level_3[:, :, i] = self.pyr_level_3[:, :, i + 1] - self.pyr_level_3[:, :, i]
            self.DoG_pyr_level_4[:, :, i] = self.pyr_level_4[:, :, i + 1] - self.pyr_level_4[:, :, i]

    def find_extrema_points(self, threshold=5):
        print('First extrema detecting...')
        #5个差分尺度，除去边界的两个尺度，还剩3个尺度


        #np.zeros(shape=(DoG_level_x.shape[0], DoG_level_x.shape[1], self.N_DoG_s-2), 只有1,2,3这三个尺度上存在特征点
        #初始化极值点所在图像
        self.points_extrema_level_1 = self.find_extrema_points_in_octave(self.DoG_pyr_level_1, 80, threshold)
        self.points_extrema_level_2 = self.find_extrema_points_in_octave(self.DoG_pyr_level_2, 40, threshold)
        self.points_extrema_level_3 = self.find_extrema_points_in_octave(self.DoG_pyr_level_3, 20, threshold)
        self.points_extrema_level_4 = self.find_extrema_points_in_octave(self.DoG_pyr_level_4, 10, threshold)

        print('Get extrema detected down...')

        print('Number of extrema in first octave: %d' %np.sum(self.points_extrema_level_1))
        print('Number of extrema in second octave: %d' % np.sum(self.points_extrema_level_2))
        print('Number of extrema in third  octave: %d' % np.sum(self.points_extrema_level_3))
        print('Number of extrema in fourth octave: %d' % np.sum(self.points_extrema_level_4))

    def find_extrema_points_in_octave(self, DoG_level_x, margin, threshold):
        '''
        :param DoG_level_x: Dog_img
        :param magrin: margin of height and width
        :return:
        '''

        points_extrema_level_x = np.zeros(shape=(DoG_level_x.shape[0], DoG_level_x.shape[1], self.N_DoG_s-2))

        for i_scale in range(1, self.N_DoG_s-1):
            #只有1,2,3这三个尺度有相邻尺度域
            for y in range(margin, DoG_level_x.shape[0]-margin):
                for x in range(margin, DoG_level_x.shape[1]-margin):
                    #过滤
                    if np.absolute(DoG_level_x[y, x, i_scale]) < threshold:
                        continue

                    maxbool = (DoG_level_x[y, x, i_scale] > 0)
                    minbool = (DoG_level_x[y, x, i_scale] < 0)

                    #---------------判断是否领域极值-----------------------
                    for di in range(-1, 2):
                        #附近的3个邻域，除自己之外
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                if di == 0 and dy == 0 and dx == 0:
                                    continue

                                #只要中间某次没有大于邻域值， 就不是空间极值
                                maxbool = maxbool and (DoG_level_x[y, x, i_scale] > DoG_level_x[y+dy, x+dx, i_scale+di])
                                minbool = minbool and (DoG_level_x[y, x, i_scale] < DoG_level_x[y+dy, x+dx, i_scale+di])
                                if not maxbool and not minbool:
                                    break
                            if not maxbool and not minbool:
                                break
                        if not maxbool and not minbool:
                            break

                    #过滤掉不稳定的边缘响应点
                    if maxbool or minbool:
                        dx = (DoG_level_x[y, x + 1, i_scale] - DoG_level_x[y, x - 1, i_scale]) * 0.5 / 255
                        dy = (DoG_level_x[y + 1, x, i_scale] - DoG_level_x[y - 1, x, i_scale]) * 0.5 / 255
                        ds = (DoG_level_x[y, x, i_scale + 1] - DoG_level_x[y, x, i_scale - 1]) * 0.5 / 255
                        dxx = (DoG_level_x[y, x + 1, i_scale] + DoG_level_x[y, x - 1, i_scale] - 2 * DoG_level_x[
                            y, x, i_scale]) * 1.0 / 255
                        dyy = (DoG_level_x[y + 1, x, i_scale] + DoG_level_x[y - 1, x, i_scale] - 2 * DoG_level_x[
                            y, x, i_scale]) * 1.0 / 255
                        dss = (DoG_level_x[y, x, i_scale + 1] + DoG_level_x[y, x, i_scale - 1] - 2 * DoG_level_x[
                            y, x, i_scale]) * 1.0 / 255
                        dxy = (DoG_level_x[y + 1, x + 1, i_scale] - DoG_level_x[y + 1, x - 1, i_scale] - DoG_level_x[
                            y - 1, x + 1, i_scale] + DoG_level_x[y - 1, x - 1, i_scale]) * 0.25 / 255
                        dxs = (DoG_level_x[y, x + 1, i_scale + 1] - DoG_level_x[y, x - 1, i_scale + 1] - DoG_level_x[
                            y, x + 1, i_scale - 1] + DoG_level_x[y, x - 1, i_scale - 1]) * 0.25 / 255
                        dys = (DoG_level_x[y + 1, x, i_scale + 1] - DoG_level_x[y - 1, x, i_scale + 1] - DoG_level_x[
                            y + 1, x, i_scale - 1] +DoG_level_x[y - 1, x, i_scale - 1]) * 0.25 / 255

                        dD = np.matrix([[dx], [dy], [ds]])
                        H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
                        x_hat = np.linalg.lstsq(H, dD)[0]
                        D_x_hat = DoG_level_x[y, x, i_scale] + 0.5 * np.dot(dD.transpose(), x_hat)

                        r = 10.0
                        if ((((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2))) and (
                                np.absolute(x_hat[0]) < 0.5) and (np.absolute(x_hat[1]) < 0.5) and (
                                np.absolute(x_hat[2]) < 0.5) and (np.absolute(D_x_hat) > 0.03):

                            points_extrema_level_x[y, x, i_scale - 1] = 1

        return points_extrema_level_x


    def cal_keypoints_orientations(self):

        #--------------------cal neborhoods mag and ori-----------------------
        #计算梯度和直方图用的是原图？而不是尺度空间中的图像，原代码错了，应该是尺度空间
        '''
        mag_scaled_img_level_1 = np.zeros(shape=(self.X2_img.shape[0], self.X2_img.shape[1], self.N_scales_extrema_p))
        mag_scaled_img_level_2 = np.zeros(shape=(self.X1_img.shape[0], self.X1_img.shape[1], self.N_scales_extrema_p))
        mag_scaled_img_level_3 = np.zeros(shape=(self.X05_img.shape[0], self.X05_img.shape[1], self.N_scales_extrema_p))
        mag_scaled_img_level_4 = np.zeros(shape=(self.X025_img.shape[0], self.X025_img.shape[1], self.N_scales_extrema_p))


        ori_scaled_img_level_1 = np.zeros(shape=(self.X2_img.shape[0], self.X2_img.shape[1], self.N_scales_extrema_p))
        ori_scaled_img_level_2 = np.zeros(shape=(self.X1_img.shape[0], self.X1_img.shape[1], self.N_scales_extrema_p))
        ori_scaled_img_level_3 = np.zeros(shape=(self.X05_img.shape[0], self.X05_img.shape[1], self.N_scales_extrema_p))
        ori_scaled_img_level_4 = np.zeros(
            shape=(self.X025_img.shape[0], self.X025_img.shape[1], self.N_scales_extrema_p))
        '''
        self.mag_scaled_img_level_1, self.ori_scaled_img_level_1 = self.cal_nebors_grid(self.pyr_level_1)
        self.mag_scaled_img_level_2, self.ori_scaled_img_level_2 = self.cal_nebors_grid(self.pyr_level_2)
        self.mag_scaled_img_level_3, self.ori_scaled_img_level_3 = self.cal_nebors_grid(self.pyr_level_3)
        self.mag_scaled_img_level_4, self.ori_scaled_img_level_4 = self.cal_nebors_grid(self.pyr_level_4)




        #-----------------------cal keypoint orientations------------------

        print('calculationg keypoint orientations...')

        self.kvectotal = np.zeros(shape=12)
        for i in range(0, 12):
            #和后面的mag_pyr_total是一一对应，也是12张图像
            self.kvectotal[i] = 1.6*(self.k**i)

        sum_extr = np.sum(self.points_extrema_level_1) + np.sum(self.points_extrema_level_2) + np.sum(
            self.points_extrema_level_3) + np.sum(self.points_extrema_level_4)

        #4 contains x, y, scale, main_ori
        self.keypoints = np.zeros(shape=(int(sum_extr), 4))

        self.count_extr = 0

        self._cal_points_ori_level_x(self.points_extrema_level_1, self.pyr_level_1, 80, self.mag_scaled_img_level_1, self.ori_scaled_img_level_1)
        self._cal_points_ori_level_x(self.points_extrema_level_2, self.pyr_level_2, 40, self.mag_scaled_img_level_2, self.ori_scaled_img_level_2)
        self._cal_points_ori_level_x(self.points_extrema_level_3, self.pyr_level_3, 20, self.mag_scaled_img_level_3, self.ori_scaled_img_level_3)
        self._cal_points_ori_level_x(self.points_extrema_level_4, self.pyr_level_4, 10, self.mag_scaled_img_level_4, self.ori_scaled_img_level_4)

        assert self.count_extr == sum_extr






    def cal_nebors_grid(self, pyr_level_x):

        mag_scaled_img_level_x = np.zeros(shape=(pyr_level_x.shape[0], pyr_level_x.shape[1], self.N_scales_extrema_p))
        ori_scaled_img_level_x = np.zeros(shape=(pyr_level_x.shape[0], pyr_level_x.shape[1], self.N_scales_extrema_p))

        #计算特征点在高斯尺度空间,1,2,3中附近像素点的梯度大小和方向
        for i in range(0, 3):
            i_scale = i + 1
            for y in range(1, mag_scaled_img_level_x.shape[0]-1):
                for x in range(1, mag_scaled_img_level_x.shape[1]-1):
                    gri_y = pyr_level_x[y+1, x, i_scale] - pyr_level_x[y-1, x, i_scale]
                    gri_x = pyr_level_x[y, x+1, i_scale] - pyr_level_x[y, x-1, i_scale]
                    mag_scaled_img_level_x[y, x, i] =np.sqrt((gri_x)**2+(gri_y)**2)
                    #限定方向的值从0-35
                    ori_scaled_img_level_x[y, x, i] = (36/(2*np.pi))*(np.pi+ np.arctan2(gri_x, gri_y))

        return mag_scaled_img_level_x, ori_scaled_img_level_x



    def _cal_points_ori_level_x(self, points_extrema_level_x, pyr_level_x, margin, mag_scaled_img_level_x, ori_scaled_img_level_x):

        for i in range(0, 3):
            i_scale = i+1
            for y in range(margin, pyr_level_x.shape[0] - margin):
                for x in range(margin, pyr_level_x.shape[1]-margin):
                    if points_extrema_level_x[y, x, i] == 1:
                        #means the points is a keypoint
                        #高斯多元分布,
                        gaussian_window = stats.multivariate_normal(mean=[y ,x], cov=((1.5*self.kvectotal[i])**2))
                        #np.floor means int, two_sd means 半径r, but 3*1.5*scale
                        two_sd = np.floor(2*1.5*self.kvectotal[i])
                        orient_hist = np.zeros([36, 1])


                        #在圆弧形中的区域图像,梯度的值是使用之前的计算结果,统计梯度方向直方图，找到主方向
                        for dy in range(int(-1*two_sd*2), int(two_sd*2)+1):

                            dxlim = int((((two_sd*2)**2)-(np.absolute(dy)**2)) **0.5)
                            for dx in range(-1*dxlim, dxlim+1):
                                if y + dy < 0 or y+dy > pyr_level_x.shape[0]-1 or x+dx < 0 or x+dx > pyr_level_x.shape[1]-1:
                                    continue

                                #梯度的大小按照离关键点的远近分配不同大小的权重
                                weight = mag_scaled_img_level_x[y+dy, x+dx, i]*gaussian_window.pdf([y+dy, x+dx])
                                #将方向限制在0-35中某个数
                                bin_idx = np.clip(np.floor(ori_scaled_img_level_x[y+dy,x+dx, i]), 0, 35)
                                orient_hist[int(bin_idx)] += weight

                        #-------------找到主方向--------------------
                        max_val = np.amax(orient_hist)
                        max_idx = np.argmax(orient_hist)

                        #why y *= 0.5 , x*=0.5
                        self.keypoints[self.count_extr, :] = np.array([int(y*0.5), int(x*0.5), self.kvectotal[i], max_idx])

                        self.count_extr += 1

                        orient_hist[max_idx] = 0
                        # Get the sub-max orientation
                        sub_max_val = np.amax(orient_hist)

                        if sub_max_val >= 0.8*max_val:
                            sub_max_idx = np.argmax(orient_hist)
                            #将辅助方向也作为一个特征点
                            self.keypoints = np.append(self.keypoints,np.array([[int(y*0.5), int(x*0.5), self.kvectotal[i], sub_max_idx]]), axis=0 )

    def cal_feature_base_keypoints(self):
        #把所有的高斯金字塔上的存在特征点的图像梯度的分辨率都转为原图的尺寸,有3个特征点
        self.mag_pyr_total = np.zeros(shape=(self.X1_img.shape[0], self.X1_img.shape[1], 12))
        self.ori_pyr_total = np.zeros(shape=(self.X1_img.shape[0], self.X1_img.shape[1], 12))

        self._init_mag_pyr_total(self.mag_scaled_img_level_1, self.ori_scaled_img_level_1, 0)
        self._init_mag_pyr_total(self.mag_scaled_img_level_2, self.ori_scaled_img_level_2, 3)
        self._init_mag_pyr_total(self.mag_scaled_img_level_3, self.ori_scaled_img_level_3, 6)
        self._init_mag_pyr_total(self.mag_scaled_img_level_4, self.ori_scaled_img_level_4, 9)

        #------------------extracting the SIFT feature ------------------
        print('extracting the SIFT feature.')

        features_SIFT = np.zeros(shape=(self.keypoints.shape[0], 128))

        for idx_keypoint in range(0, self.keypoints.shape[0]):
            for y in range(-8, 8):
                for x in range(-8, 8):
                    theta = 10*self.keypoints[idx_keypoint, 3]*np.pi/180.0
                    xrot = np.round((np.cos(theta)*x) - (np.sin(theta)*y))
                    yrot = np.round((np.sin(theta)*x) + (np.cos(theta)*y))
                    id_scale = np.argwhere(self.kvectotal == self.keypoints[idx_keypoint, 2])[0][0]
                    x0 = self.keypoints[idx_keypoint, 1]
                    y0 = self.keypoints[idx_keypoint, 0]
                    gaussian_window = stats.multivariate_normal(mean=[x0, y0], cov=8)
                    weight = self.mag_pyr_total[int(y0+yrot), int(x0+xrot), id_scale] * gaussian_window.pdf([int(y0+yrot), int(x0+xrot)])
                    #angle 需要减去旋转的角度
                    angle = self.ori_pyr_total[int(y0+yrot), int(x0+xrot), id_scale] - self.keypoints[idx_keypoint, 3]
                    if angle < 0:
                        angle += 36

                    #这次只有8个特征向量了
                    idx_bin = np.clip(int((8.0/36) * angle), 0, 7).astype(int)
                    #32 means? 8 means? 128维特征是16个*8特征flatten得到的, 0-7, 8-15...
                    features_SIFT[idx_keypoint, 32*int((x+8)/4)+8*int((y+8)/4)+int(idx_bin)] += weight

            #特征归一化
            features_SIFT[idx_keypoint, :] /= np.linalg.norm(features_SIFT[idx_keypoint, :])
            #每个特征限制到0-0.2之内，为什么？
            features_SIFT[idx_keypoint, :] = np.clip(features_SIFT[idx_keypoint, :], 0, 0.2)
            features_SIFT[idx_keypoint, :] /= np.linalg.norm(features_SIFT[idx_keypoint, :])

        print('Feature extracting down')
        return [self.keypoints, features_SIFT]

    def _init_mag_pyr_total(self, mag_scaled_img_level_x , ori_scaled_img_level_x, location):

        for i in range(0, 3):
            #max_mag = np.amax(mag_scaled_img_level_x[:, :, i])
            self.mag_pyr_total[:, :, i+location] = misc.imresize(mag_scaled_img_level_x[:, :, i], (self.mag_pyr_total.shape[0], self.mag_pyr_total.shape[1]), 'bilinear').astype(float)
            #self.mag_pyr_total[:, :, i+location] = (max_mag/np.amax(self.mag_pyr_total[:,:,i+location]))*self.mag_pyr_total[:,:,i+location]
            self.ori_pyr_total[:, :, i+location] = misc.imresize(ori_scaled_img_level_x[:, :, i], (self.ori_pyr_total.shape[0], self.ori_pyr_total.shape[1]), 'bilinear').astype(int)
            #self.ori_pyr_total[:, :, i+location] = ((36.0/np.amax(self.ori_pyr_total[:, :, i+location])) * self.ori_pyr_total[:, :, i+location]).astype(int)






if __name__ == '__main__':
    path_img = '/home/zx/dongxijia/Eigenfaces/Eigenfaces/att_faces/1/6.pgm'
    sample_sift = SIFT()
    sample_sift.init_DoG_Gaussian_pyramids(path_img)
    sample_sift.find_extrema_points(threshold=1)
    sample_sift.cal_keypoints_orientations()
    keypoints, features_SIFT = sample_sift.cal_feature_base_keypoints()
    print('keypoints number is %d'%(keypoints.shape[0]))
    print(keypoints)
    print('Features SIFT number is %d' % (features_SIFT.shape[0]))