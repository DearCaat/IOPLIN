'''
image class for ioplin. Store the patch information in the imgage and carry out corresponding necessary operations
'''
import numpy as np
from random import shuffle

'''
default image cut parameter for 1200*900 image
'''
WIN = [300,300]         
S = [300,300]

class pic:
    shape = []        #image's shape
    pics = []         #the patches after clip
    label = 0         #the label of image
    labels_bin = []     #0 reperents diseased，1 repernets normal
    disease_list = []    #the index list of diseased patch
    pre_label = 0      #predicted image label
    pres_bin = []      #the list of predicted patch score
    img = []          #the list of patchs file
    num_imgs = 0       #the number of patchs
    patch_weight = []    #patch's weight
    
    def __init__(self,img_filename = "",img = [],label = 0,win = WIN,s = S,shuffle = False):
        disease_list = []
        self.label = label
        self.shape = img.shape
        self.pics = self.imgcut(img,win,s)
        num_pics = len(self.pics)
        self.num_imgs = num_pics
        self.pres_bin = np.zeros((num_pics,2),'float16')
        if self.label == 1:
            self.labels_bin = np.ones((num_pics,1),'int8')
            self.pre_label = np.array([0,1],'float16')
        else:
            self.labels_bin = np.zeros((num_pics,1),'int8')
            self.pre_label = np.array([1,0],'float16')

        if shuffle:
            self.shuffle()
                
    def shuffle(self):
        '''
        shuffle patches in a image
        '''
        self.pics = np.array(self.pics)
        index_random = [i for i in range(len(self.labels_bin))]
        shuffle(index_random)
        
        if len(self.pics) != 0:
            self.pics = self.pics[index_random,:,:,:]
        self.labels_bin = self.labels_bin[index_random,:]
        self.pres_bin = self.pres_bin[index_random,:]
        
        return index_random

    def imgcut(self,_img,win,s): 
        '''
        clip the pic by slide the window
        Paras:
        img    np.array  shape[height,width,channel]
        win    list    [heighet,width]
        s     list    [height,width]
        Return:
        imgs   list  shape[num.height,width,channel]  the pics after clip
        '''
        imgs = []
        height_src = 0
        width_src = 0
        height_des = win[0]
        width_des = win[1]
        num_row = int((self.shape[1] - width_des) / s[1]) + 1
        num_col = int((self.shape[0] - height_des) / s[0]) + 1
        
        for i in range(0,num_col):
            width_src = 0
            width_des = win[1]
            for k in range(0,num_row):
                img_temp = _img[height_src:height_des,width_src:width_des,:]
                imgs.append(img_temp)
                width_src = width_src + s[1]
                width_des = width_des + s[1]
            height_src = height_src + s[0]
            height_des = height_des + s[0]
            
        return imgs
    
    def updateLabel_bin(self,pre = [],_thr = 0.5):
        '''
        update patch label by pre
        Paras:
        pre    np.array  keras.model.predict
        _thr    float    threshold of bin-classfiy
        
        Returns:
        num    int   number of disease
        '''
        if len(pre)== 0:
            pre = self.pres_bin
        thr = _thr
        num_changed = 0
        self.disease_list = []
        self.pres_bin = pre
        pre_max = 0
        labels_tem = self.labels_bin
        max_rec = 0
        tem = np.hsplit(pre,2)
        sorted_pre = tem[1]
        sorted_pre = np.sort(sorted_pre,axis = 0)
        if self.label == 1:
            for i in range(0,len(pre)):
                if pre_max < pre[i][1]:
                    pre_max = pre[i][1]
                    max_rec = i
                if pre[i][1] > thr and self.labels_bin[i] == 0:
                    self.labels_bin[i] = 1
                    num_changed = num_changed +1
                elif pre[i][1]<= thr and self.labels_bin[i] == 1 and pre[i][1]<= sorted_pre[(int)(len(sorted_pre) * 0.55)]:
                    self.labels_bin[i] = 0
                    num_changed = num_changed +1
                else:
                    if self.labels_bin[i] == 1:
                        self.disease_list.append(i)
                    
            self.pre_label[1] = pre_max
            self.pre_label[0] = 1 - self.pre_label[1]
            
        return len(self.disease_list),num_changed
    
    def preNor(self,pre = [],_thr = 0.5):
        '''
        according to the thrshold to detect whether the image is normal
        '''
        if len(pre)== 0:
            pre = self.pres_bin
        thr = _thr
        num_dis = 0
        pre_max = 0
        max_rec = 0
        for i in range(0,len(pre)):
            if pre_max < pre[i][1]:
                pre_max = pre[i][1]
                max_rec = i
            if pre[i][1] > thr:
                num_dis = num_dis + 1
        self.pre_label[1] = pre_max
        self.pre_label[0] = 1 - self.pre_label[1] 
        if num_dis < 1:
            return True
        else :
            return False
        
    def getSampleWeight(self,thr,pre = []):  
        '''
        return patch's weight
        Paras:
        thr     float     thrshold of binary classification
        pre     list      patch's predicted score
        '''
        if len(pre)== 0:
            pre = self.pres_bin
        s_weight = []
        for i in range(0,len(self.labels_bin)):
            tem = 1 * (pre[i][1] / thr)

            if tem < 0.1:                
                tem = 0.1
            s_weight.append(tem)
        self.patch_weight = s_weight
        self.labels_bin_bfLast = self.labels_bin
        return s_weight
        
    def cvtData(self,x = [],y = [],is_x=True,is_y=True,is_del = True):
        '''
        put the patch file to the external variable
        Paras:
        x      list     external x
        y      list     external y
        is_x    bool     whether process x
        is_y    bool     whether process y
        is_del   bool     whether delete the inner image 
        '''
        num = len(self.labels_bin)
        if is_x:
            for i in range(0,num):
                x.append(np.tile(self.pics[i]),(1,1,3))
            if is_del:
                del self.pics
        if is_y:
            for j in range(0,num):
                y.append(self.labels_bin[j])       
