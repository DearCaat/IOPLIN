import numpy as np
from .utils import preprocess_input

def predict(model,x,positive_index=1):
    '''
    predict image used ioplin.
    Paras:
    model           keras.model
    x               np.array      shape(num*1200*900*1) width 1200 height 900
    positive_index  int           positive class index in hot-dot decode
    Return:
    pre_comb       np.array      road diseased confidence of provided image
    '''
    x,_,_,picts = preprocess_input(x)
    imgs_num = picts[0].num_imgs
    pre_comb = []
    
    pre = model.predict(x)
    if positive_index == 0:
        pre = np.stack((pre[:,1],pre[:,0]),1)
        
    
    for i in range(0,len(picts)):
        picts[i].preNor(pre[i*imgs_num : i*imgs_num + imgs_num])
        pre_comb.append(picts[i].pre_label)
    pre_comb = np.array(pre_comb)
    
    return pre_comb