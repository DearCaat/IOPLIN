from .pict import *
import numpy as np
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from cv2 import createCLAHE

class roc_callback(Callback):
    '''
    keras callback class. Provide ROC calculation support during training.
    '''
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def normalization(x):
    x -= (0.485*255)
    x /= (0.229*255)
    
def clahe_process(x):
    for i in range(0,len(x)):
        clahe = createCLAHE(clipLimit = 2.0,tileGridSize = (8,8))   
        img = clahe.apply(np.array(x[i],'uint8'))
        x[i] = img.reshape((img.shape[0],img.shape[1],1))
        
    return x
    
def preprocess_input(x,y=[]):
    
    assert x.ndim in (3, 4)
    assert x.shape[-1] == 1
    
    X = []
    Y = []
    picts = []
    Y_comb = []
    
    x = clahe_process(x)
    
    for k in range(0,len(x)):
        if len(y) == 0:
            picts.append(pic(img = x[k],shuffle = True))
        else:
            picts.append(pic(img = x[k],label = y[k],shuffle = True))
    del x
    
    for k in range(0,len(picts)):
        if len(y) == 0:
            picts[k].cvtData(X,Y,is_y=False)
        else:
            picts[k].cvtData(X,Y)

    X = np.array(X,'float16')    
    normalization(X)
    
    if len(Y) != 0:
        Y = np.array(Y)
        Y = convert_to_one_hot(Y,2).T
        Y_comb = convert_to_one_hot(y, 2).T
    
    return X,Y,Y_comb,picts
