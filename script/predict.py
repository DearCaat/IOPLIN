import ioplin
import efficientnet
import tensorflow as tf
import os
import numpy as np
import PIL
from sklearn.metrics import roc_auc_score, roc_curve
import argparse

here = os.path.abspath(os.path.dirname(os.getcwd()))

def getData(type='train'):
    y = []
    path = os.path.join(here,'miniset',type)
    for root,dirs,files in os.walk(path):
        num = len(files)
        j = 0
        
        for name in files:
            if name != 'groundtruth.txt':
                p = os.path.join(path, name)
                img_o = PIL.Image.open(p)
                img = np.array(img_o,'uint8')
                img = img.reshape((img.shape[0],img.shape[1],1))                
                if j == 0:
                    img_matrix = np.zeros((num-1,img.shape[0],img.shape[1],1),'uint8')
                    img_matrix[j] = img
                    j = j+1
                else:  
                    img_matrix[j] = img
                    j = j+1
                if int(name[0:-4])<int(num/2):
                    y.append([1])
                else:
                    y.append([0])
    
    return img_matrix,np.array(y,'int8')
    
def find_optimal_cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  
    optimal_threshold = threshold[Youden_index]
    return optimal_threshold,TPR[Youden_index],FPR[Youden_index]
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Predict with IOPLIN in miniset"
    )

    parser.add_argument(
        "--path_model",
        type=str,
        default=os.path.join(here,"model","ioplin_default"),
        help="model path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(here,"results","ioplin_res"),
        help="output predicted file name",
    )
    parser.add_argument(
        "--positive_index",
        type=int,
        default=1,
        help="positive class index in hot-dot decode",
    )
    args = parser.parse_args()
    
    model = efficientnet.EfficientNetB3(classes = 2)
    model.load_weights(args.path_model,by_name=True)
    
    x_test,y_test = getData(type='test')

    print("x_test shape: "+str(x_test.shape))
    
    pre = ioplin.predict(model,x_test,args.positive_index)
    if positive_index == 0:
        pre = pre[:,0]
    else:
        pre = pre[:,1]
    if args.output_file != '':
        np.savez(args.output_file,y_test,pre)
        
    fpr,tpr,threshold = roc_curve(y_test, pre) 
    roc_auc = roc_auc_score(y_test, pre) 
    best_thr,best_tpr,best_fpr = find_optimal_cutoff(tpr,fpr,threshold)
    
    print("AUC: %.2f" % roc_auc)
    print("Best threshold: %.2f" % best_thr)
    print("In best threshold, tpr is : %.2f, fpr is : %.2f" % (best_tpr,best_fpr))