import ioplin
import efficientnet
import tensorflow as tf
import os
import numpy as np
import PIL
import argparse

here = os.path.abspath(os.path.dirname(os.getcwd()))

def getData(type='train',shape=[]):
    y = []
    path = os.path.join(here,'miniset',type)
    for root,dirs,files in os.walk(path):
        num = len(files)
        j = 0
        for name in files:
            if name != 'groundtruth.txt':
                p = os.path.join(path, name)
                img_o = PIL.Image.open(p)
                if len(shape) != 0:
                    img_o = img_o.resize((shape[0],shape[1]),PIL.Image.ANTIALIAS)
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain IOPLIN with miniset"
    )
    parser.add_argument(
        "--epoch", type=int, default=10, help="train iter epoch"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size in the train"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(here,"model","ioplin_pretrain"),
        help="output model file name",
    )
    

    args = parser.parse_args()
    model = ioplin.init_model()

    x_train,y_train = getData(shape=[300,300])
    x_val,y_val = getData(type='validation',shape=[300,300])

    print("x_train shape: "+str(x_train.shape)+"\ny_train shape: "+str(y_train.shape)+"\nx_val shape: "+str(x_val.shape)+"\ny_val shape: "+str(y_val.shape))

    ioplin.pretrain(model,x_train,y_train,x_val,y_val,args.epoch,args.batch_size,args.output_file)