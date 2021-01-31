from keras.utils import get_file
from .utils import *
import efficientnet
'''

'''

def init_model():
    '''
    build the base model EfficientNet with imagenet weight
    '''
    weights={
        'name': 'efficientnet-b3_imagenet_1000_notop.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b3_imagenet_1000_notop.h5',
        'md5': '1f7d9a8c2469d2e3d3b97680d45df1e1',
    }
    model = efficientnet.EfficientNetB3(classes = 2)
    weights_path = get_file(
            weights['name'],
            weights['url'],
            cache_subdir='models',
            md5_hash=weights['md5'],
        )
    model.load_weights(weights_path,by_name = True)
    model.compile(optimizer='nadam', loss= "binary_crossentropy", metrics=['binary_accuracy'])
    return model

def pretrain(model,x_train,y_train,x_validation,y_validation,epochs=10,batch_size=32,file=''):
    '''
    use the 300*300 thumbnail to pretrain the model.
    
    Para:
    model        keras.model
    x_trian       np.array      shape(num*300*300*3)
    y_train       np.array      shape(num*1)
    x_validation    np.array      shape(num*300*300*3)
    y_validation    np.array      shape(num*1)
    epochs        int         ioplin iter epoch
    batch_size     int         train batch size
    file         string       model save file path
    Return:
    model        keras.model
    his         keras.History
    '''
    
    assert x_train.ndim in (3, 4) and x_validation.ndim in (3, 4)
    assert x_train.shape[-1] == x_validation.shape[-1]
    
    x_train = np.array(x_train,'float16')
    x_validation = np.array(x_validation,'float16')
    
    x_train = normalization(x_train)
    x_validation = normalization(x_validation)
    
    if x_train.shape[-1] == 1:
        x_train = np.tile(x_train,(1,1,1,3))
        x_validation = np.tile(x_validation,(1,1,1,3))
        
    y_train = convert_to_one_hot(y_train, 2).T
    y_validation = convert_to_one_hot(y_validation, 2).T
    
    his = model.fit(x_train, y_train,batch_size = batch_size,epochs = epochs,validation_data=(x_validation,y_validation),
                    callbacks=[roc_callback(training_data=(x_train, y_train),validation_data=(x_validation,y_validation))])
    if file!='':
        model.save(file)
    return model,his

def train(model,x_train,y_train,x_validation,y_validation,epochs=10,batch_size=32,file=''):
    '''
    ioplin main train function.
    Para:
    model         keras.model
    x_trian       np.array      shape(num*1200*900*1)  width 1200 height 900
    y_train       np.array      shape(num*1)        0 denote normal, 1 denote diseased
    x_validation    np.array      shape(num*1200*900*1)
    y_validation    np.array      shape(num*1)
    epochs        int         ioplin iter epoch
    batch_size     int         train batch size
    file         string       model save file path
    '''
    x_train,y_train,y_train_comb,picts_tr = preprocess_input(x_train,y_train)
    x_validation,y_validation,y_validation_comb,picts_val = preprocess_input(x_validation,y_validation)
        
    # init paras
    imgs_num = picts_tr[0].num_imgs    #patch's number of image
    _thr = 0.5                  # the threshold of binary classify
    auc_validation_bf = 0           #the auc of validation in the last epoch
    auc_train_bf = 0    
    auc_train = 0                #the auc of train-set the this epoch
    auc_validation = 0 
    ratio = 0                   #the ratio of diseased patch and normal patch
    
    for m in range(0,epochs):
        epoch_inner = 2
        print(str(m) + "  epochs: ")
        num_dis = 0             #count of patch diseased in train-set
        s_weight = []           #sample weight
        pre_comb_tr = []        #image score predicted in train-set
        pre_comb_val = []        #image score predicted in test-set

        # Train
        if m == 0:
            his = model.fit(x_train, y_train,batch_size = batch_size,epochs = epoch_inner,shuffle = True)
        else:
            his = model.fit(x_train, y_train,batch_size = batch_size,epochs = epoch_inner,shuffle = True,sample_weight = s_weight)


        # Update label of Train-set, only dieased part
        pre_tr = model.predict(x_train)       
        pre_val = model.predict(x_validation)

        y_train = []
        for i in range(0,len(picts_tr)):
            if picts_tr[i].label == 1:
                _num,num_changed_tr = picts_tr[i].updateLabel_bin(pre_tr[i*imgs_num:(i*imgs_num)+imgs_num],_thr)
                pre_comb_tr.append(picts_tr[i].pre_label)
                num_dis = num_dis + _num
            else:
                picts_tr[i].preNor(pre_tr[i*imgs_num : i*imgs_num + imgs_num],_thr)
                pre_comb_tr.append(picts_tr[i].pre_label)

            picts_tr[i].cvtData(y = y_train,is_x=False)

        #test validation
        for i in range(0,len(picts_val)):
            if picts_val[i].label == 1:
                __num,num_changed_val = picts_val[i].updateLabel_bin(pre_val[i*imgs_num:(i*imgs_num)+imgs_num],_thr)
                pre_comb_val.append(picts_val[i].pre_label)
            else:
                picts_val[i].preNor(pre_val[i*imgs_num : i*imgs_num + imgs_num],_thr)
                pre_comb_val.append(picts_val[i].pre_label)

        #compute roc_auc
        auc_validation_bf = auc_validation
        auc_train_bf = auc_train
        pre_comb_tr = np.array(pre_comb_tr)
        pre_comb_val = np.array(pre_comb_val)
        auc_train = roc_auc_score(y_train_comb, pre_comb_tr)
        auc_validation = roc_auc_score(y_validation_comb, pre_comb_val)
        print("Train: roc_auc: " + str(auc_train) +"  Validation:"+  "roc_auc: " + str(auc_validation) )

        y_train = np.array(y_train,'Int8')
        y_train = convert_to_one_hot(y_train, 2).T

        ratio = num_dis/ len(x_train)                              
        _thr = ratio
        
        print("Threshold: ",_thr)

        #update weight
        for i in range(0,len(picts_tr)):
            s_weight.append(picts_tr[i].getSampleWeight(_thr,pre = pre_tr[i*imgs_num:(i*imgs_num)+imgs_num]))
        s_weight = np.array(s_weight,'float16')
        s_weight = s_weight.flatten()
        if auc_validation < auc_validation_bf and auc_train > auc_train_bf and m>2:
            break
        model_last = model
        
    if file!='':
        model_last.save(file)
    return model_last