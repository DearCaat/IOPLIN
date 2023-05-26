# IOPLIN 
This repo is the official implementation of ["Iteratively Optimized Patch Label Inference Network for Automatic Pavement Disease Detection"](https://ieeexplore.ieee.org/abstract/document/9447759) based on Keras, and IOPLIN has published in: [IEEE Transactions on Intelligent Transportation Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6979). The source code is placed at [/ioplin](https://github.com/DearCaat/ioplin/tree/main/ioplin), the script is placed at [/script](https://github.com/DearCaat/ioplin/tree/main/script), and the required mini dataset should place at [/miniset](https://github.com/DearCaat/ioplin/tree/main/miniset).

For more details of the pavement dataset CQU-BPDD used in paper, please refer to [CQU-BPDD](https://github.com/DearCaat/CQU-BPDD).
 (Note: CQU-BPDD can be only used in the uncommercial case and is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).)

For more details of this task, see [Pavement Distress Classification](https://github.com/DearCaat/Pavement-Distress-Classification).

## Installation

### Requirements

* `Keras == 2.2.4` / `TensorFlow <= 1.15.0`
* `keras_applications >= 1.0.7`
* `scikit-image`
* `opencv-python`
* `efficientnet <= 0.0.3`

### Installing from the source

```bash
$ python setup.py install
```

## Examples
* *Initializing the model*:

```python
# models can be build with Keras

import ioplin

model = ioplin.init_model()  

```

* *Loading the pre-trained weights*:

```python
import ioplin

model = ioplin.load_model('path/to/model.h5')
```

* *Predicting*:

```python
import ioplin

pre = ioplin.predict(model,data)
```

* *Pretrain and train*:

```python
import ioplin

model = ioplin.pretrain(model,data_x,data_y)

model = ioplin.train(model,data_x,data_y)
```

## Script
Simple application script of IOPLIN, and the required dataset and default model of IOPLIN can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1eNu3IJ_N4ND3rlvuADsQd19wTIxE_T9Y?usp=sharing)
### Pretrain
```bash
$ python pretrain.py --batch_size=32 --epoch=10
```
### Train
```bash
$ python train.py --batch_size=32 --epoch=10 --path_pretrain_model=path/pretrain_model.h5
```
### Predict

#### If you want to use the trained IOPLIN model to predict data, you should download the model file and  enter:
```bash
$ python predict.py --path_model=path/trained_model.h5 --positive_index=0 
```
#### If you want to use the IOPLIN model trained by yourself to predict data, you only need enter:
```bash
$ python predict.py --path_model=path/your_model.h5
```
