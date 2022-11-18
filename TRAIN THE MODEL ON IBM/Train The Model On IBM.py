#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
ls
drive/  sample_data/
cd /content/drive/MyDrive/dataset
content(/drive/MyDrive/dataset)
ls
model-bw.json                             naturaldisaster.h5  test_set/
naturaldisaster-classification-model.tgz  readme.txt          train_set/
pwd
'/content/drive/MyDrive/dataset'
# importing imagedatagenerator library

from tensorflow.keras.preprocessing.image import ImageDataGenerator
Image Data Augmentation
#Configuring image Data Generator Class
#Setting Parameter for Image Augmentation for training data

train_datagen=ImageDataGenerator(rescale=1./255,horizontal_flip=True,vertical_flip=True, zoom_range=0.2)
#Image Data Augmentation for testing data

test_datagen= ImageDataGenerator(rescale=1./255)
Apply ImageDataGenerator Functionality To Train And Test Dataset
x_train = train_datagen.flow_from_directory('/content/drive/MyDrive/dataset/train_set', target_size = (64,64), batch_size = 5, color_mode = 'rgb', class_mode = 'categorical')
Found 744 images belonging to 4 classes.
#performing data augmentation to test data
x_test = test_datagen.flow_from_directory('/content/drive/MyDrive/dataset/test_set', target_size = (64,64), batch_size = 5, color_mode = 'rgb', class_mode = 'categorical')
Found 198 images belonging to 4 classes.
#importing neccessary libraries

import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
# initialising the model and adding CNN layers

model = Sequential()

# First convolution layer and pooling

model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Second convolution layer and pooling

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Flattening the layers

model.add(Flatten())

#Adding Dense Layers

model.add(Dense(300,activation='relu'))
model.add(Dense(4,activation='softmax'))
# Summary of our model

model.summary()
Model: "sequential"
_______________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 62, 62, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 31, 31, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 29, 29, 32)        9248      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 14, 14, 32)       0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 6272)              0         
                                                                 
 dense (Dense)               (None, 300)               1881900   
                                                                 
 dense_1 (Dense)             (None, 4)                 1204      
                                                                 
=================================================================
Total params: 1,893,248
Trainable params: 1,893,248
Non-trainable params: 0
_______________________
# Compiling the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Fitting the model

model.fit_generator(generator=x_train,steps_per_epoch=len(x_train),epochs=20,validation_data=x_test,validation_steps=len(x_test))
usr(/local/lib/python3.7/dist-packages/ipykernel_launcher.py:3:, UserWarning:, `Model.fit_generator`, is, deprecated, and, will, be, removed, in, a, future, version., Please, use, `Model.fit`,, which, supports, generators.)
  This is separate from the ipykernel package so we can avoid doing imports until
Epoch 1/20
149/149 [==============================] - 234s 2s/step - loss: 1.2562 - accuracy: 0.4274 - val_loss: 1.1455 - val_accuracy: 0.4545
Epoch 2/20
149/149 [==============================] - 40s 271ms/step - loss: 0.9740 - accuracy: 0.5726 - val_loss: 0.9891 - val_accuracy: 0.5758
Epoch 3/20
149/149 [==============================] - 44s 299ms/step - loss: 0.8745 - accuracy: 0.6250 - val_loss: 1.0895 - val_accuracy: 0.4697
Epoch 4/20
149/149 [==============================] - 42s 284ms/step - loss: 0.7803 - accuracy: 0.6640 - val_loss: 0.8878 - val_accuracy: 0.5960
Epoch 5/20
149/149 [==============================] - 41s 277ms/step - loss: 0.7086 - accuracy: 0.6962 - val_loss: 0.6702 - val_accuracy: 0.7323
Epoch 6/20
149/149 [==============================] - 42s 282ms/step - loss: 0.6888 - accuracy: 0.7137 - val_loss: 0.8625 - val_accuracy: 0.6414
Epoch 7/20
149/149 [==============================] - 42s 280ms/step - loss: 0.6015 - accuracy: 0.7513 - val_loss: 0.9248 - val_accuracy: 0.6465
Epoch 8/20
149/149 [==============================] - 42s 282ms/step - loss: 0.5680 - accuracy: 0.7742 - val_loss: 0.8679 - val_accuracy: 0.6515
Epoch 9/20
149/149 [==============================] - 42s 280ms/step - loss: 0.5170 - accuracy: 0.7903 - val_loss: 0.6369 - val_accuracy: 0.7727
Epoch 10/20
149/149 [==============================] - 42s 281ms/step - loss: 0.5122 - accuracy: 0.8051 - val_loss: 0.6164 - val_accuracy: 0.7980
Epoch 11/20
149/149 [==============================] - 43s 290ms/step - loss: 0.4704 - accuracy: 0.8199 - val_loss: 0.8725 - val_accuracy: 0.6616
Epoch 12/20
149/149 [==============================] - 42s 282ms/step - loss: 0.4568 - accuracy: 0.8293 - val_loss: 0.6015 - val_accuracy: 0.8434
Epoch 13/20
149/149 [==============================] - 42s 280ms/step - loss: 0.4047 - accuracy: 0.8468 - val_loss: 0.6312 - val_accuracy: 0.8182
Epoch 14/20
149/149 [==============================] - 42s 284ms/step - loss: 0.3708 - accuracy: 0.8508 - val_loss: 0.7347 - val_accuracy: 0.7778
Epoch 15/20
149/149 [==============================] - 42s 279ms/step - loss: 0.3534 - accuracy: 0.8642 - val_loss: 0.8544 - val_accuracy: 0.7626
Epoch 16/20
149/149 [==============================] - 42s 284ms/step - loss: 0.3371 - accuracy: 0.8750 - val_loss: 0.7410 - val_accuracy: 0.7828
Epoch 17/20
149/149 [==============================] - 41s 274ms/step - loss: 0.3326 - accuracy: 0.8710 - val_loss: 0.8407 - val_accuracy: 0.7475
Epoch 18/20
149/149 [==============================] - 43s 287ms/step - loss: 0.3509 - accuracy: 0.8723 - val_loss: 0.8589 - val_accuracy: 0.7576
Epoch 19/20
149/149 [==============================] - 43s 288ms/step - loss: 0.2753 - accuracy: 0.9032 - val_loss: 0.8497 - val_accuracy: 0.7778
Epoch 20/20
149/149 [==============================] - 42s 285ms/step - loss: 0.2949 - accuracy: 0.8817 - val_loss: 0.9368 - val_accuracy: 0.7424
# Save the model

model.save('naturaldisaster.h5')
model_json = model.to_json()
with open("model-bw.json", "w") as json_file:
  json_file.write(model_json)
# Load the saved model

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model = load_model('naturaldisaster.h5')
# zipping the Model

get_ipython().system('tar -zcvf naturaldisaster-classification-model.tgz naturaldisaster.h5')
naturaldisaster.h5
# connecting with IBM Cloud

get_ipython().system('pip install watson-machine-learning-client')
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting watson-machine-learning-client
  Downloading watson_machine_learning_client-1.0.391-py3-none-any.whl (538 kB)
     |████████████████████████████████| 538 kB 5.0 MB/s 
Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from watson-machine-learning-client) (1.3.5)
Requirement already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from watson-machine-learning-client) (2022.9.24)
Requirement already satisfied: tabulate in /usr/local/lib/python3.7/dist-packages (from watson-machine-learning-client) (0.8.10)
Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from watson-machine-learning-client) (2.23.0)
Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from watson-machine-learning-client) (1.24.3)
Collecting ibm-cos-sdk
  Downloading ibm-cos-sdk-2.12.0.tar.gz (55 kB)
     |████████████████████████████████| 55 kB 3.1 MB/s 
Collecting boto3
  Downloading boto3-1.26.4-py3-none-any.whl (132 kB)
     |████████████████████████████████| 132 kB 52.9 MB/s 
Collecting lomond
  Downloading lomond-0.3.3-py2.py3-none-any.whl (35 kB)
Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from watson-machine-learning-client) (4.64.1)
Collecting botocore<1.30.0,>=1.29.4
  Downloading botocore-1.29.4-py3-none-any.whl (9.8 MB)
     |████████████████████████████████| 9.8 MB 48.7 MB/s 
Collecting s3transfer<0.7.0,>=0.6.0
  Downloading s3transfer-0.6.0-py3-none-any.whl (79 kB)
     |████████████████████████████████| 79 kB 6.3 MB/s 
Collecting jmespath<2.0.0,>=0.7.1
  Downloading jmespath-1.0.1-py3-none-any.whl (20 kB)
Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /usr/local/lib/python3.7/dist-packages (from botocore<1.30.0,>=1.29.4->boto3->watson-machine-learning-client) (2.8.2)
Collecting urllib3
  Downloading urllib3-1.26.12-py2.py3-none-any.whl (140 kB)
     |████████████████████████████████| 140 kB 53.7 MB/s 
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil<3.0.0,>=2.1->botocore<1.30.0,>=1.29.4->boto3->watson-machine-learning-client) (1.15.0)
Collecting ibm-cos-sdk-core==2.12.0
  Downloading ibm-cos-sdk-core-2.12.0.tar.gz (956 kB)
     |████████████████████████████████| 956 kB 54.6 MB/s 
Collecting ibm-cos-sdk-s3transfer==2.12.0
  Downloading ibm-cos-sdk-s3transfer-2.12.0.tar.gz (135 kB)
     |████████████████████████████████| 135 kB 52.8 MB/s 
Collecting jmespath<2.0.0,>=0.7.1
  Downloading jmespath-0.10.0-py2.py3-none-any.whl (24 kB)
Collecting requests
  Downloading requests-2.28.1-py3-none-any.whl (62 kB)
     |████████████████████████████████| 62 kB 1.5 MB/s 
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->watson-machine-learning-client) (2.10)
Requirement already satisfied: charset-normalizer<3,>=2 in /usr/local/lib/python3.7/dist-packages (from requests->watson-machine-learning-client) (2.1.1)
Requirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.7/dist-packages (from pandas->watson-machine-learning-client) (1.21.6)
Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->watson-machine-learning-client) (2022.5)
Building wheels for collected packages: ibm-cos-sdk, ibm-cos-sdk-core, ibm-cos-sdk-s3transfer
  Building wheel for ibm-cos-sdk (setup.py) ... done
  Created wheel for ibm-cos-sdk: filename=ibm_cos_sdk-2.12.0-py3-none-any.whl size=73930 sha256=e429043fbb75f37468a1a657062378abd48d13e62c0a789c926efbe1c2719c57
  Stored in directory: /root/.cache/pip/wheels/ec/94/29/2b57327cf00664b6614304f7958abd29d77ea0e5bbece2ea57
  Building wheel for ibm-cos-sdk-core (setup.py) ... done
  Created wheel for ibm-cos-sdk-core: filename=ibm_cos_sdk_core-2.12.0-py3-none-any.whl size=562962 sha256=c161928b4e8c9c94915ef2bfabeee5c1a02721acaefed7b2efd7bd0b6ae2b469
  Stored in directory: /root/.cache/pip/wheels/64/56/fb/5cd6f4f40406c828a5289b95b2752a4d142a9afb359244ed8d
  Building wheel for ibm-cos-sdk-s3transfer (setup.py) ... done
  Created wheel for ibm-cos-sdk-s3transfer: filename=ibm_cos_sdk_s3transfer-2.12.0-py3-none-any.whl size=89778 sha256=79ee31233df54b39cc410411bef45ddf31b1cd67a4dcf67b472882c34318d81c
  Stored in directory: /root/.cache/pip/wheels/57/79/6a/ffe3370ed7ebc00604f9f76766e1e0348dcdcad2b2e32df9e1
Successfully built ibm-cos-sdk ibm-cos-sdk-core ibm-cos-sdk-s3transfer
Installing collected packages: urllib3, requests, jmespath, ibm-cos-sdk-core, botocore, s3transfer, ibm-cos-sdk-s3transfer, lomond, ibm-cos-sdk, boto3, watson-machine-learning-client
  Attempting uninstall: urllib3
    Found existing installation: urllib3 1.24.3
    Uninstalling urllib3-1.24.3:
      Successfully uninstalled urllib3-1.24.3
  Attempting uninstall: requests
    Found existing installation: requests 2.23.0
    Uninstalling requests-2.23.0:
      Successfully uninstalled requests-2.23.0
Successfully installed boto3-1.26.4 botocore-1.29.4 ibm-cos-sdk-2.12.0 ibm-cos-sdk-core-2.12.0 ibm-cos-sdk-s3transfer-2.12.0 jmespath-0.10.0 lomond-0.3.3 requests-2.28.1 s3transfer-0.6.0 urllib3-1.26.12 watson-machine-learning-client-1.0.391
get_ipython().system('pip install ibm_watson_machine_learning')
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting ibm_watson_machine_learning
  Downloading ibm_watson_machine_learning-1.0.257-py3-none-any.whl (1.8 MB)
     | 1.8 MB 5.5 MB/s 
Requirement already satisfied: tabulate in /usr/local/lib/python3.7/dist-packages (from ibm_watson_machine_learning) (0.8.10)
Requirement already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from ibm_watson_machine_learning) (2022.9.24)
Requirement already satisfied: pandas<1.5.0,>=0.24.2 in /usr/local/lib/python3.7/dist-packages (from ibm_watson_machine_learning) (1.3.5)
Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from ibm_watson_machine_learning) (4.13.0)
Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from ibm_watson_machine_learning) (2.28.1)
Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from ibm_watson_machine_learning) (21.3)
Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from ibm_watson_machine_learning) (1.26.12)
Requirement already satisfied: lomond in /usr/local/lib/python3.7/dist-packages (from ibm_watson_machine_learning) (0.3.3)
Collecting ibm-cos-sdk==2.7.*
  Downloading ibm-cos-sdk-2.7.0.tar.gz (51 kB)
     |████████████████████████████████| 51 kB 699 kB/s 
Collecting ibm-cos-sdk-core==2.7.0
  Downloading ibm-cos-sdk-core-2.7.0.tar.gz (824 kB)
     |████████████████████████████████| 824 kB 41.5 MB/s 
Collecting ibm-cos-sdk-s3transfer==2.7.0
  Downloading ibm-cos-sdk-s3transfer-2.7.0.tar.gz (133 kB)
     |████████████████████████████████| 133 kB 51.4 MB/s 
Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.7/dist-packages (from ibm-cos-sdk==2.7.*->ibm_watson_machine_learning) (0.10.0)
Collecting docutils<0.16,>=0.10
  Downloading docutils-0.15.2-py3-none-any.whl (547 kB)
     |████████████████████████████████| 547 kB 63.9 MB/s 
Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /usr/local/lib/python3.7/dist-packages (from ibm-cos-sdk-core==2.7.0->ibm-cos-sdk==2.7.*->ibm_watson_machine_learning) (2.8.2)
Requirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.7/dist-packages (from pandas<1.5.0,>=0.24.2->ibm_watson_machine_learning) (1.21.6)
Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas<1.5.0,>=0.24.2->ibm_watson_machine_learning) (2022.5)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil<3.0.0,>=2.1->ibm-cos-sdk-core==2.7.0->ibm-cos-sdk==2.7.*->ibm_watson_machine_learning) (1.15.0)
Requirement already satisfied: charset-normalizer<3,>=2 in /usr/local/lib/python3.7/dist-packages (from requests->ibm_watson_machine_learning) (2.1.1)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->ibm_watson_machine_learning) (2.10)
Requirement already satisfied: typing-extensions>=3.6.4 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->ibm_watson_machine_learning) (4.1.1)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->ibm_watson_machine_learning) (3.10.0)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->ibm_watson_machine_learning) (3.0.9)
Building wheels for collected packages: ibm-cos-sdk, ibm-cos-sdk-core, ibm-cos-sdk-s3transfer
  Building wheel for ibm-cos-sdk (setup.py) ... done
  Created wheel for ibm-cos-sdk: filename=ibm_cos_sdk-2.7.0-py2.py3-none-any.whl size=72564 sha256=6d973438356404c8cb2d47a49ad095241ef62e00cec0f11b122865b2bb7428aa
  Stored in directory: /root/.cache/pip/wheels/47/22/bf/e1154ff0f5de93cc477acd0ca69abfbb8b799c5b28a66b44c2
  Building wheel for ibm-cos-sdk-core (setup.py) ... done
  Created wheel for ibm-cos-sdk-core: filename=ibm_cos_sdk_core-2.7.0-py2.py3-none-any.whl size=501013 sha256=4dde424ca01f763b0b20f8623a8005683241051c2b59e823e75650fe94aa28f3
  Stored in directory: /root/.cache/pip/wheels/6c/a2/e4/c16d02f809a3ea998e17cfd02c13369281f3d232aaf5902c19
  Building wheel for ibm-cos-sdk-s3transfer (setup.py) ... done
  Created wheel for ibm-cos-sdk-s3transfer: filename=ibm_cos_sdk_s3transfer-2.7.0-py2.py3-none-any.whl size=88619 sha256=9ff2be68476a118dd2c8e4b7c7fdde52e2ed187cb3fcceaa8cb44e36b20cfc27
  Stored in directory: /root/.cache/pip/wheels/5f/b7/14/fbe02bc1ef1af890650c7e51743d1c83890852e598d164b9da
Successfully built ibm-cos-sdk ibm-cos-sdk-core ibm-cos-sdk-s3transfer
Installing collected packages: docutils, ibm-cos-sdk-core, ibm-cos-sdk-s3transfer, ibm-cos-sdk, ibm-watson-machine-learning
  Attempting uninstall: docutils
    Found existing installation: docutils 0.17.1
    Uninstalling docutils-0.17.1:
      Successfully uninstalled docutils-0.17.1
  Attempting uninstall: ibm-cos-sdk-core
    Found existing installation: ibm-cos-sdk-core 2.12.0
    Uninstalling ibm-cos-sdk-core-2.12.0:
      Successfully uninstalled ibm-cos-sdk-core-2.12.0
  Attempting uninstall: ibm-cos-sdk-s3transfer
    Found existing installation: ibm-cos-sdk-s3transfer 2.12.0
    Uninstalling ibm-cos-sdk-s3transfer-2.12.0:
      Successfully uninstalled ibm-cos-sdk-s3transfer-2.12.0
  Attempting uninstall: ibm-cos-sdk
    Found existing installation: ibm-cos-sdk 2.12.0
    Uninstalling ibm-cos-sdk-2.12.0:
      Successfully uninstalled ibm-cos-sdk-2.12.0
Successfully installed docutils-0.15.2 ibm-cos-sdk-2.7.0 ibm-cos-sdk-core-2.7.0 ibm-cos-sdk-s3transfer-2.7.0 ibm-watson-machine-learning-1.0.257
from ibm_watson_machine_learning import APIClient
wml_credentials = {
    "url" : "https://us-south.ml.cloud.ibm.com",
    "apikey" :"vC0huO9WFiLZVMOSDTra99pdVQRI7nlGFSpgQvBepP9J"
}
wml_client=APIClient(wml_credentials)
wml_client.spaces.list()
Python 3.7 and 3.8 frameworks are deprecated and will be removed in a future release. Use Python 3.9 framework instead.
Note: 'limit' is not provided. Only first 50 records will be displayed if the number of records exceed 50
------------------------------------  --------------------------------------  ------------------------
ID                                    NAME                                    CREATED
c147dde1-e498-4151-bcf8-7b82eebf2ee3  NaturalDisasterIntensityClassification  2022-11-08T14:27:07.638Z
------------------------------------  --------------------------------------  ------------------------
space_id="c147dde1-e498-4151-bcf8-7b82eebf2ee3"
wml_client.set.default_space(space_id)
'SUCCESS'
wml_client.software_specifications.list(500)
-------------------------------  ------------------------------------  ----
NAME                             ASSET_ID                              TYPE
default_py3.6                    0062b8c9-8b7d-44a0-a9b9-46c416adcbd9  base
kernel-spark3.2-scala2.12        020d69ce-7ac1-5e68-ac1a-31189867356a  base
pytorch-onnx_1.3-py3.7-edt       069ea134-3346-5748-b513-49120e15d288  base
scikit-learn_0.20-py3.6          09c5a1d0-9c1e-4473-a344-eb7b665ff687  base
spark-mllib_3.0-scala_2.12       09f4cff0-90a7-5899-b9ed-1ef348aebdee  base
pytorch-onnx_rt22.1-py3.9        0b848dd4-e681-5599-be41-b5f6fccc6471  base
ai-function_0.1-py3.6            0cdb0f1e-5376-4f4d-92dd-da3b69aa9bda  base
shiny-r3.6                       0e6e79df-875e-4f24-8ae9-62dcc2148306  base
tensorflow_2.4-py3.7-horovod     1092590a-307d-563d-9b62-4eb7d64b3f22  base
pytorch_1.1-py3.6                10ac12d6-6b30-4ccd-8392-3e922c096a92  base
tensorflow_1.15-py3.6-ddl        111e41b3-de2d-5422-a4d6-bf776828c4b7  base
runtime-22.1-py3.9               12b83a17-24d8-5082-900f-0ab31fbfd3cb  base
scikit-learn_0.22-py3.6          154010fa-5b3b-4ac1-82af-4d5ee5abbc85  base
default_r3.6                     1b70aec3-ab34-4b87-8aa0-a4a3c8296a36  base
pytorch-onnx_1.3-py3.6           1bc6029a-cc97-56da-b8e0-39c3880dbbe7  base
kernel-spark3.3-r3.6             1c9e5454-f216-59dd-a20e-474a5cdf5988  base
pytorch-onnx_rt22.1-py3.9-edt    1d362186-7ad5-5b59-8b6c-9d0880bde37f  base
tensorflow_2.1-py3.6             1eb25b84-d6ed-5dde-b6a5-3fbdf1665666  base
spark-mllib_3.2                  20047f72-0a98-58c7-9ff5-a77b012eb8f5  base
tensorflow_2.4-py3.8-horovod     217c16f6-178f-56bf-824a-b19f20564c49  base
runtime-22.1-py3.9-cuda          26215f05-08c3-5a41-a1b0-da66306ce658  base
do_py3.8                         295addb5-9ef9-547e-9bf4-92ae3563e720  base
autoai-ts_3.8-py3.8              2aa0c932-798f-5ae9-abd6-15e0c2402fb5  base
tensorflow_1.15-py3.6            2b73a275-7cbf-420b-a912-eae7f436e0bc  base
kernel-spark3.3-py3.9            2b7961e2-e3b1-5a8c-a491-482c8368839a  base
pytorch_1.2-py3.6                2c8ef57d-2687-4b7d-acce-01f94976dac1  base
spark-mllib_2.3                  2e51f700-bca0-4b0d-88dc-5c6791338875  base
pytorch-onnx_1.1-py3.6-edt       32983cea-3f32-4400-8965-dde874a8d67e  base
spark-mllib_3.0-py37             36507ebe-8770-55ba-ab2a-eafe787600e9  base
spark-mllib_2.4                  390d21f8-e58b-4fac-9c55-d7ceda621326  base
xgboost_0.82-py3.6               39e31acd-5f30-41dc-ae44-60233c80306e  base
pytorch-onnx_1.2-py3.6-edt       40589d0e-7019-4e28-8daa-fb03b6f4fe12  base
default_r36py38                  41c247d3-45f8-5a71-b065-8580229facf0  base
autoai-ts_rt22.1-py3.9           4269d26e-07ba-5d40-8f66-2d495b0c71f7  base
autoai-obm_3.0                   42b92e18-d9ab-567f-988a-4240ba1ed5f7  base
pmml-3.0_4.3                     493bcb95-16f1-5bc5-bee8-81b8af80e9c7  base
spark-mllib_2.4-r_3.6            49403dff-92e9-4c87-a3d7-a42d0021c095  base
xgboost_0.90-py3.6               4ff8d6c2-1343-4c18-85e1-689c965304d3  base
pytorch-onnx_1.1-py3.6           50f95b2a-bc16-43bb-bc94-b0bed208c60b  base
autoai-ts_3.9-py3.8              52c57136-80fa-572e-8728-a5e7cbb42cde  base
spark-mllib_2.4-scala_2.11       55a70f99-7320-4be5-9fb9-9edb5a443af5  base
spark-mllib_3.0                  5c1b0ca2-4977-5c2e-9439-ffd44ea8ffe9  base
autoai-obm_2.0                   5c2e37fa-80b8-5e77-840f-d912469614ee  base
spss-modeler_18.1                5c3cad7e-507f-4b2a-a9a3-ab53a21dee8b  base
cuda-py3.8                       5d3232bf-c86b-5df4-a2cd-7bb870a1cd4e  base
autoai-kb_3.1-py3.7              632d4b22-10aa-5180-88f0-f52dfb6444d7  base
pytorch-onnx_1.7-py3.8           634d3cdc-b562-5bf9-a2d4-ea90a478456b  base
spark-mllib_2.3-r_3.6            6586b9e3-ccd6-4f92-900f-0f8cb2bd6f0c  base
tensorflow_2.4-py3.7             65e171d7-72d1-55d9-8ebb-f813d620c9bb  base
spss-modeler_18.2                687eddc9-028a-4117-b9dd-e57b36f1efa5  base
pytorch-onnx_1.2-py3.6           692a6a4d-2c4d-45ff-a1ed-b167ee55469a  base
spark-mllib_2.3-scala_2.11       7963efe5-bbec-417e-92cf-0574e21b4e8d  base
spark-mllib_2.4-py37             7abc992b-b685-532b-a122-a396a3cdbaab  base
caffe_1.0-py3.6                  7bb3dbe2-da6e-4145-918d-b6d84aa93b6b  base
pytorch-onnx_1.7-py3.7           812c6631-42b7-5613-982b-02098e6c909c  base
cuda-py3.6                       82c79ece-4d12-40e6-8787-a7b9e0f62770  base
tensorflow_1.15-py3.6-horovod    8964680e-d5e4-5bb8-919b-8342c6c0dfd8  base
hybrid_0.1                       8c1a58c6-62b5-4dc4-987a-df751c2756b6  base
pytorch-onnx_1.3-py3.7           8d5d8a87-a912-54cf-81ec-3914adaa988d  base
caffe-ibm_1.0-py3.6              8d863266-7927-4d1e-97d7-56a7f4c0a19b  base
spss-modeler_17.1                902d0051-84bd-4af6-ab6b-8f6aa6fdeabb  base
do_12.10                         9100fd72-8159-4eb9-8a0b-a87e12eefa36  base
do_py3.7                         9447fa8b-2051-4d24-9eef-5acb0e3c59f8  base
spark-mllib_3.0-r_3.6            94bb6052-c837-589d-83f1-f4142f219e32  base
cuda-py3.7-opence                94e9652b-7f2d-59d5-ba5a-23a414ea488f  base
nlp-py3.8                        96e60351-99d4-5a1c-9cc0-473ac1b5a864  base
cuda-py3.7                       9a44990c-1aa1-4c7d-baf8-c4099011741c  base
hybrid_0.2                       9b3f9040-9cee-4ead-8d7a-780600f542f7  base
spark-mllib_3.0-py38             9f7a8fc1-4d3c-5e65-ab90-41fa8de2d418  base
autoai-kb_3.3-py3.7              a545cca3-02df-5c61-9e88-998b09dc79af  base
spark-mllib_3.0-py39             a6082a27-5acc-5163-b02c-6b96916eb5e0  base
runtime-22.1-py3.9-do            a7e7dbf1-1d03-5544-994d-e5ec845ce99a  base
default_py3.8                    ab9e1b80-f2ce-592c-a7d2-4f2344f77194  base
tensorflow_rt22.1-py3.9          acd9c798-6974-5d2f-a657-ce06e986df4d  base
kernel-spark3.2-py3.9            ad7033ee-794e-58cf-812e-a95f4b64b207  base
autoai-obm_2.0 with Spark 3.0    af10f35f-69fa-5d66-9bf5-acb58434263a  base
default_py3.7_opence             c2057dd4-f42c-5f77-a02f-72bdbd3282c9  base
tensorflow_2.1-py3.7             c4032338-2a40-500a-beef-b01ab2667e27  base
do_py3.7_opence                  cc8f8976-b74a-551a-bb66-6377f8d865b4  base
spark-mllib_3.3                  d11f2434-4fc7-58b7-8a62-755da64fdaf8  base
autoai-kb_3.0-py3.6              d139f196-e04b-5d8b-9140-9a10ca1fa91a  base
spark-mllib_3.0-py36             d82546d5-dd78-5fbb-9131-2ec309bc56ed  base
autoai-kb_3.4-py3.8              da9b39c3-758c-5a4f-9cfd-457dd4d8c395  base
kernel-spark3.2-r3.6             db2fe4d6-d641-5d05-9972-73c654c60e0a  base
autoai-kb_rt22.1-py3.9           db6afe93-665f-5910-b117-d879897404d9  base
tensorflow_rt22.1-py3.9-horovod  dda170cc-ca67-5da7-9b7a-cf84c6987fae  base
autoai-ts_1.0-py3.7              deef04f0-0c42-5147-9711-89f9904299db  base
tensorflow_2.1-py3.7-horovod     e384fce5-fdd1-53f8-bc71-11326c9c635f  base
default_py3.7                    e4429883-c883-42b6-87a8-f419d64088cd  base
do_22.1                          e51999ba-6452-5f1f-8287-17228b88b652  base
autoai-obm_3.2                   eae86aab-da30-5229-a6a6-1d0d4e368983  base
do_20.1                          f686cdd9-7904-5f9d-a732-01b0d6b10dc5  base
scikit-learn_0.19-py3.6          f963fa9d-4bb7-5652-9c5d-8d9289ef6ad9  base
tensorflow_2.4-py3.8             fe185c44-9a99-5425-986b-59bd1d2eda46  base
-------------------------------  ------------------------------------  ----
software_spec_uid=wml_client.software_specifications.get_uid_by_name("tensorflow_rt22.1-py3.9")
software_spec_uid
'acd9c798-6974-5d2f-a657-ce06e986df4d'
model_details = wml_client.repository.store_model(model="naturaldisaster-classification-model.tgz",meta_props={
    wml_client.repository.ModelMetaNames.NAME : "CNN model",
    wml_client.repository.ModelMetaNames.TYPE : "tensorflow_2.7",
    wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid
})
model_id = wml_client.repository.get_model_id(model_details)
model_id
'4557f719-8098-4f5d-9580-9fa833e7a0cc'
wml_client.repository.download(model_id,'naturaldisaster.tar.gb')
Successfully saved model content to file: 'naturaldisaster.tar.gb'
'/content/drive/MyDrive/dataset/naturaldisaster.tar.gb'
model_details
{'entity': {'hybrid_pipeline_software_specs': [],
  'software_spec': {'id': 'acd9c798-6974-5d2f-a657-ce06e986df4d',
   'name': 'tensorflow_rt22.1-py3.9'},
  'type': 'tensorflow_2.7'},
 'metadata': {'created_at': '2022-11-08T15:20:53.376Z',
  'id': '4557f719-8098-4f5d-9580-9fa833e7a0cc',
  'modified_at': '2022-11-08T15:20:57.953Z',
  'name': 'CNN model',
  'owner': 'IBMid-6630043HKX',
  'resource_key': 'ddc1d12c-300e-4cff-b63a-4ee630ac4f6d',
  'space_id': 'c147dde1-e498-4151-bcf8-7b82eebf2ee3'},
 'system': {'warnings': []}}

