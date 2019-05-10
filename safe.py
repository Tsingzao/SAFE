
# coding: utf-8

# In[1]:

import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
os.system('echo $CUDA_VISIBLE_DEVICES')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# In[2]: Model Defination

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.utils.np_utils import to_categorical
from utils import *

def denoise_autoencoder(shape):
    input_img = Input(shape=shape)  

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder, Model(input_img,encoded)

def update_weight(x_train, decoded_imgs, lam):
    print('Calculate Sample Weigth...')
    for j in range(x_train.shape[0]):
        if mean_square_loss(decoded_imgs[j], x_train[j]) > 2*lam:
            sample_weights[j] = 0
        elif mean_square_loss(decoded_imgs[j], x_train[j]) < lam:
            sample_weights[j] = 1
        else:
            sample_weights[j] = 2-mean_square_loss(decoded_imgs[j], x_train[j])/lam    
    return sample_weights
    
def update_lambda(i):
    return 0.3    
    
# In[3]:

autoencoder, encoder = denoise_autoencoder(shape)
autoencoder.compile(optimizer='adadelta', loss='mse')
 
sample_weights = np.ones((len(x_train),))
# In[5]:

for i in range(iters):
    
    print("Outer Epoch {}/{}".format(i+1,epochs))
    autoencoder.fit(x_train, x_train, epochs=5, batch_size=256, shuffle=True,
                    validation_data=(x_test, x_test), sample_weight=sample_weights)
    decoded_imgs = autoencoder.predict(x_test)
    if show_result:
        show_image(x_test, decoded_imgs)
        
    decoded_imgs = autoencoder.predict(x_train)
    lam = update_lambda(i)
    
    if self_pace:
        sample_weights = update_weight(decoded_imgs, x_train, lam)
    
encoded_train= encoder.predict(x_train,verbose=1,batch_size=256)
encoded_test = encoder.predict(x_test,verbose=1,batch_size=256)


# In[10]:
classifier = fc_classifier(encoded_train.shape[1])
classifier.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
classifier.fit(encoded_train, label_train, verbose=1, 
               batch_size=256, epochs=300, validation_data=(encoded_test, label_test))
