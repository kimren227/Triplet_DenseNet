from keras.models import Model
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
import models
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

def identity_loss(y_true, y_pred):

    return K.mean(y_pred - 0 * y_true)


def bpr_triplet_loss(X):

    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss
input_placeholder = Input(shape=(224,224,3),name="placeholder")
q_input = Input(shape=(224,224,3),name='q_data')
p_input = Input(shape=(224,224,3),name='p_data')
n_input = Input(shape=(224,224,3),name='n_data')

feature_extractor = models.Feature_extractor(input_placeholder)
shared_module = Model(input_placeholder,feature_extractor)
positive_item_embedding = shared_module(p_input)
negative_item_embedding = shared_module(n_input)
query_item_embedding = shared_module(q_input)

loss = merge(
        [positive_item_embedding, negative_item_embedding, query_item_embedding],
        mode=bpr_triplet_loss,
        name='loss',
        output_shape=(1, ))

model = Model(
        input=[p_input, n_input, q_input],
        output=loss)
model.compile(loss=identity_loss, optimizer=Adam())
data = np.expand_dims(np.random.randint(1000,size=(224,224,3)),0)
for i in range(100):
    data = np.concatenate((data,np.expand_dims(np.random.randint(1000,size=(224,224,3)),0)))
X = {
        'q_data': data,
        'p_data': data,
        'n_data': data,
    }
print(model.summary())

num_epochs = 10
for i in range(num_epochs):
    model.fit(X,
                  np.ones((101,1)),
                  batch_size=64,
                  nb_epoch=1,
                  verbose=1,
                  shuffle=True)
    print('hiuhiuhiu')

