import keras
from keras.applications import VGG16
from keras import models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from collections import OrderedDict
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# loading model
modelvgg = load_model('./image_caption_generation/VanillaModel')
model = load_model('./image_caption_generation/vgg16_trained.h5',compile=False) #same file path

# loading tokenizer
f = open('./image_caption_generation/tokenizer.pckl', 'rb')
tokenizer = pickle.load(f)
f.close()

index_word = dict([(index,word) for word, index in tokenizer.word_index.items()])

def get_image(image_id):
    npix = 224
    target_size = (npix,npix,3)
    image = load_img(image_id, target_size=target_size)
    image = img_to_array(image)
    nimage = preprocess_input(image)
    y_pred = modelvgg.predict(nimage.reshape( (1,) + nimage.shape[:3]))
    _output = y_pred
    return _output

maxlen = 30

def predict_caption(image):
    '''
    image.shape = (1,4462)
    '''

    in_text = 'startseq'

    for iword in range(maxlen):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence],maxlen)
        yhat = model.predict([image,sequence],verbose=0)
        yhat = np.argmax(yhat)
        newword = index_word[yhat]
        in_text += " " + newword
        if newword == "endseq":
            break
    return in_text

# caption = predict_caption(_output.reshape(1,len(_output)))
# print(caption)
# model.summary()