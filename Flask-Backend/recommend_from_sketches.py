import streamlit as st
import os
import time
import pandas  as pd
import pickle
import nltk
import spacy
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from streamlit_card import card
from PIL import Image
import tensorflow as tf
import keras
import cv2
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow import expand_dims
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import joblib
from tensorflow.keras.layers import GlobalMaxPooling2D
from keras import Sequential
from keras.layers import Dense, Flatten
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image

from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import gc


import streamlit as st
import os
import time
import pandas  as pd
import pickle
import nltk
import spacy
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from streamlit_card import card
from numpy.linalg import norm
from PIL import Image
import tensorflow as tf
import keras
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import expand_dims
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from tensorflow.keras.layers import GlobalMaxPooling2D
from keras import Sequential
from keras.layers import Dense, Flatten
from numpy.linalg import norm



from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import gc

from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras import Sequential

from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D, Flatten
import cv2
import os
import random
from tqdm import tqdm
import numpy as np
from keras.preprocessing import image
import os
import random
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import streamlit as st
import shutil


from flask import Flask, jsonify, request, render_template
import requests, pickle


def recommend_images_drawings(filename):
    
    def vgg():
        vgg16 = VGG16(weights='imagenet', input_shape = (224,224,3), include_top=False)
        for layers in vgg16.layers:
            layers.trainable=False
        
        return vgg16
    def features_image_sketch():
        features_images = pickle.load(open('sketches_8000_embeddings.pkl', 'rb'))
        return features_images
        
    filenames_sketch = pickle.load(open('sketches_8000_filenames.pkl', 'rb'))
    
    def extract(img_path, vgg16):
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        expanded_img = np.expand_dims(img_array,axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        
        result = vgg16.predict(preprocessed_img, verbose=0).flatten()
        return result
    ls = []

    index_pos_shirts = []
    index_pos_pants = []
    index_pos_shoes = []
    index_pos_shorts = []
    filenames_sketches = []
    index_pos = []

      
        
    if os.path.exists('uploads3/'):
        shutil.rmtree('uploads3/')
        os.mkdir('uploads3/')
    # else:
    else:
        os.mkdir('uploads3/')

    similarity=[]
    features_images_sketch = features_image_sketch()

    # if st.checkbox('Recommend'):
        # for file in os.listdir('results/predict/crops/'):
        #     similarity = []
    for img in os.listdir('uploads3/'):

        vgg16 = vgg()
        # gc.collect()
        features = extract('uploads3/' + img, vgg16)

        for i in range(len(features_images_sketch)):

            similarity.append((cosine_similarity(features.reshape(1,-1), features_images_sketch[i].reshape(1, -1))))

        similarity = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])
        # print(similarity)

        # if st.checkbox('Show'):
            # for file in os.listdir('uploads/'):
            #     if file == 'short':
            # with st.expander('Top 10 recommndations'):
        
                # columns = st.columns(10)
        for i in range(len(10)):
            # with columns[i]:
                # print(filenames_sketch[similarity[i][0]].split('/')[-1:])
                temp = ' '.join(filenames_sketch[similarity[i][0]].split('/')[-1:])
                temp = temp.split('.')[-2]
                # print(temp)
                # print(filenames[similarity[0][0]].split('/')[-1:])
                # path='/'.join(filenames_sketch[similarity[i][0]].split('/')[-1:])
                # print('images/' +'/'.join(filenames[similarity[0][0]].split('/')[-1:]))
                print('images/' + temp + '.jpg')
                image = cv2.imread(('images/' + temp + '.jpg'))
                # image = cv2.resize(image, (224, 224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # st.write(final_df.iloc[index_pos_shirts[i],4])

                filenames_sketches.append('images/' + temp + '.jpg')

    return jsonify({
        "status": "success",
        "prediction": filenames_sketches
        # "confidence": str(classes[0][0][2]),
        # "upload_time": datetime.now()
    })        #     st.write('None')
