import streamlit as st
import numpy as np
import pandas as pd
import cv2
import zipfile
import tempfile
from keras.models import load_model
import cv2
import keras 
from keras.layers import *
from keras.models import *
from keras import backend as K
import imageio as iio
from skimage import io

st.write("# Age and Gender Prediction")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    stream = st.file_uploader('TF.Keras model file (.h5.zip)', type='zip')
    if stream is not None:
        myzipfile = zipfile.ZipFile(stream)
        with tempfile.TemporaryDirectory() as tmp_dir:
            myzipfile.extractall(tmp_dir)
            root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
            model_dir = os.path.join(tmp_dir, root_folder)
    #st.info(f'trying to load model from tmp dir {model_dir}...')
    model = tf.keras.models.load_model(model_dir)
    data = cv2.imread(uploaded_file)
          
    def age_group(age):
        if age >=0 and age < 18:
            return 1
        elif age < 30:
            return 2
        elif age < 80:
            return 3
        else:
            return 4
    def get_age(distr):
        distr = distr*4
        if distr >= 0.65 and distr <= 1.4:return "0-18"
        if distr >= 1.65 and distr <= 2.4:return "19-30"
        if distr >= 2.65 and distr <= 3.4:return "31-80"
        if distr >= 3.65 and distr <= 4.4:return "80 +"
        return "Unknown"
    def get_gender(prob):
        if prob < 0.5:
            return "Male"
        else: 
            return "Female"
    def get_result(data):
        sample = io.imread(data, as_gray=True)
        sample=cv2.resize(sample,(64, 64))
        val = model.predict(np.array([ sample ]))
        print(val)
        age = get_age(val[0])
        gender = get_gender(val[1])
        st.write("Values",val,"\nPredicted Gender:",gender,"Predicted Age:",age)
   
    #display(path)
    #print("Actual Gender:",get_gender(genders[idx]),"Age:",ages[idx])
    res = get_result(data)
