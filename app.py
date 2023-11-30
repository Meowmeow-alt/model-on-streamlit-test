import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np

model = load_model('model.h5')

st.title('Revenue Prediction')

temperature = st.number_input('Input Temperature')

if st.button('Predict'):
    temperature = np.array([temperature]).reshape(-1,1)
    prediction = model.predict(temperature)

    st.caption('Revenue Prediction:')
    st.success(prediction[0][0])
    st.balloons()
