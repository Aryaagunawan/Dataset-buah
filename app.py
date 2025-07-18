import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ------------------ Load dan Compile Model ------------------ #
@st.cache_resource
def load_model():
    from keras.layers import InputLayer  # Tambahkan ini
    model = tf.keras.models.load_model('model_pisang.h5', custom_objects={'InputLayer': InputLayer})
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


