import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import matplotlib.pyplot as plt

# Set Streamlit options
st.set_option('deprecation.showfileUploaderEncoding', False)

# Set the Page Title and Configuration
st.set_page_config(
    page_title="Hindi Character Recognition",
    page_icon="ICON_PATH_OR_URL"
)

# Load the Model
@st.cache(allow_output_mutation=True)
def load_model():
    with st.spinner("Model is being loaded..."):
        model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# Define Hindi characters list
hindi_characters = ['ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'क', 'न', 'प', 'फ', 'क', 'य', 'र', 'ल', 'व', 'ख', 'श', 'ष', 'स', 'ह', 'ऋ', 'त्र', 'श', 'ग', 'घ', 'ज', 'झ', 'e', '१', '२', '३', '४', '५', '६', '७', '८', 'ब', 'भ', 'म', 'ब', 'ङ', 'च', 'उ']

# Use the loaded model
