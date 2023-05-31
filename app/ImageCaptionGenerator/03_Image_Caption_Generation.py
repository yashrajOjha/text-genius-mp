import streamlit as st
from PIL import Image
import numpy as np
from beautify_icg_output import *
from get_icg_output import *
st.set_page_config(page_title="ICG",page_icon="🌉",layout="wide")
st.title("Text Genius's Caption Generator for Social Media")

def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

# Upload image
file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if file is not None:
    col1, col2 = st.columns(2)
    #displaying image
    with col1:
        st.markdown('### Input Image')
        img = load_image(file)
        st.image(img)

    _output = get_image(file)
    caption = predict_caption(_output)
    new_caption = get_suggestions(caption)
    
    with col2:
        st.markdown('### Predicted Scene Description')
        st.write(" ".join(caption.split(' ')[1:-1]).upper())

        st.markdown('### Suggested Social Media Captions')
        for ele in new_caption:
          st.markdown(f'- "{ele[0]}" ~{ele[1]}')