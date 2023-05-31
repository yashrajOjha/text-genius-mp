import streamlit as st
from PIL import Image
from ocr import model


st.title(":blue[Handwritten Text Recognition app]")

st.sidebar.header("Upload an Image here")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Recognize Text'):
        text = model(image)
        st.write(text)
        
        
