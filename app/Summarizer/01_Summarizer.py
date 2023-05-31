import streamlit as st
import time
from get_summarizer_output import *

# Set up Streamlit app
st.set_page_config(page_title="Text Summarizer",page_icon="📄",layout="wide")
st.title("Text Genius's Text Summarizer")
st.subheader('Developed using T5 transformer and trained on news data.')

text_input = st.text_input(label='Enter text to summarize')

if st.button('Summarize'):
    # Summarize input text
    if text_input:
      start = time.time()
      output = summarize(text_input)
      endtime = time.time()
      st.text(f'Total Time taken for Summarizing {endtime-start} seconds')
      col1, col2 = st.columns(2)
      with col1:
          st.subheader('Original Text')
          st.write(text_input)
      with col2:
          st.subheader('Summary')
          st.write(output)