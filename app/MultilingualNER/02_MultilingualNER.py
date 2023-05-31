import streamlit as st
from predict import predict_nertags
st.set_page_config(page_title="InKaNER", layout="wide")
st.title('InKaNER - India 🇮🇳 Ka Named Entity Recognizer')
st.markdown("- An XLM-Roberta Model trained to recognize entities in **6 Indian 🇮🇳 Languages**")
st.markdown("- Try out the model in **_Hindi, Kannada, Bengali, Marathi, Tamil, and Telugu!_**")
st.markdown("- While tesing out please enter full name of people, full forms or organizations.")
name = st.text_input("Enter Your Statement", "Please enter your statement")

print_string = """
<style>
.entity-person {
    background-color: #87CEEB;
    color:black;

}

.entity-organization {
    background-color: #FF6D60;
    color:black;
}

.entity-location {
    background-color: #F3E99F;
    color: black;
}
</style>

<ul>
"""

if(st.button('Submit')):
    result = name.title()
    ner_res =  predict_nertags(result)
    # split_ = result.split(' ')
    # combined_ = [print_string,'<p>']
    # print(ner_res)
    # for word in split_:
    #     if word in ner_res:
    #         if ner_res[word]=='Person':
    #             combined_.append(f"<span class='entity-person'> {word} ({ner_res[word]})</span>")
    #         if ner_res[word]=='Location':
    #             combined_.append(f"<span class='entity-location'> {word} ({ner_res[word]})</span>")
    #         if ner_res[word]=='Organization':
    #             combined_.append(f"<span class='entity-organization'>{word} ({ner_res[word]})</span>")
    #     else:
    #         combined_.append(word)
    # combined_.append('</p>')
    # final_ = " ".join(combined_)

    for key,values in ner_res.items():
        if values=='Person':
            print_string+=f"<li><span class='entity-person'> {key} ({values})</span></li>"
        if values=='Location':
            print_string+=f"<li><span class='entity-location'> {key} ({values})</span></li>"
        if values=='Organization':
            print_string+=f"<li> <span class='entity-organization'>{key} ({values})</span></li>"
        
    st.markdown(print_string+"</ul>",unsafe_allow_html=True)

# removing hamburger and streamlit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# adding a footer
footer="""<style>
a:link , a:visited{
color: white;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
opacity:0.75
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;' href="https://github.com/yashrajOjha" target="_blank">Yash Raj Ojha</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

