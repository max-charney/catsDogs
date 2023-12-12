"""
Made by Max Charney
"""

from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

model = tf.keras.models.load_model("model.h5")

folder_path = "pets/one.jpg"

predictions = []

st.set_page_config(
   page_title="Cat or Dog",
   page_icon="🐕",
   layout="centered"
)


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; position: absolute; top: 0;}
        </style>
        """

st.markdown(hide_menu_style, unsafe_allow_html=True)


hide_footer_style = """
        <style>
        footer {visibility: hidden; position: absolute; top: 0;}
        </style>
        """

st.markdown(hide_footer_style, unsafe_allow_html=True)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)






st.title('Dog or Cat?')




uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img)
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    predictions.append(result)
    st.image(img, caption='Uploaded Image.', width=None)

#print the list of predictions
# print(predictions)
# cats = 0
# dogs = 0


for i in range(len(predictions)):
    if predictions[i] > 0.5:
        #print('dog')
        st.subheader(":black[That is likely a dog! 🐶]")
        st.subheader(":black[(Confidence: " + (int(predictions[i]) * 10)]))
        # dogs += 1
    else:
        #print('cat')
        st.subheader(":black[That is likely a cat! 🐱]")
        st.subheader(":black[(Confidence: " + (int(100-predictions[i]) * 10)]))
        # cats += 1

st.write("This model was trained on 10,000 images of cats and dogs for 50 epochs using a Convolutional Neural Network (CNN). To use the model, simply upload your desired photo.")
st.write('My github: https://github.com/max-charney')
# print('')

# print("dogs: "+ str(dogs))
# print("cats: "+ str(cats))

# print('')

# if cats>dogs:
#     if cats-dogs==1:
#         print("There is 1 more cat than dog")
#     else:
#         print("There are "+str(cats-dogs)+" more cats than dogs")
# elif dogs>cats:
#     if dogs-cats==1:
#         print("There is 1 more dog than cat")
#     else:
#         print("There are "+str(dogs-cats)+" more dogs than cats")
# elif dogs==cats:
#     print("There are the same number of dogs and cats")
# else:
#     print("Something went wrong")
