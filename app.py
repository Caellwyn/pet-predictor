import streamlit as st
from src import PetModel

#Create the model
if not model:
    model = PetModel()
    print('Created Model')

#Create a header for the webpage
st.title('Is That a Dog or a Cat?')

#create an upload widget to receive an image file.  
#Give it a name and define allowed file types
image = st.file_uploader("Upload Your Pet!", type=["png","jpeg","jpg"])
print('downloaded image')

#make a prediction and return the results on a new text line
if image:
    result = st.empty()
    st.image(image)
    result.write('Inspecting Image...')
    response = model.predict_pet(image)
    print('made a prediction')
    result.write(response)

#show an image with a mask showing model activations
if image:
    st.header('If you wait a minute or two, I can explain my prediction')
    result2 = st.empty()
    result2.write('Please wait while I gather my thoughts...')
    explanation = model.explain_prediction()
    print('created an explanation image')
    result2.write('This is what I noticed:\nGreen makes me think this is a dog.\nRed makes me think this is a cat')
    st.image(explanation, use_column_width='always')
    del explanation