import streamlit as st
from src import PetModel
from time import time
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries

print('starting over')

@st.cache_resource()
def load_model():
    return PetModel()

# @st.cache_data
def explain_prediction(_model, image):
    """Create an image an a mask to explain model prediction"""
    converted_image = _model.convert_image(image)
    try:
        explainer = LimeImageExplainer()
        exp = explainer.explain_instance(converted_image[0],
                                        _model.model.predict,
                                        num_samples=25,
                                        top_labels=1)

        new_image, mask = exp.get_image_and_mask(0,
                                            positive_only=False, 
                                            negative_only=False,
                                            hide_rest=False,
                                            min_weight=0
                                        )
        explanation = mark_boundaries(new_image, mask)
        return explanation
    except AttributeError:
        print('Model not fit yet')



#Create a header for the webpage
st.title('Is That a Dog or a Cat?')

#create an upload widget to receive an image file.  
image = st.file_uploader("Upload Your Pet!", type=["png","jpeg","jpg"])

model = load_model()

#make a prediction and return the results on a new text line
if image:
    print('Downloaded Image!')
    st.image(image)
    result = st.empty()
    result.write('Inspecting Image...')

    # predict image
    response, prediction = model.predict_pet(image)
    if prediction == 'dog':
        not_prediction = 'cat'
    else:
        not_prediction = 'dog'
    print('made a prediction')
    result.write(response)

    #show an image with a mask showing model activations
    st.header('Explaining my prediction')
    result2 = st.empty()
    result2.write('Please wait while I gather my thoughts...')

    explanation = explain_prediction(model, image)
    result2.write(f'This is what I noticed:\nGreen makes me think this is a {prediction}.\nRed makes me think this is a {not_prediction}')
    st.image(explanation, use_column_width='always')

    del explanation
    del image


