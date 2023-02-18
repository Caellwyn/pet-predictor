import streamlit as st
from src import PetModel
from time import time
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries

print('starting over')

@st.cache_resource()
def load_model():
    return PetModel()

def explain_prediction(model, image):
    """Create an image an a mask to explain model prediction"""
    image = model.convert_image(image)
    try:
        explainer = LimeImageExplainer()
        exp = explainer.explain_instance(image[0],
                                        model.model.predict,
                                        num_samples=100)

        image, mask = exp.get_image_and_mask(0,
                                            positive_only=False, 
                                            negative_only=False,
                                            hide_rest=False,
                                            min_weight=0
                                        )
        explanation = mark_boundaries(image, mask)
        del explainer
        del exp
        del image
        del mask
        return explanation
    except AttributeError:
        print('Model not fit yet')



if __name__ == "__main__":

    #Create a header for the webpage
    st.title('Is That a Dog or a Cat?')

    #create an upload widget to receive an image file.  
    image = st.file_uploader("Upload Your Pet!", type=["png","jpeg","jpg"])


    #make a prediction and return the results on a new text line
    if image:
        print('Downloaded Image!')
        st.image(image)
        result = st.empty()
        result.write('Inspecting Image...')

        #start loading model
        start = time()
        model = load_model()
        print(f'model loaded in {time() - start}')

        # predict image
        start = time()
        response = model.predict_pet(image)
        print(f'model predicted in {time() - start}')
        print('made a prediction')
        result.write(response)

        #show an image with a mask showing model activations
        st.header('Explaining my prediction')
        result2 = st.empty()
        result2.write('Please wait while I gather my thoughts...')

        start = time()
        explanation = explain_prediction(model, image)
        print(f'model explained predictions in {time() - start}')
        print('created an explanation image')
        result2.write('This is what I noticed:\nGreen makes me think this is a dog.\nRed makes me think this is a cat')
        st.image(explanation, use_column_width='always')

        del explanation
        del image
        del model


