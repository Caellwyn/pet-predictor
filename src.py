from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from PIL import Image
import numpy as np
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import tensorflow_hub as hub
import pickle

class PetModel():
    # A model that predict whether an image contains a cat or a dog
    def __init__(self):
        """instantiate the model object"""
        self.model = self.create_model()
    
    def create_model(self):
        """Builds the model using a efficient_v2 model pretrained on imagenet as the first layers 
        and loads 1 pretrained hidden dense layer and an output layer from weights."""
              
        handle = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/classification/2"
        efficientnetlayer = hub.KerasLayer(handle,
                              trainable=False,
                                     input_shape=(200, 200, 3))

        with open('top_weights.pkl', 'rb') as f:
            layer_weights = pickle.load(f)


        input_layer = Input(shape=(200,200,3))
        dense1 = Dense(128, activation='selu')
        output = Dense(1, activation='sigmoid')

        model = Sequential([input_layer, efficientnetlayer, dense1, output])
        model.layers[-2].set_weights(layer_weights[0])
        model.layers[-1].set_weights(layer_weights[1])
        
        #It is not necessary to compile a model in order to make a prediction
        return model
            
    def convert_image(self, image):
        """Convert an image file into the right format and size for the model"""
        
        img = Image.open(image)
        img = img.resize((200,200))
        img = np.asarray(img)
        img = img.reshape((1,200,200,3))
        img = img / 255
        
        return img
        
    def predict_pet(self, image):
        """Return a prediction, dog or cat, and confidence for a passed image file"""
        
        self.img = self.convert_image(image)
        proba = self.model.predict(self.img)[0]
        
        if proba >= .6:
            certainty = int(proba * 100)
            return f"I am {certainty}% certain this is a dog"
        elif proba <= .4: 
            certainty = int((1 - proba) * 100)
            return f"I am {certainty}% certain this is a cat"
        else:
            return f"I don't have a clue what this is.  Would you like to try a different image?"
    
    def explain_prediction(self):
        """Create an image an a mask to explain model prediction"""
        
        try:
            explainer = LimeImageExplainer()
            exp = explainer.explain_instance(self.img[0],
                                            self.model.predict,
                                            num_samples=25)
            del explainer
            image, mask = exp.get_image_and_mask(0,
                                             positive_only=False, 
                                             negative_only=False,
                                             hide_rest=False,
                                             min_weight=0
                                            )
            del exp
            explanation = mark_boundaries(image, mask)
            del image
            del mask
            return explanation
        except AttributeError:
            print("Please make a prediction first")