from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from PIL import Image
import numpy as np
from tensorflow_hub import KerasLayer
import pickle

class PetModel():
    # A model that predict whether an image contains a cat or a dog
    def __init__(self):
        """Builds the model using a efficient_v2 model pretrained on imagenet as the first layers 
        and loads 1 pretrained hidden dense layer and an output layer from weights."""

        handle = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/classification/2"

        self.model = Sequential([Input(shape=(200,200,3)),
                                KerasLayer(handle,
                                           trainable=False,
                                           input_shape=(200, 200, 3)),
                                Dense(128, activation='selu'),
                                Dense(1, activation='sigmoid')
                                ])

        with open('top_weights.pkl', 'rb') as f:
            layer_weights = pickle.load(f)       

        self.model.layers[-2].set_weights(layer_weights[0])
        self.model.layers[-1].set_weights(layer_weights[1])
        
        print('created new model')
            
    def convert_image(self, img):
        """Convert an image file into the right format and size for the model"""
        
        img = Image.open(img)
        img = img.resize((200,200))
        img = np.array(img)
        print(img.shape)
        img = img[:,:,:3]
        img = img.reshape((1,200,200,3))
        img = img / 255
        
        return img
        
    def predict_pet(self, image):
        """Return a prediction, dog or cat, and confidence for a passed image file"""
        
        self.img = self.convert_image(image)
        proba = self.model.predict(self.img)[0]
        
        if proba >= .6:
            certainty = int(proba * 100)
            return f"I am {certainty}% certain this is a dog", 'dog'
        elif proba <= .4: 
            certainty = int((1 - proba) * 100)
            return f"I am {certainty}% certain this is a cat", 'cat'
        else:
            return f"I don't have a clue what this is.  Would you like to try a different image?"
    
