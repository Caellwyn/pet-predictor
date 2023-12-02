# streamlit-image-classifier-demo
## A demonstration deployment using streamlit

Here's what this repo creates when deployed: [Pet-Predictor App](https://caellwyn-pet-predictor-app-b0q293.streamlit.app/)

Here is template prediction website ready to deploy to heroku.  When deployed, this website allows users to upload an image of an cat or a dog and returns the model's prediction of which it is.

The src.py contains the class that does all of the modeling work.  In order to replace it with your own model, that's where you would start making changes.
Notice that the model class loads a model from saved weights.  You will not want the Heroku servers to train your model, but instead use a pretrained model.

## Deployment considerations and requirements:
1. Procfile must have that exact name for deployment to work.  This file tells the web server how to get started: load the setup.sh and run app.py.

2. setup.sh gives the server some instructions on how to properly configure the Streamlit server

3. requirements.txt tells the web server which Python to install in order to run your code.  The server will install those packages and all dependencies.  It's necessary to specify the versions.

4. You should create a new environment for your Streamlit app.  This will make keeping track of and exporting requirements much easier.

5. app.py is the program that creates the streamlit server and controls the layout and backend code for the website.  It's called app.py because that's what Procfile tells Heroku to run.  It could be called anything as long as the .py file and the Procfile match.

## The Streamlit app in app.py

As you see, this code is very simple!  It loads the model class, which does most of the backend work, from the src.  The streamlit package is serving the website and has simple methods to add components.

### In this example:

1. st.title() is adding a text title.
2. st.file_uploader() is creating a widget to upload a file.  It's also giving that widget a name and specifying allowed file types.
3. st.empty() is creating an empty object that can be changed.  We use this to create a temporary text, "Inspecting Image..." to let users know that the model is makign a predicting behind the scenes (this process takes a moment)
Then we replace the placeholder text with the output of the predictive model object, in this case text depending on the results of the prediction.
4. st.image() displays an image.

That's it!  

You can test your new site locally by navigating to your repo locally and running `streamlit run app.py`.  That will start a streamlit server and open a browser page that shows your website.  Make sure to test your project locally before deploying it online!

## Deploying to Streamlit

In order to deploy you need to create Streamlit account and connect to your GitHub account.  

Once you have done that, just follow the prompts and connect to the repository.  The 'app.py' will be the primary app file.

In order to save space, I only included the trained weights for the last 3 layers of my model in the repo.  The first layers of the model are a ResNet50V2 pretrained on the ImageNet dataset.  I have my code (in the src) download those ImageNet weights during runtime to reduce the slug size further.


Here's what this repo creates when deployed: [Streamlit Pet Predictor]([https://pet-predictor.herokuapp.com](https://caellwyn-pet-predictor-app-b0q293.streamlit.app/)https://caellwyn-pet-predictor-app-b0q293.streamlit.app/)

# Enjoy!

#
