# streamlit-image-classifier-demo
## A demonstration deployment using streamlit

Here's what this repo creates when deployed: [pet-predictor.herokuapp.com](https://caellwyn-pet-predictor-app-gdwpxx.streamlit.app/)

Here is template prediction website ready to deploy to heroku.  When deployed, this website allows users to upload an image of an cat or a dog and returns the model's prediction of which it is.

The src.py contains the class that does all of the modeling work.  In order to replace it with your own model, that's where you would start making changes.
Notice that the model class loads a model from saved weights.  You will not want the Heroku servers to train your model, but instead use a pretrained model.

##Deployment considerations and requirements:
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

## Deploying to Heroku

In order to deploy you need to [create a free Heroku account](https://signup.heroku.com/login).  

Once you have done that, create a new project.  Then you can connect it to your own repo, or push from your local machine directly to heroku using the Heroku Git CLI
The Heroku CLI app allows you to push local repos directly to Heroku project repos from your terminal.

As long as you have your repo properly set up, as per above, it should deploy.

### Unless...

There is a maximum 'slug size' for the free Heroku account of 500mb.  That means that all of the files you in your repo + all of the python packages combined need to be smaller than that.
If you look in the requirements.txt file you'll see that I included `tensorflow-cpu`.  Heroku does not make use of GPUs and the full Tensorflow package includes hundres of MB of code just to support GPUs. 
The `tensorflow-cpu` does not include the code for the GPUs and is thus much smaller.  You'll have to do this.  On the other hand, you don't need to change any of the imports in your code.  They should still work fine unless you are specifically trying to set Tensorflow to use a GPU.

Notice I also only included the pretrained weights for the last 3 layers of my model in the repo.  The first layers of the model are a ResNet50V2 pretrained on the ImageNet dataset.  I have my code (in the src) download those ImageNet weights during runtime to reduce the slug size further.


Here's what this repo creates when deployed: [pet-predictor.herokuapp.com](https://pet-predictor.herokuapp.com)

# Enjoy!

#
