# Sprint 3 

## Table of Content
  * [Website Building](#website-building)
  * [Feature Extraction](#feature-extraction)
  * [Flask Integration](#flask-integration)
  * [Technology Used](#technology-used)
<br><br>

# Website Building
In website building, It usisally consists of text box where the user should provide URL. The URL provided by the user is parsed into the python file "app.py". We have a machine learning model which is well trained with the necessary features. the URL parsed into the python file is fed into feature extraction pyhton file where the features will be extracted from the pyhton file which will be explained briefly in feature extraction section. After the feature extraction process is complete, the features will be fed into the ML model for prediction the model will predict the output whether the URL is geniunie or not with the accuracy percentage. The output is fed in to the website using appropiate technology.

[<img src="https://i.ibb.co/SwsLMV5/1.png" alt="1" border="0">]()

<br>

# Feature Extraction

Feature Extraction is a python code which we used to convert the given url by the user in the website to the equivalent datasetvalues in the code. This is one of the important steps in our project since our Model is trained and tested with the dataset values. 
[<img src="https://i.ibb.co/K9njRfM/feature-extraction.png" alt="1" border="0">]()

<br>

# Flask Integration

Flask is used as pipeline framework in our project where our model is connected to our website. The values given by the user in website is obtained by the flask, where with the help of feature extraction we will convert the urls to the equivalent dataset and its given to the model. The predictions obtained from our model is again pushed back to the website to show the user the output.
[<img src="https://i.ibb.co/XV2tPXX/flask111.png" alt="1" border="0">]()

<br>

# Technology Used

[<img target="_blank" src="https://i.ibb.co/7nDGQks/flask-removebg-preview.png" width=200>](https://flask.palletsprojects.com/en/2.0.x/) 
[<img target="_blank" src="https://www.freepnglogos.com/uploads/html5-logo-png/html5-logo-best-web-design-psd-html-cms-development-ecommerce-6.png" width=200>](https://www.w3schools.com/htmL/default.asp) 
 

