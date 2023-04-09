<h2> Udacity-Project2-Disaster-Response</h2>

<img width="1004" alt="app_1" src="https://user-images.githubusercontent.com/62621380/230476171-6ef75049-8284-4dd5-9cc2-7d3ae38041d1.PNG">


<h4>Description</h4>
This Project is part of Data Science Nanodegree Program by Udacity. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided in the following key sections:

Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
Build a machine learning pipeline to train the which can classify text message in various categories
Run a web app which can show model results in real time


<h4>File Descriptions</h4>

Folder: data<br>
<b>disaster_categories.csv :</b> categories of the messages<br>
<b>disaster_messages.csv :</b> real messages sent during disaster events (provided by Figure Eight)<br>
<b>process_data.py :</b> ETL pipeline used to load, clean, extract feature and store data in SQLite database<br>
<b>ETL Pipeline Preparation.ipynb :</b> Jupyter Notebook used to prepare ETL pipeline<br>
<b>beso_disaster_response_db.db :</b> cleaned data stored in SQlite database<br>

Folder: models<br>
<b>train_classifier.py :</b> ML pipeline used to load cleaned data, train model and save trained model as pickle (.pkl) file for use later<br>
<b>classifier.pkl :</b> pickle file contains trained model<br>
<b>ML Pipeline Preparation.ipynb :</b> Jupyter Notebook used to prepare ML pipeline<br>

Folder: app<br>
<b>run.py :</b> python script to launch web application.<br>
<b>Folder: templates :</b> web dependency files (go.html & master.html) required to run the web application.<br>

Folder: Screenshots<br>
<b>app_1 & app_2 & app_3 :</b> countain screenshouts for the web app that we use <br>
<b>process_data_code :</b> it is an imge of excuting the code for processing data<br>
<b>train_classifier_code :</b> it is an imge of excuting the code for classifier trainning <br>
<b>runing web app :</b> it is an imge of excuting the code for the webapp<br>


<h4> Installation: </h4>
There should be no extra libraries required to install apart from those coming together with Anaconda distribution. There should be no issue to run the codes using Python 3.5 and above


<h4> Executing Program: </h4>
1.run ETL pipeline that cleans data and stores in database:<br>
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/beso_disaster_response_db.db<br>
2.run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file:<br>
python models/train_classifier.py data/beso_disaster_response_db.db models/classifier.pkl<br>
3.Run the following command in the app's directory to run web app :<br>
python run.py<br>
4.visit : http://0.0.0.0:3001/



<h4>Additional Material</h4>
In the data and models folder you can find two jupyter notebook that can help you understand the model and tack you step by step of how it work:

ETL Preparation Notebook: learn everything about the implemented ETL pipeline
ML Pipeline Preparation Notebook: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn
You can use ML Pipeline Preparation Notebook to re-train the model or tune it through a dedicated Grid Search section.


Important Files
app/templates/*: templates/html files for web app

data/process_data.py: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

models/train_classifier.py: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use

run.py: This file can be used to launch the Flask web app used to classify disaster messages


<h4>Screenshots</h4>
<h6> The fllowing screanshot is for clean,proccing data and store in Sql database
<img width="799" alt="process_data_code" src="https://user-images.githubusercontent.com/62621380/230484194-5d5bdebe-5e5a-4910-a9ec-6f9b496e3032.PNG">


<h6> Next screenshot is for classifier and tranning modle </h6>
<img width="700" alt="train_classifier_code_1" src="https://user-images.githubusercontent.com/62621380/230765565-674ae2d1-8b47-4556-bdf7-b53668470bce.PNG">


<img width="610" alt="train_classifier_code_2" src="https://user-images.githubusercontent.com/62621380/230765581-9c4858e2-1055-4156-a164-d1d8159a8c3c.PNG">


<img width="615" alt="train_classifier_code_3" src="https://user-images.githubusercontent.com/62621380/230765595-2909f0d8-1fea-49ba-ad3b-386e56ff93d2.PNG">


<img width="667" alt="train_classifier_code_4" src="https://user-images.githubusercontent.com/62621380/230765607-21d26a2a-ec40-4f4f-b0de-02be2a148872.PNG">


<img width="639" alt="train_classifier_code_5" src="https://user-images.githubusercontent.com/62621380/230765616-2738d486-5a12-4de3-a6dd-687c9add0ec4.PNG">

You can view more in folder models under the name : train classifier


<h6> Final screenshot to start web app </h6>
<img width="465" alt="runing web app" src="https://user-images.githubusercontent.com/62621380/230485173-04ad9c28-2b48-4b66-a6ee-3d3c4d2a9faa.PNG">


<h6>These screenshots are from the web app</h6>
<img width="1004" alt="app_1" src="https://user-images.githubusercontent.com/62621380/230476171-6ef75049-8284-4dd5-9cc2-7d3ae38041d1.PNG">
<img width="1027" alt="app 2" src="https://user-images.githubusercontent.com/62621380/230485441-581f41ff-44b8-4a6f-804b-712a34ff43f9.PNG">
<img width="868" alt="app 3" src="https://user-images.githubusercontent.com/62621380/230485455-74b7bc80-dd30-48cb-99b7-a90993a0396f.PNG">


<h4>Searching result in the web app</h4>

<img width="948" alt="Trying model 1" src="https://user-images.githubusercontent.com/62621380/230765733-d62d0690-292a-4576-b0a7-6da68802eb02.PNG">
<img width="869" alt="model classfication result 1" src="https://user-images.githubusercontent.com/62621380/230765741-e3a79b91-a2b2-4686-846b-11e65e255c35.PNG">
<img width="859" alt="model classfication result 2" src="https://user-images.githubusercontent.com/62621380/230765745-34da7620-dbf3-4a17-b5e9-d7beac047173.PNG">
<img width="856" alt="model classfication result 3" src="https://user-images.githubusercontent.com/62621380/230765750-1b91eeae-302c-4e0a-bcaa-9a0fb4d301db.PNG">


