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
<img width="702" alt="train_classifier_code" src="https://user-images.githubusercontent.com/62621380/230484844-725e23a4-5526-4dea-b0ad-015dfc13a855.PNG">
