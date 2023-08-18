# Football Match Prediction
Predicts whether the home team will win the given match or not  


# Files in this repository
1. Data: Contains data obtained after feature engineering  
2. Datasets: contains the raw unprocessed data  
3. Performance: Contains the evaluation metrics/classification reports of various models used


# Dataset
Premier league match data from the past 22 years. You can download it from this [link](https://www.kaggle.com/datasets/saife245/english-premier-league?select=Datasets)  

Save the folder in your projects main folder

# Data Preparation

Run the file DataPreparation.py to prepare the final data  

Features were engineered from raw data to obtain relevant features
such as cumulative team points for the season, current form, head to head past matches etc.  

# Models

<h3>Logistic Regression</h3>

Model Accuracy: 65%  
F1 Score: **0.64**    
Run the file logistic_regression.py to train the model  
Check out the performance details [here](Performance/lr_Performance.txt)

<h4>Confusion Matrix</h4>

![logisticregression](Plots/logistic_regression_cm.png)

<h3>Random Forest</h3>

Model Accuracy: 63%   
F1 Score: **0.63**    

Run the file random_forest.py to train the model  
Check out the performance details [here](Performance/randomF_Performance.txt)

<h4>Confusion Matrix</h4>

![randomforest](Plots/random_forest_cm.png)

<h3>Support Vector Classifier</h3>

Model Accuracy: 65%  
F1 Score: **0.65**    

Run the file SVC.py to train the model  
Check out the performance details [here](Performance/SVC_Performance.txt)

<h4>Confusion Matrix</h4>

![svc](Plots/svc_cm.png)


<h3>XGBoost</h3>

Model Accuracy: 60%  
F1 Score: **0.60**    

Run the file xgb.py to train the model  
Check out the performance details [here](Performance/xgboost_Performance.txt)

![xgboost](Plots/xgboost_cm.png)

<h3>GridSearchCV on SVC Model</h3>

Scoring: F1   
Training: **0.7108**  
Test    : **0.7104**  
Accuracy: 62.55%  

Check out the performance details [here](Performance/SVC_gridsearchcv_Performance.txt)










