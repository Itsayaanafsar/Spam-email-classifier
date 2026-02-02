
# Spam Email Detection using TF-IDF and Multinomial Naive Bayes 📧🚫

A machine learning project that classifies emails as Spam or Not Spam (Ham) using Natural Language Processing (NLP) techniques.

## Overview
This project builds a spam email classifier by cleaning text data, converting it into numerical features using TF-IDF, and training a Multinomial Naive Bayes model. The model achieves high accuracy (~96%) on a large dataset and supports prediction on custom email inputs.

## Technologies Used
Python  
Pandas, NumPy  
Scikit-learn  

## Methodology
Text cleaning and preprocessing  
TF-IDF vectorization with stop-word removal and rare-word filtering  
Model training using Multinomial Naive Bayes  
Evaluation using accuracy and classification report  

## Performance
Accuracy: ~96%  
Balanced precision and recall for spam and ham classes  

## Dataset
This project uses the **190K Spam/Ham Email Dataset** from Kaggle:  
https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification?resource=download  
Place the downloaded CSV file as `spam_Emails_data.csv` in the project directory before running the script.

## How to Run
python spam_classifier.py

## Example Prediction
predict_spam("Congratulations! You have won a cash prize")

Output:  
Spam

## Author
Ayaan Afsar
