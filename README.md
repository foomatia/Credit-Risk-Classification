# Credit-Risk-Classification

## Purpose of the analysis:
The purpose of this analysis is to use various techniques to train and evaluate a model based on loan risk. I am using historical data on lending activity from a peer-to-peer lendince services company to build a model that can identify the creditworthiness of borrowers.

We work with two values,ie. 0 in the “loan_status” column which means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting. The dataset came already with a loan_status column which had to be removed as these values needed to be predicted again. A value_counts() check showed that there are 75036 values for 0 and 2500 values for 1.

There are two parts to the analysis. 
For the first part, the data was split using the train_test_split with a random_state=1. 
Using the original data, a logistic regression model was fit. A prediction was made using the testing data. The model's performance was evaluated using balanced_accuracy_score and a confusion matrix. Finally a report was printed which showed the precision, recall and f1-score for the 0 and 1 labels. 
In the second part, the data was resampled using the RandomOverSampler from the imbalanced library. A value.counts() confirmed an equal split of the data for 0 and 1 to 56277 datapoints. Just like with the original data, a LogistigRegression model was instantiated and evaluated using the same parameters. 

## Results
Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.
Following is a summary of the results of both the datasets that were looked at.

### Machine Learning Model 1: Original Data

balanced_accuracy_score = 0.9514410363025747

confusion matrix ([18678,  81]
                  [   58, 567])
                  
              | precision | recall | f1-score | support
0             |  1.00     | 1.00   |   1.00   |   18759
--------------------------------------------------------
1             |  0.88     | 0.91   |   0.89   |     625

accuracy                           |   0.99   |   19384
macro avg     |  0.94     | 0.95   |   0.94   |   19384
weighted avg  |  0.99     | 0.99   |   0.99   |   19384

### Machine Learning Model 2: Oversampled Data

balanced_accuracy_score = 0.994180571103648

confusion matrix ([55945,   332]
                  [  323, 55954])
                  
                precision  recall  f1-score  support
0                0.99      0.99      0.99      56277
1                0.99      0.99      0.99      56277

accuracy                             0.99     112554
macro avg        0.99      0.99      0.99     112554
weighted avg     0.99      0.99      0.99     112554


## Summary

Based on the balanced_accuracy_score, Machine Learning Model 2 with the oversampled data seems to perform best. This conclusion is based on the higher score of 0.9942 compared to the score of 0.9514 for Model 1. The score takes into account both precision and recall for each label, provising a balanced measure of overall model performance. 

Looking at the precision and recall data, Model 1 using original data gives a high precision for lable 0 (health loan) but lower recall for lable 1 (high-risk loan). This model may be suitable if priority is to minimize false positives (misclassifying healthy loans as high-risk).
For Model 2 using oversampled data, prevision and recall are high for both labels. This model is suitable if a balanced performance across both labels is important or when misclassifications for high-risk and health loan need to be minimized.

Seeing that both models performed quite well in their own way, I would refrain from recommending either model without further background on the business requirements that iniated this analysis. Specific preference on a higher balanced_accuracy_score, minimizing false positives or chosing which prediction is more relevant (0s or 1s) would play a key difference to chosing which model to pick.  

