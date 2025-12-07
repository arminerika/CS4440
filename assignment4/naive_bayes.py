#-------------------------------------------------------------------------
# AUTHOR: Armin Erika Polanco
# FILENAME: naive_bayes.py
# SPECIFICATION: Naive Bayes classifier with grid search for weather prediction
# FOR: CS 4440 (Data Mining) - Assignment #4
# TIME SPENT: ~3 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#reading the training data
training_data = pd.read_csv('weather_training.csv')

#update the training class values according to the discretization (11 values only)
X_training = training_data.iloc[:, 1:-1].values  # Skip 'Formatted Date' and use all features except last column
y_training_continuous = training_data.iloc[:, -1].values  # Last column is the target variable

# Discretize the temperature values
def discretize_temp(temp):
    """Discretize temperature into 11 classes"""
    for i in range(len(classes) - 1):
        if classes[i] <= temp < classes[i + 1]:
            return classes[i]
    return classes[-1]  # For values >= 34

y_training = np.array([discretize_temp(temp) for temp in y_training_continuous])

#reading the test data
test_data = pd.read_csv('weather_test.csv')

#update the test class values according to the discretization (11 values only)
X_test = test_data.iloc[:, 1:-1].values  # Skip 'Formatted Date' and use all features except last column
y_test_continuous = test_data.iloc[:, -1].values
y_test = np.array([discretize_temp(temp) for temp in y_test_continuous])

# Track the highest accuracy
highest_accuracy = 0.0
best_s = None

#loop over the hyperparameter value (s)
for s in s_values:

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_training, y_training)

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    correct_predictions = 0
    total_predictions = len(y_test)
    
    for i in range(len(X_test)):
        # Get prediction
        prediction = clf.predict([X_test[i]])[0]
        real_value = y_test_continuous[i]  # Use continuous value for percentage calculation
        
        # Calculate percentage difference
        # Handle the case where real_value might be close to zero
        if abs(real_value) < 0.01:  # Very close to zero
            percent_diff = abs(prediction - real_value) * 100
        else:
            percent_diff = 100 * abs(prediction - real_value) / abs(real_value)
        
        # Check if prediction is within ±15% of real value
        if percent_diff <= 15:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions

    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_s = s
        print(f"Highest Naive Bayes accuracy so far: {highest_accuracy:.2f}, Parameter: s={s}")