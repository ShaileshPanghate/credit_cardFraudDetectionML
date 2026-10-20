import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


credit_card_data = pd.read_csv('creditcard.csv')
# print(credit_card_data.head())

# checking missing values in data
# print(credit_card_data.isnull().sum())


# distribution of legit transaction (0) and fraudulent transaction(1)
print(credit_card_data['Class'].value_counts())

# 0 - 284315
# 1 - 492
# this dataset is unbalanced it's clear from above code


#  separate legit transaction and fraudulent transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print("")
print("Fraud Transactions")
print( fraud.Amount.describe())

print("")
print("Legit Transactions")
print(legit.Amount.describe())


# taking random 492 sample datapoints from legit
legit_sample = legit.sample(n=492)

# concat two dataframes fraud and legit_sample
new_data = pd.concat([legit_sample, fraud], axis=0)

print(new_data['Class'].value_counts())
# Now the new_data is balanced
# 0 - 492
# 1 - 492

X = new_data.drop(columns = 'Class', axis = 1 )
Y = new_data['Class']


# split data into  training and testing data
X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2 )

scaler = StandardScaler()
model = LogisticRegression(max_iter=1000)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model.fit(X_train, Y_train)

X_train_predict = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predict, Y_train)

print("Accuracy on training data : " , training_data_accuracy)
print()

X_test_predict = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_predict, Y_test)

print("Accuracy on testing data : " , testing_data_accuracy)
print()

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

