# Import necessary libraries
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

from keras.models import Sequential
from keras.layers import Dense

# 1. Read the data
ds = pd.read_csv('Dataset of Diabetes.csv')

# 2. Make necessary corrections on the data
ds['Gender'] = ds['Gender'].replace({'f': 'F', 'm': 'M'})
ds['CLASS'] = ds['CLASS'].replace({'N ': 'N', 'Y ': 'Y'})
# Convert 'Gender' column to one-hot encoding
ds = pd.get_dummies(ds, columns=['Gender'])

# Map the class labels to integers
class_mapping = {'Y': 0, 'N': 1, 'P': 2}
ds['CLASS'] = ds['CLASS'].map(class_mapping)

# Drop rows with 'NaN' values
ds = ds.dropna()

#split dataset into X and Y
X = pd.DataFrame(ds.iloc[:, 2:13].values)
Y = ds.iloc[:, 13].values

# 3. Create training and testing datasets
X = ds.drop('CLASS', axis=1)
y = ds['CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build and train the neural network
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)

# 5. Calculate the accuracy and precision of the neural network using a confusion matrix
y_pred_prob = model.predict(X_test)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
