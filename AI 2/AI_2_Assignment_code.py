import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score


# Load dataset
dataset = pd.read_csv('Dataset of Diabetes.csv')
#Make corections In dataset
dataset['CLASS'] = dataset['CLASS'].replace({'N ': 'N', 'Y ': 'Y'})
dataset['Gender'] = dataset['Gender'].replace({'f': 'F', 'm': 'M'})
#split dataset into X and Y
X = pd.DataFrame(dataset.iloc[:, 2:13].values)
Y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_2 = LabelEncoder()
X.loc[:, 2] = labelencoder_X_2.fit_transform(X.iloc[:, 2])

labelencoder_X_1 = LabelEncoder()
X.loc[:, 1] = labelencoder_X_1.fit_transform(X.iloc[:, 1])