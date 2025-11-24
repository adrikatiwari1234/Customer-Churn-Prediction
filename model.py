import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN

model = pd.read_csv("tel_churn.csv")
model = model.drop('Unnamed: 0', axis=1)

x=model.drop('Churn', axis=1)

y=model['Churn']