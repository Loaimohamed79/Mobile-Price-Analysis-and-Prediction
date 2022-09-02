
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("L:\\Some Models\\Mobile Price\\train.csv")

df.nunique()
HeadData = df.head()
TailData = df.tail()
Describe = df.describe()

df.columns


"""Dual sim mobiles ??%"""

sns.countplot(df["dual_sim"])



plt.pie(df["dual_sim"].value_counts().values, labels=[
        "Support Dual Sim", "Not Support"], autopct='%1.1f%%')

"""Dual sim and 4G supporter"""
df.groupby('dual_sim').four_g.sum().plot(kind='pie')

"""Mobile have touch screen"""
sns.countplot(df["touch_screen"])


sns.boxplot(df['price_range'], df['battery_power'])

sns.pointplot(y=df["int_memory"], x=df['price_range'])

sns.pointplot(y=df["n_cores"], x=df['price_range'])


print(df['four_g'].value_counts().values)
print(df['three_g'].value_counts())
scores4G = df['four_g'].value_counts().values
scores3G = df['three_g'].value_counts().values
labels4G = ['4G Supported', 'Not Supported']
labels3G = ['3G Supported', 'Not Supported']

"""pie chart appear mobile percent support 4G"""
plt.pie(scores4G, labels=labels4G, autopct='%1.1f%%')
"""In data we have 52.1% from mobiles support 4G """
plt.pie(scores3G, labels=labels3G, autopct='%1.1f%%')


sns.countplot(df['blue'])


X = df.drop(columns=['price_range'])
Y = df['price_range']


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3)

X_train.shape, X_test.shape


logreg = LogisticRegression()

logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
conf = confusion_matrix(y_pred, y_test)

accuracy_score(y_pred, y_test)*100


# Always takes the odd neighbors


knn = KNeighborsClassifier(n_neighbors=9, weights='uniform')
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)
confknn = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy_score(y_pred, y_test)*100)
classifireport = classification_report(y_test, y_pred)

# 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
pram = [{'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'weights':['uniform']},
        {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'weights':['distance'],
         }]

gridSearch = GridSearchCV(estimator=knn, param_grid=pram, scoring='accuracy', cv=10)
gridSearch.fit(X_train,y_train)


BestScore = gridSearch.best_score_
BestPram = gridSearch.best_params_




ErrorList = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    ErrorList.append(np.mean(y_pred != y_test))

ErrorList

plt.plot(range(1, 20), ErrorList, marker='o')

