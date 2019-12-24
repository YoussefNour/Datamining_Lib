import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import maxabs_scale
#from sklearn.preprocessing import min_scale
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling


class Preprocessor:
    def __init__(self):
        pass

    def scale(self, Method, data):
        if Method == "sc":
            scaler = StandardScaler()
            scaleddata = scaler.fit_transform(data)
            return scaleddata
        if Method == "mms":
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaleddata = scaler.fit_transform(X)
            return scaleddata
        if Method == "mins":
            pass
        if Method == "maxs":
            pass

    def encode(self, column):
        le = LabelEncoder()
        column_encoded = le.fit_transform(column)
        return column_encoded

    def drop_missing(self, df):
        return df.dropna()


class Classifier:
    knnModel = KNeighborsClassifier()
    dtModel = DecisionTreeClassifier()
    bcModel = GaussianNB()
    rfModel = RandomForestClassifier()
    y_pred = 0
    def __init__(self):
        pass

    def fit(self, Method, X_train, y_train):
        if Method == "knn":
            print("please enter number of neighbours: ")
            n_neighbors = int(input())
            self.knnModel = KNeighborsClassifier(n_neighbors=n_neighbors)
            self.knnModel.fit(X_train, y_train)
        elif Method == "dt":
            self.dtModel = self.dtModel.fit(X_train, y_train)
        elif Method == "bc":
            self.bcModel.fit(X_train, y_train)
        elif Method == "rf":
            print("please enter number of estimators: ")
            n_estimators = int(input())
            self.rfModel = RandomForestClassifier(n_estimators=n_estimators)
            self.rfModel.fit(X_train, y_train)

    def predict(self, Method, X_test):
        if Method == "knn":
            self.y_pred = self.knnModel.predict(X_test)
        elif Method == "dt":
            self.y_pred = self.dtModel.predict(X_test)
        elif Method == "bc":
            self.y_pred = self.bcModel.predict(X_test)
        elif Method == "rf":
            self.y_pred = self.rfModel.predict(X_test)

    def score(self, y_test):
        print("Accuracy", metrics.accuracy_score(y_test, self.y_pred))

class Regressor:
    linear = LinearRegression()
    dtRegressor = DecisionTreeRegressor()
    knnRegressor = KNeighborsRegressor()
    y_pred = 0

    def __init__(self):
        pass

    def fit(self, Method, X_train, y_train):
        if Method == "line":
            self.linear.fit(X_train, y_train)
        elif Method == "dt":
            self.dtRegressor = self.dtRegressor.fit(X_train, y_train)   
        elif Method == "knn":
            print("please enter number of neighbours: ")
            n_neighbors = int(input())
            self.knnRegressor = KNeighborsRegressor(n_neighbors=n_neighbors)
            self.knnRegressor.fit(X_train, y_train)    
            
    def predict(self, Method, X_test):
        if Method == "line":
            self.y_pred = self.linear.predict(X_test)
        elif Method == "dt":
            self.y_pred = self.dtRegressor.predict(X_test)
        elif Method == "knn":
            self.y_pred = self.knnRegressor.predict(X_test)

    def score(self, Method, y_test):
        if Method == "line":
            print("Accuracy", metrics.accuracy_score(y_test, self.y_pred.round()))
        else:
            print("Accuracy", metrics.accuracy_score(y_test, self.y_pred.round()))


class Cluster:

    kmeans = 0
    y_kmeans = 0

    def __init__(self,k):
      self.kmeans = KMeans(n_clusters=int(k)) 

    def fit(self,X):
        self.kmeans.fit(X)

    def predict(self, X_test):     
        self.y_kmeans = self.kmeans.predict(X)
        print(self.y_kmeans)

# The Program Starts Here 

print("Select The Dataset :")
print("1. Iris")
print("2. Breast Cancer Wisconsin")
print("3. Diamonds")

datasetChoice = input()

preprocessor = Preprocessor()

if datasetChoice == "1":
    X = pd.read_csv('Samples/IRIS.csv')  # dataframe
    X = preprocessor.drop_missing(X)
    X['species'] = preprocessor.encode(X['species'])  
    y = X.pop('species')  # output column
elif datasetChoice == "2":
    X = pd.read_csv('Samples/Breast Cancer Wisconsin.csv')  # dataframe
    X = X.loc[:, ~X.columns.str.contains('^Unnamed')]
    X = preprocessor.drop_missing(X)
    X['diagnosis'] = preprocessor.encode(X['diagnosis'])  
    y = X.pop('diagnosis')  # output column
elif datasetChoice == "3":
    X = pd.read_csv('Samples/Diamonds.csv')  # dataframe
    X = preprocessor.drop_missing(X)
    X['cut'] = preprocessor.encode(X['cut'])  
    X['clarity'] = preprocessor.encode(X['clarity'])  
    X['color'] = preprocessor.encode(X['color'])  
    y = X.pop('price')  # output column

print("Select The Desirable Scaling Algorithm :")
print("1. Standard Scalar")
print("2. Min Max Scalar")
print("3. Max Scalar")
print("4. Min Scalar")

scalarChoice = input()

if scalarChoice == "1":
    rescaledX = preprocessor.scale("sc", X)
elif scalarChoice == "2":
    rescaledX = preprocessor.scale("mms", X)
elif scalarChoice == "3":
    rescaledX = preprocessor.scale("maxs", X)
elif scalarChoice == "4":
    rescaledX = preprocessor.scale("mins", X)
# np.set_printoptions(precision=3)
# print(rescaledX[0:5,:])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


print("What You want to apply on your Dataset  :")
print("1. Classification")
print("2. Regression")
print("3. Clustering by K-means")

algoChoice = input()

if algoChoice == "1":
    classifier = Classifier()
    print("Select The Classifier :")
    print("1. Bayesian Classifier")
    print("2. Decision Tree")
    print("3. KNN-Classification")
    print("4. Random Forest")

    classifierChoice = input()

    if classifierChoice == "1":
        classifierChoice = "bc"
    elif classifierChoice == "2":
        classifierChoice = "dt"
    elif classifierChoice == "3":
        classifierChoice = "knn"
    elif classifierChoice == "4":
        classifierChoice = "rf"

    classifier.fit(classifierChoice, X_train, y_train)
    classifier.predict(classifierChoice, X_test)
    classifier.score(y_test)

elif algoChoice == "2":
    regressor = Regressor()
    print("Select The Regressor:")
    print("1. Linear Regression")
    print("2. Polynomial Regression")
    print("3. Decision Tree")
    print("4. KNN Regressor")

    choice = input()

    if choice == "1":
        choice = "line"
    elif choice == "2":
        choice = "poly"
    elif choice == "3":
        choice = "dt"
    elif choice == "4":
        choice = "knn"
        
    regressor.fit(choice, X_train, y_train)
    regressor.predict(choice, X_test)
    regressor.score(choice,y_test)

elif algoChoice == "3":
    print("Enter The Value of K")
    k=input()
    cluster = Cluster(k)
    cluster.fit(X)
    cluster.predict(X)