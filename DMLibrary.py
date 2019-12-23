import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import maxabs_scale
#from sklearn.preprocessing import min_scale
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class Preprocessor:
    def __init__(self):
        pass
    
    def Scale(self,Method,data="sc"):
        if Method=="sc":
            scaler = StandardScaler()
            scaleddata = scaler.fit_transform(data)
            return scaleddata
        if Method=="mms":
            scaler = MinMaxScaler(feature_range=(0, 1)) 
            scaleddata = scaler.fit_transform(X)
            return scaleddata
        if Method=="mins":
            pass
        if Method=="maxs":
           pass
    #def encode(self,data,*cols):
        #le=LabelEncoder()
        #for col in cols:
            
X,y = load_iris(return_X_y = True) # get data inputs and outputs
S = Preprocessor()
rescaledX = S.Scale("sc",X)
np.set_printoptions(precision=3) 
print(rescaledX[0:5,:]) 