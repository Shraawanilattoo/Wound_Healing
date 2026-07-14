import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data for diabetic animals
data = {
    "Time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Lactate_1": [0.1045, 30.51241, 19.12105, 12.72915, 8.49883, 7.10641, 4.42765, 1.29801, 0.76756, -0.01485],
    "Wound_Closure_1": [ 0,1.73977,3.47954,5.21931,6.95908,7.65415,8.34922,16.04526333 ,23.74130667 ,31.43735],
    "Lactate_2": [3.40456, 13.9922, 13.25121, 11.64315, 11.07636, 10.38842, 10.07623, 9.57512, 9.3853, 8.83005],
    "Wound_Closure_2": [ 0,5.61711,11.23422,16.85133,22.46844,37.216075 ,51.96371,52.428,52.89229,  53.35658 ],
    "Lactate_3": [4.13573,4.11922,5.52599,2.28754,1.18606,1.23556,0.63325,0,0,0],
    "Wound_Closure_3": [ 0,3.3421375,6.684275,10.0264125,13.36855,23.329965,33.29138,36.24123667, 39.19109333, 42.14095],
    "Lactate_4": [0,9.42907,11.92618,9.28489,5.93426,3.8639,3.05652,1.68973,1.19954,0.16148],
    "Wound_Closure_4": [ 0,10.6209625,21.241925,31.8628875,42.48385,43.062815,43.64178,49.07621667 ,54.51065333, 59.94509],
    "Lactate_5": [1.19515,34.87792,27.67914,25.53388,21.32077,18.37933,16.4663,14.65278,11.43489,8.57086],
    "Wound_Closure_5": [ 0,1.281535,2.56307,3.844605,5.12614,24.156915,43.18769,46.58155333,49.97541667, 53.36928],
    "Lactate_6": [0,24.3798,15.68485,12.81616,10.17374,5.70505,3.77374,2.56162,0,0],
    "Wound_Closure_6": [ 0,1.50892167,3.01784333,4.526765,6.03568667, 12.59926262,19.16283857 ,26.15949021, 33.15614185, 40.15279349],
    "Lactate_7": [0,23.0197,24.12879,27.20152,25.22576,25.10455,26.99545,25.05,27.57727,22.70033778],
    "Wound_Closure_7": [ 0,8.6256475,17.251295,25.8769425,34.50259,42.533545,50.5645,52.25030333 ,53.93610667, 55.62191],
    "Lactate_8": [0,1.76676,7.93197,13.70845,16.31293,6.38484,4.05248,0,0,0],
    "Wound_Closure_8": [ 0,1.0739625,2.147925,3.2218875,4.29585,16.057505,27.81916,31.44868333 ,35.07820667 ,38.70773],
    "Lactate_9": [0,21.61481,2.18765,7.0963,4.60741,3.51111,1.8321,0.64691,0.07407,0],
    "Wound_Closure_9": [ 0,4.9189175,9.837835,14.7567525,19.67567,27.585415,35.49516,42.62629667, 49.75743333, 56.88857],
    "Lactate_10": [0,25.0255,25.77231,25.57741,6.80146,7.90893,3.69763,2.8816,0.45537,2.45902],
    "Wound_Closure_10": [ 0,5.863325,11.72665,17.589975,23.4533,27.57272,31.69214,34.36913667 ,37.04613333, 39.72313]
}

df = pd.DataFrame(data)

df_1 = df[['Time', 'Lactate_1', 'Wound_Closure_1']].rename(columns={'Lactate_1': 'Lactate', 'Wound_Closure_1': 'Wound_Closure'})
df_2 = df[['Time', 'Lactate_2', 'Wound_Closure_2']].rename(columns={'Lactate_2': 'Lactate', 'Wound_Closure_2': 'Wound_Closure'})
df_3 = df[['Time', 'Lactate_3', 'Wound_Closure_3']].rename(columns={'Lactate_3': 'Lactate', 'Wound_Closure_3': 'Wound_Closure'})
df_4 = df[['Time', 'Lactate_4', 'Wound_Closure_4']].rename(columns={'Lactate_4': 'Lactate', 'Wound_Closure_4': 'Wound_Closure'})
df_5 = df[['Time', 'Lactate_5', 'Wound_Closure_5']].rename(columns={'Lactate_5': 'Lactate', 'Wound_Closure_5': 'Wound_Closure'})
df_6 = df[['Time', 'Lactate_6', 'Wound_Closure_6']].rename(columns={'Lactate_6': 'Lactate', 'Wound_Closure_6': 'Wound_Closure'})
df_7 = df[['Time', 'Lactate_7', 'Wound_Closure_7']].rename(columns={'Lactate_7': 'Lactate', 'Wound_Closure_7': 'Wound_Closure'})
df_8 = df[['Time', 'Lactate_8', 'Wound_Closure_8']].rename(columns={'Lactate_8': 'Lactate', 'Wound_Closure_8': 'Wound_Closure'})
df_9 = df[['Time', 'Lactate_9', 'Wound_Closure_9']].rename(columns={'Lactate_9': 'Lactate', 'Wound_Closure_9': 'Wound_Closure'})
df_10 = df[['Time', 'Lactate_10', 'Wound_Closure_10']].rename(columns={'Lactate_10': 'Lactate', 'Wound_Closure_10': 'Wound_Closure'})

df_train = pd.concat([df_1, df_2,df_3,df_4,df_5,df_6,df_7,df_8], ignore_index=True)  
#df_train = df_train.dropna(subset=["Wound_Closure"])  # Remove NaN targets

df_test = pd.concat([df_9,df_10], ignore_index=True)  
#df_test = df_test.dropna(subset=["Wound_Closure"])  # Remove NaN targets

X_train = df_train[["Time", "Lactate"]]
y_train = df_train["Wound_Closure"]

X_test = df_test[["Time", "Lactate"]]
y_test = df_test["Wound_Closure"]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Test Predictions:", y_pred)

import matplotlib.pyplot as plt

plt.plot(y_pred, label='Predicted Values', marker='o', linestyle='-', color='b')
plt.plot(y_test, label='True Values', marker='x', linestyle='--', color='r')

plt.xlabel('Sample Index')
plt.ylabel('Value')         

plt.title('Comparison of Predicted and True Values')
plt.legend()
plt.show()
