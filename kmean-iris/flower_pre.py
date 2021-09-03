import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import preprocessing 
from sklearn import model_selection
#đọc data
data=pd.read_csv("Iris.csv") 
#mã hóa nhãn đầu ra y
label_encoder = preprocessing.LabelEncoder() 
data['Species']= label_encoder.fit_transform(data['Species']) 
data=data.values
#chia đầu vào và đầu ra
x=data[:,1:-1]
y=data[:,-1]
#scale đầu vào
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
x = data_scaler.fit_transform(x)
#chia dữ liệu 7/3_train/test
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.3,random_state=5)
#tạo và huấn luyện bộ phân cụm
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0).fit(x_train)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
#tạo hàm tính độ chính xác bộ test
def flower_pre():
    y_pre=[1.]*kmeans.predict(x_test)
    print(y_pre)
    print(y_test)
    accuracy = 100.0 * (y_test == y_pre).sum() / x_test.shape[0]
    print("do chinh xac :",accuracy)

flower_pre()


