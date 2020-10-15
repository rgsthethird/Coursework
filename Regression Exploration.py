
# coding: utf-8

# In[86]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn import tree
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

num_lines = sum(1 for line in open('graph.txt'))
print("Graph num_lines: "+str(num_lines))
num_lines = sum(1 for line in open('posts_train.txt'))
print("post_train num_lines: "+str(num_lines))
f = open('graph.txt','r') 
connections = {}
j=0
while True:
    j+=1
    x=f.readline()
    x = x.strip()
    if not x:break
    conn = list(map(int,x.split()))  
    if(len(conn)==2):
        if conn[0] in connections:
            connections[conn[0]].append(conn[1])
        else:
            connections[conn[0]]=[conn[1]]
f.close()
print("Lines loaded: "+str(j))
print(len(connections))
#I expected an even number of lines
#going to loop through and check
for key, value in connections.items():
    for v in value:
        if(key not in connections[v]):
            print("This key "+str(key)+"is not in its child "+str(v))


#################################################################################################################
def kaggle_RMSE(y_true,y_preds):
    return math.sqrt(mean_squared_error(y_test, y_pred)/(2))
##################################################################################################################
f = open('posts_train.txt','r')
full_posts = []
metainf = f.readline().strip().split(",")
data_size=0
deleted = 0
while True:
    x=f.readline()
    x=x.strip()
    if not x:break
    post = list(map(float,x.split(",")))
    if(len(post)==7):
        if(post[4]==0 and post[5]==0):
            #deleting records that are "masked locations"
            #print("get rid of this record "+str(post))
            deleted+=1
        else:
            data_size+=1
            full_posts.append(post)
f.close()
print("Training Data Size: " + str(data_size))
print("Deleted records: "+str(deleted))
posts = np.asarray(full_posts)
user_id = list(map(int,list(posts[:,0])))
hour_1 = list(posts[:,1])
hour_2 = list(posts[:,2])
hour_3 = list(posts[:,3])
lat=list(posts[:,4])
long = list(posts[:,5])
post_no = list(posts[:,6])
X = [[hour_1[i],hour_2[i],hour_3[i],post_no[i]] for i in range(len(hour_1))]
print(X[0])


# In[87]:


#rescale
min_max_scaler = preprocessing.MinMaxScaler()
#X_minmax = min_max_scaler.fit_transform(X)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_minmax = quantile_transformer.fit_transform(X)
#train test split
y = [(lat[i],long[i])for i in range(len(lat))]
X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.33, random_state=42)


# In[95]:


fig, ax = plt.subplots()
ax.scatter(lat,hour_1)
ax.set(xlabel='latitude', ylabel='hour_1',title='Determining trend')
ax.grid()
#fig.savefig("test.png")
plt.show()
fig, ax = plt.subplots()
ax.scatter(long,hour_1)
ax.set(xlabel='longitude', ylabel='hour_1',title='Determining trend')
ax.grid()
#fig.savefig("test.png")
plt.show()


# In[18]:


#test out different regressions
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print("Linear Regression Error: " + str(kaggle_RMSE(y_test, y_pred)))
print("")
for i in range(11):
    alpha = i/10.0
    regr = linear_model.Ridge(alpha=alpha)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print("Ridge Regression Alpha: "+str(alpha)+" Error: "+ str(kaggle_RMSE(y_test, y_pred)))
print("")
y_s = []
for i in range(1,11):
    alpha = i/10.0
    regr = linear_model.Lasso(alpha=alpha)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    y_s = y_pred
    print("Lasso Regression Alpha: "+str(alpha)+" Error: "+ str(kaggle_RMSE(y_test, y_pred)))
print("")    
clf = tree.DecisionTreeRegressor()
clf =clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Decision Tree Regressor: "+ str(kaggle_RMSE(y_test, y_pred)))
print("")

regr = LinearSVR(tol=1e-5)
lat = [i[0]for i in y_train]
long = [i[1]for i in y_train]
regr.fit(X_train,lat)
lat_pred = regr.predict(X_test)
regr.fit(X_train,long)
long_pred = regr.predict(X_test)
y_pred = [(lat_pred[i],long_pred[i])for i in range(len(lat_pred))]
print("Linear SVR Regressor: "+ str(kaggle_RMSE(y_test, y_pred)))
print("")


# In[34]:


#K-Elbow for determining best Nearest neighbour
scatter_points = []
scatter_values = []
for i in range(40,90):
    neigh = KNeighborsRegressor(n_neighbors=i)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    scatter_points.append(i)
    scatter_values.append(kaggle_RMSE(y_test, y_pred))


# In[64]:


neigh = KNeighborsRegressor(n_neighbors=75)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
print("KNeighborsRegressor Neighbours "+str(75)+" Error: "+ str(kaggle_RMSE(y_test, y_pred)))


# In[35]:


fig, ax = plt.subplots()
ax.plot(scatter_points, scatter_values)
ax.set(xlabel='Nearest Neighbours', ylabel='Kaggle Error',title='Elbow Test for best Nearest Neighbour')
ax.grid()
#fig.savefig("test.png")
plt.show()
#     print("KNeighborsRegressor Neighbours "+str(i)+" Error: "+ str(kaggle_RMSE(y_test, y_pred)))


# In[31]:


#PCA Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(X_train)
print(pca.score_samples(X_train))
print(pca.singular_values_)
X_sample = pca.transform(X_train)
X_2 = pca.transform(X_test)
neigh = KNeighborsRegressor(n_neighbors=30)
neigh.fit(X_sample, y_train)
y_pred = neigh.predict(X_2)
print("KNeighborsRegressor Neighbours "+str(30)+" Error: "+ str(kaggle_RMSE(y_test, y_pred)))


# In[84]:



regr = KNeighborsRegressor(n_neighbors=30)
lat = [i[0]for i in y_train]
long = [i[1]for i in y_train]
regr.fit(X_train,lat)
lat_pred = regr.predict(X_test)
regr.fit(X_train,long)
long_pred = regr.predict(X_test)
y_pred = [(lat_pred[i],long_pred[i])for i in range(len(lat_pred))]
lat_test = [i[0]for i in y_test]
long_test = [i[1]for i in y_test]
print("K Neighbors Regressor: "+ str(kaggle_RMSE(long_test, long_pred)))
print("")

# fit the model
regr = AdaBoostRegressor(n_estimators=100)
lat = [i[0]for i in y_train]
long = [i[1]for i in y_train]
regr.fit(X_train,lat)
lat_pred = regr.predict(X_test)
regr.fit(X_train,long)
long_pred = regr.predict(X_test)
y_pred = [(lat_pred[i],long_pred[i])for i in range(len(lat_pred))]
#scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print("Adaboost Regressor: "+ str(kaggle_RMSE(y_test, y_pred)))
print("")

regr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1)
lat = [i[0]for i in y_train]
long = [i[1]for i in y_train]
regr.fit(X_train,lat)
lat_pred = regr.predict(X_test)
regr.fit(X_train,long)
long_pred = regr.predict(X_test)
y_pred = [(lat_pred[i],long_pred[i])for i in range(len(lat_pred))]
#scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print("Gradientboost Regressor: "+ str(kaggle_RMSE(y_test, y_pred)))
print("")


# In[46]:


# from sklearn.svm import NuSVR
# for nu in range(1,11):
#     nu = nu/10
#     regr = NuSVR(gamma=1, C=1.0, nu=nu)
#     lat = [i[0]for i in y_train]
#     long = [i[1]for i in y_train]
#     regr.fit(X_train,lat)
#     lat_pred = regr.predict(X_test)
#     regr.fit(X_train,long)
#     long_pred = regr.predict(X_test)
#     y_pred = [(lat_pred[i],long_pred[i])for i in range(len(lat_pred))]
#     print("SVM NUSVR Regressor with nu "+str(nu)+ " ERROR:"+str(kaggle_RMSE(y_test, y_pred)))
    
# print("")


# In[ ]:


# from sklearn.svm import NuSVR
# c_s = [1, 10, 100, 1000]
# for c in c_s:
#     nu = nu/10
#     regr = NuSVR(gamma=1, C=c, nu=0.3)
#     lat = [i[0]for i in y_train]
#     long = [i[1]for i in y_train]
#     regr.fit(X_train,lat)
#     lat_pred = regr.predict(X_test)
#     regr.fit(X_train,long)
#     long_pred = regr.predict(X_test)
#     y_pred = [(lat_pred[i],long_pred[i])for i in range(len(lat_pred))]
#     print("SVM NUSVR Regressor with C "+str(c)+ " ERROR:"+str(kaggle_RMSE(y_test, y_pred)))
# print("")


# In[51]:


from sklearn.svm import SVR

regr = SVR(gamma=1, C=1000.0, epsilon=0.2)
lat = [i[0]for i in y_train]
long = [i[1]for i in y_train]
regr.fit(X_train,lat)
lat_pred = regr.predict(X_test)
regr.fit(X_train,long)
long_pred = regr.predict(X_test)
y_pred = [(lat_pred[i],long_pred[i])for i in range(len(lat_pred))]
print("SVr rbf Regressor: "+ str(kaggle_RMSE(y_test, y_pred)))


# In[42]:


# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# lat = [i[0]for i in y_train]
# long = [i[1]for i in y_train]

# param_grid = [{'kernel': ['rbf'], 'gamma': [1.0, 0.1, 0.01, 0.001],'C': [1, 10, 100]},
#               {'kernel': ['sigmoid'], 'gamma': [0.1, 1, 10, 100],'C': [1, 10, 100]}]
#               #{'kernel': ['poly'],'degree': [2, 3, 4, 5],'C': [1, 10, 100, 1000]}]

# clf = GridSearchCV(SVR(), param_grid, cv=5)
# clf.fit(X_train, lat)
# lat_pred = clf.predict(X_test)

# print("Best parameters set:")
# print(clf.best_params_)
# print()


# In[31]:


#load test cases to be submitted
f = open('posts_test.txt','r')
posts_tests = []
metainf = f.readline().strip().split(",")
j=0
while True:
    j+=1
    x=f.readline()
    x=x.strip()
    if not x:break
    post = list(map(float,x.split(",")))
    if(len(post)==5):
        posts_tests.append(post)
posts_tests = np.asarray(posts_tests)
user_id_test = list(map(int,list(posts_tests[:,0])))
hour_1_test = list(posts_tests[:,1])
hour_2_test = list(posts_tests[:,2])
hour_3_test = list(posts_tests[:,3])
post_no_test = list(posts_tests[:,4])
X_final= [[hour_1_test[i],hour_2_test[i],hour_3_test[i],post_no_test[i]] for i in range(len(hour_1_test))]

X_test_minmax = min_max_scaler.fit_transform(X_final)
X_test_trans = quantile_transformer.transform(X_final)


#train test split
# y = [(lat[i],long[i])for i in range(len(lat))]
# X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.33, random_state=42)
# regr = KNeighborsRegressor(n_neighbors=30)
# lat = [i[0]for i in y_train]
# long = [i[1]for i in y_train]
# regr.fit(X_train,lat)
# lat_pred = regr.predict(X_test)
# regr.fit(X_train,long)
# long_pred = regr.predict(X_test)
# y_pred = [(lat_pred[i],long_pred[i])for i in range(len(lat_pred))]
# lat_test = [i[0]for i in y_test]
# long_test = [i[1]for i in y_test]
# print("K Neighbors Regressor: "+ str(kaggle_RMSE(long_test, long_pred)))
# print("")

neigh = KNeighborsRegressor(n_neighbors=30)
neigh.fit(X_minmax, y) 
y_final = neigh.predict(X_test_minmax)

# write data to file. 
file1 = open("submission-Neigbours-example.txt","w") 
L = "Id,Lat,Lon\n"
file1.write(L) 
for i in range(len(user_id_test)):
    Line = str(user_id_test[i])+","+str(y_final[i][0])+","+str(y_final[i][1])+" \n"
    file1.write(Line) 
file1.close()


# In[79]:




