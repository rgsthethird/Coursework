
# coding: utf-8

# In[36]:


from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from statistics import median
import weightedstats as ws

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
print("Connection Lines loaded: "+str(j))
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
print("Deleted Null Islands: "+str(deleted))
posts = np.asarray(full_posts)
user_id = list(map(int,list(posts[:,0])))
hour_1 = list(posts[:,1])
hour_2 = list(posts[:,2])
hour_3 = list(posts[:,3])
hour_dict = {}
for i in range(len(user_id)):
    hour_dict[user_id[i]] = [hour_1[i],hour_2[i],hour_3[i]]

lat=list(posts[:,4])
long = list(posts[:,5])

location_dict = {}
for i in range(len(user_id)):
    location_dict[user_id[i]] = [lat[i],long[i]]

post_no = list(posts[:,6])

posts_dict = {}
for i in range(len(user_id)):
    posts_dict[user_id[i]] = [post_no[i]]


# In[37]:


#apply features from friends
med_lats=[]
med_longs = []
f_avhr =[]
k=0
for user in user_id:
    med_geo = []
    av_hr = 0
    if user in connections.keys():
        user_con = connections[user]
        for friend in user_con:
            if friend in location_dict.keys():
                med_geo.append(location_dict[friend])
                av_hr = (av_hr+(sum(hour_dict[i])/len(hour_dict[i])))/2
    curr_lat = [i[0] for i in med_geo]
    curr_long = [i[1] for i in med_geo]
    f_avhr.append(av_hr)
    if user not in location_dict.keys(): print("Something is wrong")
    if(len(med_geo)!=0):
        med_lats.append(median(curr_lat))
        med_longs.append(median(curr_long))
#         med_lats.append(ws.weighted_median(curr_lat,weights=post_number))
#         med_longs.append(ws.weighted_median(curr_long,weights=post_number))
    else:
        k+=1
        med_lats.append(0)
        med_longs.append(0)
    
print(k)
X = [[hour_1[i],hour_2[i],hour_3[i],post_no[i],med_lats[i],med_longs[i]] for i in range(len(hour_1))]


# In[31]:


# #apply features from friends
# med_lats=[]
# med_longs = []
# k=0
# for user in user_id:
#     med_geo = []
#     post_number =[]
#     if user in connections.keys():
#         user_con = connections[user]
#         for friend in user_con:
#             if friend in connections.keys():
#                 friend_con = connections[friend]
#                 for friends_friend in friend_con:
#                     if friends_friend in location_dict.keys():
#                         med_geo.append(location_dict[friends_friend])
#     curr_lat = [i[0] for i in med_geo]
#     curr_long = [i[1] for i in med_geo]
#     if user not in location_dict.keys(): print("Something is off")
#     if(len(med_geo)!=0):
#         med_lats.append(median(curr_lat))
#         med_longs.append(median(curr_long))
# #         med_lats.append(ws.weighted_median(curr_lat,weights=post_number))
# #         med_longs.append(ws.weighted_median(curr_long,weights=post_number))
#     else:
#         k+=1
#         med_lats.append(0)
#         med_longs.append(0)
    
# print(k)
# X = [[hour_1[i],hour_2[i],hour_3[i],post_no[i],med_lats[i],med_longs[i]] for i in range(len(hour_1))]


# In[38]:


#rescale
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)
#train test split
y = [(lat[i],long[i])for i in range(len(lat))]
X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.33, random_state=42)
regr = KNeighborsRegressor(n_neighbors=25)
# regr.fit(X_train, y_train)

lat = [i[0]for i in y_train]
long = [i[1]for i in y_train]
regr.fit(X_train,lat)
lat_pred = regr.predict(X_test)
regr.fit(X_train,long)
long_pred = regr.predict(X_test)
y_pred = [(lat_pred[i],long_pred[i])for i in range(len(lat_pred))]
# lat_test = [i[0]for i in y_test]
# long_test = [i[1]for i in y_test]

print("KNeighbors Regressor Neighbours Error: "+ str(kaggle_RMSE(y_test, y_pred)))


# In[33]:


#K-Elbow for determining best Nearest neighbour
scatter_points = []
scatter_values = []
for i in range(10,60):
    neigh = KNeighborsRegressor(n_neighbors=i)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    scatter_points.append(i)
    scatter_values.append(kaggle_RMSE(y_test, y_pred))
fig, ax = plt.subplots()
ax.plot(scatter_points, scatter_values)
ax.set(xlabel='Nearest Neighbours', ylabel='Kaggle Error',title='Elbow Test for best Nearest Neighbour')
ax.grid()
#fig.savefig("test.png")
plt.show()


# In[34]:


# from sklearn.neural_network import MLPRegressor

# regr = MLPRegressor()
# regr.fit(X_train, y_train)

# lat = [i[0]for i in y_train]
# long = [i[1]for i in y_train]
# regr.fit(X_train,lat)
# lat_pred = regr.predict(X_test)
# regr.fit(X_train,long)
# long_pred = regr.predict(X_test)
# y_pred = [(lat_pred[i],long_pred[i])for i in range(len(lat_pred))]
# lat_test = [i[0]for i in y_test]
# long_test = [i[1]for i in y_test]

# print("MLP Regressor Neighbours Error: "+ str(kaggle_RMSE(y_test, y_pred)))


# In[15]:


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


med_lats_test=[]
med_longs_test = []

k=0
for user in user_id_test:
    med_geo_test = []
    if user in connections.keys():
        user_con = connections[user]
        for friend in user_con:
            if friend in location_dict.keys():
                med_geo_test.append(location_dict[friend])
    curr_lat = [i[0] for i in med_geo_test]
    curr_long = [i[1] for i in med_geo_test]
    if(len(med_geo_test)!=0):
        med_lats_test.append(median(curr_lat))
        med_longs_test.append(median(curr_long))
    else:
        k+=1
        med_lats_test.append(0)
        med_longs_test.append(0)
        

# for user in user_id_test:
#     med_geo_test = []
#     if user in connections.keys():
#         user_con = connections[user]
#         for friend in user_con:
#             if friend in connections.keys():
#                 friend_con = connections[friend]
#                 for friends_friend in friend_con:
#                     if friends_friend in location_dict.keys():
#                         med_geo_test.append(location_dict[friends_friend])
#     curr_lat_test = [i[0] for i in med_geo_test]
#     curr_long_test = [i[1] for i in med_geo_test]
#     if(len(med_geo_test)!=0):
#         med_lats_test.append(median(curr_lat_test))
#         med_longs_test.append(median(curr_long_test))
# #         med_lats.append(ws.weighted_median(curr_lat,weights=post_number))
# #         med_longs.append(ws.weighted_median(curr_long,weights=post_number))
#     else:
#         k+=1
#         med_lats_test.append(0)
#         med_longs_test.append(0)
        
        
print(k)

X_final= [[hour_1_test[i],hour_2_test[i],hour_3_test[i],post_no_test[i],med_lats_test[i],med_longs_test[i]] for i in range(len(hour_1_test))]

X_test_minmax = min_max_scaler.fit_transform(X_final)

neigh = KNeighborsRegressor(n_neighbors=40)
#neigh = MLPRegressor()
# neigh.fit(X_minmax, y) 
lat = [i[0]for i in y_train]
long = [i[1]for i in y_train]
neigh.fit(X_minmax,lat)
lat_pred = regr.predict(X_test_minmax)
neigh.fit(X_minmax,long)
long_pred = neigh.predict(X_test_minmax)
y_final = [(lat_pred[i],long_pred[i])for i in range(len(lat_pred))]
y_final = neigh.predict(X_test_minmax)

# write data to file. 
file1 = open("submission-Neigbours-example.txt","w") 
L = "Id,Lat,Lon\n"
file1.write(L) 
for i in range(len(user_id_test)):
    Line = str(user_id_test[i])+","+str(y_final[i][0])+","+str(y_final[i][1])+" \n"
    file1.write(Line) 
file1.close()


# In[24]:


# import numpy as np
# from sklearn.neural_network import BernoulliRBM
# model = BernoulliRBM(n_components=6)
# model.fit_transform(X_minmax)
# print(model.get_params())


# In[26]:


# from sklearn.ensemble import ExtraTreesRegressor
# regr = ExtraTreesRegressor(n_estimators=1000)
# lat = [i[0]for i in y_train]
# long = [i[1]for i in y_train]
# regr.fit(X_train,lat)
# lat_pred = regr.predict(X_test)
# regr.fit(X_train,long)
# long_pred = regr.predict(X_test)
# y_pred = [(lat_pred[i],long_pred[i])for i in range(len(lat_pred))]
# lat_test = [i[0]for i in y_test]
# long_test = [i[1]for i in y_test]

# print("ExtraTreesRegressor Error: "+ str(kaggle_RMSE(y_test, y_pred)))

