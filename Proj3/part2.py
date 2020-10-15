import numpy as np
import dill as pickle
import sklearn
import matplotlib.pyplot as plt
import helpers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#SVC is for Support Vector Classifier -- we called it SVM in class
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=DeprecationWarning)

if __name__ == "__main__":
    assert 1/2 == 0.5, "Are you sure you're using python 3?"
    print(f"Version of sklearn: {sklearn.__version__}")
    print("(It should be 0.20.0)")

def looking(pickle_file):

    fin = open(pickle_file, "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test

    for i in range(10):
        count = 0
        for num in y_tr:
            if i == num:
                count = count+1
        print("There are "+str(count)+" "+str(i)+"'s.")

    for run in range(10):
        figure = plt.figure()
        to_average = []
        for spot in range(len(y_tr)):
            if y_tr[spot] == run:
                to_average.append(X_tr[spot])
        to_display = np.average(to_average, axis=0)
        axis = figure.add_subplot(1,5,1)
        helpers.plot_num(axis,to_display)
        plt.show()

def training(pickle_file):

    fin = open(pickle_file, "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test

    lr = LogisticRegression(multi_class='ovr').fit(X_tr,y_tr)
    predicted_tr = lr.predict(X_tr)
    predicted_te = lr.predict(X_te)

    print(confusion_matrix(y_te, predicted_te))
    print(classification_report(y_te, predicted_te))

def preprocessing(pickle_file):

    fin = open(pickle_file, "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test

    figure = plt.figure()
    knc = KNeighborsClassifier(n_neighbors = 3)
    knc.fit(X_tr, y_tr)
    predicted_te = (knc.predict(X_te))
    print(confusion_matrix(y_te, predicted_te))

    scaler = MinMaxScaler()
    scaler.fit(X_tr)
    transformed_xtr = scaler.transform(X_tr)
    transformed_xte = scaler.transform(X_te)

    figure = plt.figure()
    knc = KNeighborsClassifier(n_neighbors = 3)
    knc.fit(transformed_xtr, y_tr)
    predicted_te = (knc.predict(transformed_xte))
    print(confusion_matrix(y_te, predicted_te))

def tuning(pickle_file):

    fin = open(pickle_file, "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test

    scaler = MinMaxScaler()
    scaler.fit(X_tr)
    transformed_xtr = scaler.transform(X_tr)
    transformed_xte = scaler.transform(X_te)

    svc = SVC()
    parameters = [
    {'kernel':['rbf'],'gamma':[1.0,0.1,0.01,0.001],'C':[1,10,100,1000]},
    {'kernel':['poly'],'degree':[2,3,4,5],'C':[1,10,100,1000]},
    {'kernel':['sigmoid'],'degree':[0.1,1,10,100],'C':[1,10,100,1000]}
    ]
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(transformed_xtr, y_tr)
    best_model = clf.best_params_

    if best_model['kernel'] == 'rbf':
        new_svc = SVC(kernel = best_model['kernel'], C = best_model['C'], gamma = best_model['gamma'])
    else:
        new_svc = SVC(kernel = best_model['kernel'], C = best_model['C'], degree = best_model['degree'])
    new_svc.fit(transformed_xtr, y_tr)
    predicted_tr = new_svc.predict(transformed_xtr)
    predicted_te = new_svc.predict(transformed_xte)

    print(best_model)
    print(confusion_matrix(y_te, predicted_te))
    print(classification_report(y_te, predicted_te))

digits = "digits.pkl"
cancer = "cancer.pkl"
looking(digits)
training(digits)
preprocessing(cancer)
tuning(cancer)
