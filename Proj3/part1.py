import numpy as np
import dill as pickle
import sklearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import helpers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#SVC is for Support Vector Classifier -- we called it SVM in class
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss

# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=DeprecationWarning)

if __name__ == "__main__":
    assert 1/2 == 0.5, "Are you sure you're using python 3?"
    print(f"Version of sklearn: {sklearn.__version__}")
    print("(It should be 0.20.0)")

def kNN(pickle_file):

    fin = open(pickle_file, "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test

    n = [1,3,5,7,9]

    figure = plt.figure()
    for spot in range(len(n)):
        knc = KNeighborsClassifier(n_neighbors = n[spot])
        knc.fit(X_tr, y_tr)
        predicted_tr = (knc.predict(X_tr))
        predicted_te = (knc.predict(X_te))

        axis = figure.add_subplot(1,5,spot+1)

        xtr_1 = []
        xtr_2 = []

        for pair in X_tr:
            xtr_1.append(pair[0])
            xtr_2.append(pair[1])

        xte_1 = []
        xte_2 = []

        for pair in X_te:
            xte_1.append(pair[0])
            xte_2.append(pair[1])

        colors = ListedColormap(['#FF0000', '#0000FF'])

        axis.scatter(xtr_1,xtr_2, c = y_tr, cmap = colors, edgecolors = 'k')
        axis.scatter(xte_1,xte_2, marker="*", c = y_te, cmap = colors, edgecolors = 'k')
        x1min, x1max, x2min, x2max = helpers.get_bounds(X_tr)
        helpers.plot_decision_boundary(axis, knc, x1min, x1max, x2min, x2max)
        axis.set_title("n_neighbors = " + str(n[spot]))

        tr_loss = round(zero_one_loss(y_tr,predicted_tr),2)
        te_loss = round(zero_one_loss(y_te,predicted_te),2)

        axis.set_xlabel("Tr loss: " + str(tr_loss)+"\n Te loss: " + str(te_loss))

    plt.show()

def decision_tree(pickle_file):

    fin = open(pickle_file, "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test

    n = [1,2,3,4,0]

    figure = plt.figure()
    for spot in range(len(n)):
        if n[spot] != 0:
            dtc = DecisionTreeClassifier(criterion = "entropy", max_depth=n[spot])
        else:
            dtc = DecisionTreeClassifier(criterion = "entropy")
        dtc.fit(X_tr, y_tr)
        predicted_tr = (dtc.predict(X_tr))
        predicted_te = (dtc.predict(X_te))

        axis = figure.add_subplot(1,5,spot+1)

        xtr_1 = []
        xtr_2 = []

        for pair in X_tr:
            xtr_1.append(pair[0])
            xtr_2.append(pair[1])

        xte_1 = []
        xte_2 = []

        for pair in X_te:
            xte_1.append(pair[0])
            xte_2.append(pair[1])

        colors = ListedColormap(['#FF0000', '#0000FF'])

        axis.scatter(xtr_1,xtr_2, c = y_tr, cmap = colors, edgecolors = 'k')
        axis.scatter(xte_1,xte_2, marker="*", c = y_te, cmap = colors, edgecolors = 'k')
        x1min, x1max, x2min, x2max = helpers.get_bounds(X_tr)
        helpers.plot_decision_boundary(axis, dtc, x1min, x1max, x2min, x2max)

        if n[spot] != 0:
            axis.set_title("max_depth = " + str(n[spot]))
        else:
            axis.set_title("max_depth = none")

        tr_loss = round(zero_one_loss(y_tr,predicted_tr),2)
        te_loss = round(zero_one_loss(y_te,predicted_te),2)

        axis.set_xlabel("Tr loss: " + str(tr_loss)+"\n Te loss: " + str(te_loss))

    plt.show()

def svm(pickle_file):

    fin = open(pickle_file, "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test

    n = ["linear","rbf","poly"]

    figure = plt.figure()
    for spot in range(len(n)):
        svc = SVC(kernel = n[spot])
        svc.fit(X_tr, y_tr)
        predicted_tr = (svc.predict(X_tr))
        predicted_te = (svc.predict(X_te))

        axis = figure.add_subplot(1,5,spot+1)

        xtr_1 = []
        xtr_2 = []

        for pair in X_tr:
            xtr_1.append(pair[0])
            xtr_2.append(pair[1])

        xte_1 = []
        xte_2 = []

        for pair in X_te:
            xte_1.append(pair[0])
            xte_2.append(pair[1])

        colors = ListedColormap(['#FF0000', '#0000FF'])

        axis.scatter(xtr_1,xtr_2, c = y_tr, cmap = colors, edgecolors = 'k')
        axis.scatter(xte_1,xte_2, marker="*", c = y_te, cmap = colors, edgecolors = 'k')
        x1min, x1max, x2min, x2max = helpers.get_bounds(X_tr)
        helpers.plot_decision_boundary(axis, svc, x1min, x1max, x2min, x2max)

        if n[spot] != 0:
            axis.set_title("kernel = " + str(n[spot]))
        else:
            axis.set_title("kernel = none")

        tr_loss = round(zero_one_loss(y_tr,predicted_tr),2)
        te_loss = round(zero_one_loss(y_te,predicted_te),2)

        axis.set_xlabel("Tr loss: " + str(tr_loss)+"\n Te loss: " + str(te_loss))

    plt.show()


simple = "simple_task.pkl"
moons = "moons.pkl"

kNN(simple)
kNN(moons)
decision_tree(simple)
decision_tree(moons)
svm(simple)
svm(moons)
