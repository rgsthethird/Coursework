import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#SVC is for Support Vector Classifier -- we called it SVM in class
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss

def get_favorite_data():

    d = 4

    mu0 = np.array([-5 for i in range(d)])
    mu1 = np.array([ 5 for i in range(d)])

    y = np.random.binomial(1, 0.5) #flip a coin for y

    if y == 0:
        x = np.random.multivariate_normal(mean = mu0, cov = np.eye(d))
    else:
        x = np.random.multivariate_normal(mean = mu1, cov = np.eye(d))

    return x, y

def example_get_favorite_data():
    # Two, far apart, spherical Gaussian blobs
    d = 5

    mu0 = np.array([-5 for i in range(d)])
    mu1 = np.array([ 5 for i in range(d)])

    y = np.random.binomial(1, 0.5) #flip a coin for y

    if y == 0:
        x = np.random.multivariate_normal(mean = mu0, cov = np.eye(d))
    else:
        x = np.random.multivariate_normal(mean = mu1, cov = np.eye(d))

    return x, y

def get_lots_of_favorite_data(n = 100, data_fun = get_favorite_data):
    pts = [data_fun() for _ in range(n)]
    Xs, ys = zip(*pts)
    X = np.array(Xs)
    y = np.array(ys)
    return (X, y)

if __name__ == "__main__":
    print("Here are some points from get_favorite_data:")
    for i in range(4):
        x, y = get_favorite_data()
        print(f"\tx: {x}")
        print(f"\ty: {y}")

    print("And here we use get_lots_of_favorite_data to obtain X and y:")
    X, y = get_lots_of_favorite_data(100, get_favorite_data)

    rand_spots = np.random.choice(len(X), 75, replace = False)
    print(rand_spots)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    test_spots = []

    for i in range(len(X)):
        if i not in rand_spots:
            test_spots.append(i)

    for spot in rand_spots:
        x_train.append(X[spot])
        y_train.append(y[spot])

    for spot in test_spots:
        x_test.append(X[spot])
        y_test.append(y[spot])

    #kNN tests
    n = [1,3,5,7,9]
    for spot in range(len(n)):
        knc = KNeighborsClassifier(n_neighbors = n[spot])
        knc.fit(x_train, y_train)
        predicted_tr = (knc.predict(x_train))
        predicted_te = (knc.predict(x_test))

        te_loss = round(zero_one_loss(y_test,predicted_te),4)

        print("kNN "+str(n[spot])+" had loss of "+str(te_loss))

    #tree tests
    n = [1,2,3,4,0]
    for spot in range(len(n)):
        if n[spot] != 0:
            dtc = DecisionTreeClassifier(criterion = "entropy", max_depth=n[spot])
        else:
            dtc = DecisionTreeClassifier(criterion = "entropy")
        dtc.fit(x_train, y_train)
        predicted_tr = (dtc.predict(x_train))
        predicted_te = (dtc.predict(x_test))

        te_loss = round(zero_one_loss(y_test,predicted_te),4)

        print("Decision Tree "+str(n[spot])+" had loss of "+str(te_loss))

    #svm tests
    n = ["linear","rbf","poly"]
    for spot in range(len(n)):
        svc = SVC(kernel = n[spot])
        svc.fit(x_train, y_train)
        predicted_tr = (svc.predict(x_train))
        predicted_te = (svc.predict(x_test))

        te_loss = round(zero_one_loss(y_test,predicted_te),4)

        print("SVC "+str(n[spot])+" had loss of "+str(te_loss))
