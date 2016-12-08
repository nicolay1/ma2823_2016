import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import cross_validation

X = np.loadtxt('train.csv',  delimiter=',', skiprows=1, usecols=range(2, 13))

X_test = np.loadtxt('test.csv',  delimiter=',', skiprows=1, usecols=range(2, 13))

y = np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=[14])

scaler = preprocessing.StandardScaler()
X_norm = preprocessing.normalize(X)

clf = linear_model.LogisticRegression(C=1e6)
folds = cross_validation.StratifiedKFold(y, 10, shuffle=True)

def cross_validate_with_scaling(design_matrix, labels, classifier, cv_folds):
    """ Perform a cross-validation and returns the predictions. 
    Use a scaler to scale the features to mean 0, standard deviation 1.
    
    Parameters:
    -----------
    design_matrix: (n_samples, n_features) np.array
        Design matrix for the experiment.
    labels: (n_samples, ) np.array
        Vector of labels.
    classifier:  sklearn classifier object
        Classifier instance; must have the following methods:
        - fit(X, y) to train the classifier on the data X, y
        - predict_proba(X) to apply the trained classifier to the data X and return probability estimates 
    cv_folds: sklearn cross-validation object
        Cross-validation iterator.
        
    Return:
    -------
    pred: (n_samples, ) np.array
        Vectors of predictions (same order as labels).
    """
    pred = np.zeros(labels.shape)
    for i,(tr,te) in enumerate(cv_folds):
	print(i)
        Xtr = scaler.fit_transform(design_matrix[tr])
        Xte = scaler.fit_transform(design_matrix[te])
        classifier.fit(Xtr,labels[tr])
        predict_proba=classifier.predict_proba(Xte)
        j=0
        for predict in predict_proba:
            pred[te[j]]=predict_proba[j][1]
            j+=1
    return pred

#print(cross_validate_with_scaling(X_norm,y,clf,folds))

clf.fit(X,y)

file = open("out.txt", "w+")
file.write(str([i for i in clf.predict(X_test)]))
file.close()
