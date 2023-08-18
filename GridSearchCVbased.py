from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.svm import SVC
import pandas as pd
from DataPreparation import preprocess_features
from sklearn.model_selection import train_test_split
from PCA import PrincipalComponentAnalysis
from sklearn.preprocessing import LabelEncoder

def gridsearch(X_train, X_test,  y_train, y_test):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    parameters = { 'C': [0.1, 1, 10, 100, 1000],
                   'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                   'kernel': ['rbf', 'sigmoid', 'linear']
             } 
    clf = SVC()
    f1_scorer = make_scorer(f1_score)    
    grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        refit=True,
                        verbose=3,
                        cv=5)
    
    grid_obj = grid_obj.fit(X_train,y_train)
    clf = grid_obj.best_estimator_
    print(clf)

    f1_train, acc_train = predict_labels(clf, X_train, y_train)
    print( "F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1_train , acc_train))   

    f1_test, acc_test = predict_labels(clf, X_test, y_test)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1_test , acc_test))

    with open('Performance/SVC_gridsearchcv_Performance.txt', 'w') as f:
        f.write('SVC GridSearchCV\n')
        f.write(f'num_features = {X_train.shape[1]}\n\n')
        f.write("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1_train , acc_train))
        f.write("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1_test , acc_test))
    

# Returns F1 score and Accuracy:
def predict_labels(clf, features, target):

    y_pred = clf.predict(features)
    
    return f1_score(target, y_pred), sum(target == y_pred) / float(len(y_pred))

if __name__ == '__main__':
    df = pd.read_csv('Data/processed_data.csv')

    X= df.drop(['FTR'], axis=1)
    Y= df['FTR']

    X = preprocess_features(X, df)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)      
    X_train, X_test = PrincipalComponentAnalysis(X_train, X_test, num=20)

    gridsearch(X_train, X_test, y_train, y_test)