from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from DataPreparation import home_nhome
from PCA import PrincipalComponentAnalysis
import pandas as pd
from DataPreparation import preprocess_features
from sklearn.model_selection import train_test_split

def svc_classifier(X_train, X_test, y_train, y_test):
    classifier = SVC(kernel = 'linear',random_state = 0)
    classifier.fit(X_train, y_train)
    Y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, Y_pred)
    plt.figure()
    plt.title('SVC Confusion Matrix')
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig('Plots/svc_cm.png')
    print(classification_report(y_test, Y_pred))
    with open(f'Performance/SVC_Performance.txt', 'w') as f:
        f.write('SVC\n')
        f.write(f'num_features = {X_train.shape[1]}\n\n')
        f.write(classification_report(y_test, Y_pred))


if __name__ == '__main__':
    df = pd.read_csv('Data/processed_data.csv')

    X= df.drop(['FTR'], axis=1)
    Y= df['FTR']

    X = preprocess_features(X, df)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)      
    X_train, X_test = PrincipalComponentAnalysis(X_train, X_test, num=20)

    svc_classifier(X_train, X_test, y_train, y_test)