from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from DataPreparation import preprocess_features
from PCA import PrincipalComponentAnalysis


def random_forest(X_train, X_test, y_train, y_test):
    classifier = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='sqrt',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
    
    classifier.fit(X_train, y_train)
    Y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, Y_pred)
    plt.figure()
    plt.title('Random Forest Confusion Matrix')
    sns.heatmap(cm, annot=True,fmt='d')
    plt.savefig('Plots/random_forest_cm.png')
    print(classification_report(y_test, Y_pred))
    with open('Performance/randomF_Performance.txt', 'w') as f:
        f.write('Random Forest\n')
        f.write(f'num_features = {X_train.shape[1]}\n\n')
        f.write(classification_report(y_test, Y_pred))


if __name__ == '__main__':
    df = pd.read_csv('Data/processed_data.csv')

    X= df.drop(['FTR'], axis=1)
    Y= df['FTR']

    X = preprocess_features(X, df)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)      
    X_train, X_test = PrincipalComponentAnalysis(X_train, X_test, num=20)

    random_forest(X_train, X_test, y_train, y_test)