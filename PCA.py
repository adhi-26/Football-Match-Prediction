from sklearn.preprocessing import StandardScaler as SS
from sklearn.decomposition import PCA
import numpy as np

def PrincipalComponentAnalysis(X_train, X_test, num):
    SC = SS()

    X_train = SC.fit_transform(X_train)
    X_test = SC.transform(X_test)

    pca = PCA(n_components=num)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    print(f'Explained Variance: {np.sum(pca.explained_variance_ratio_)}')
    return X_train, X_test
