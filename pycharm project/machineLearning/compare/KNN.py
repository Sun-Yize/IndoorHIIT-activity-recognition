from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def model(inputs, outputs):
    print('K Nearest Neighbors Start')
    inputs = StandardScaler().fit_transform(inputs)
    knn = KNeighborsClassifier()
    recall_score = cross_val_score(knn, inputs, outputs, cv=10)
    print('Finish')
    return recall_score.mean()
