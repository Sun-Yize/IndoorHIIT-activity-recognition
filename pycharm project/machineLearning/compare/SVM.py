from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def model(inputs, outputs):
    print('Support Vector Machine Start')
    inputs = StandardScaler().fit_transform(inputs)
    svc = SVC(kernel='rbf', gamma='scale')
    recall_score = cross_val_score(svc, inputs, outputs, cv=10)
    print('Finish')
    return recall_score.mean()
