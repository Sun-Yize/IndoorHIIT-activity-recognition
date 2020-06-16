from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def model(inputs, outputs):
    print('Random Forest Start')
    inputs = StandardScaler().fit_transform(inputs)
    rfc = RandomForestClassifier(n_estimators=100)
    recall_score = cross_val_score(rfc, inputs, outputs, cv=10)
    print('Finish')
    return recall_score.mean()
