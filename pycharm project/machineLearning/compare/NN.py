from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


def model(inputs, outputs):
    print('Neural Network Start')
    nn = MLPClassifier(max_iter=300)
    recall_score = cross_val_score(nn, inputs, outputs, cv=10)
    print('Finish')
    return recall_score.mean()
