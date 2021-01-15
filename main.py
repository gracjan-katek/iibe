"""
@author: Gracjan Katek
"""
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, plot_roc_curve, plot_confusion_matrix, \
    fbeta_score, \
    matthews_corrcoef, auc, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier

amino_acids_list = ['A',
                    'C',
                    'D',
                    'E',
                    'F',
                    'G',
                    'H',
                    'I',
                    'K',
                    'L',
                    'M',
                    'N',
                    'P',
                    'Q',
                    'R',
                    'S',
                    'T',
                    'V',
                    'W',
                    'Y']

MHC_dict = {
    'H2-Db': [0],
    'H2-Kb': [1],
    'H2-Dd': [2],
    'H2-Kd': [3],
    'H2-Kk': [4],
    'H2-Ld': [5],
}

MHC_sequences = {
    'H2-Db': 'HPSKGEETSNRYE',
    'H2-Dd': 'HRPEGDETSKHEE',
    'H2-Kb': 'HSPEDKEISYHYQ',
    'H2-Kd': 'YPSQVDDTANHHK',
    'H2-Kk': 'RSPEDKETSYHYQ',
    'H2-Ld': 'HPSKGEETSNRYE'
}

prot_dict = {
    'X': [0],
    'A': [1],
    'R': [2],
    'N': [3],
    'D': [4],
    'C': [5],
    'F': [6],
    'G': [7],
    'Q': [8],
    'E': [9],
    'H': [10],
    'I': [11],
    'L': [12],
    'K': [13],
    'M': [14],
    'P': [15],
    'S': [16],
    'T': [17],
    'W': [18],
    'Y': [19],
    'V': [20]
}

def get_peptide_vector(sequence):
    result = []
    for char in sequence:
        result += prot_dict[char]
    result += prot_dict['X'] * (11 - len(sequence))
    return result

def get_MHC_sequence_vector(MHC):
    result = []
    for char in MHC_sequences[MHC]:
        result += prot_dict[char]
    return result


def get_counts(sequence):
    result = []
    for amino_acid in amino_acids_list:
        result.append(sequence.count(amino_acid))
    return result


def get_vector(tuple_sequence_MHC):
    # vector = get_counts(tuple_sequence_MHC[0])
    vector = get_peptide_vector(tuple_sequence_MHC[0])
    vector += get_MHC_sequence_vector(tuple_sequence_MHC[1])
    vector += MHC_dict[tuple_sequence_MHC[1]]
    return vector


def set_dataset():
    list_data = [line.split() for line in
                 open('/Users/gracjankatek/Documents/IIBE/train.tsv').read().strip().split('\n')[1:]]
    dataset = []
    print(MHC_dict)
    for entry in list_data:
        vector = get_vector((entry[0], entry[1]))
        label = float(entry[2])
        dataset.append(vector + [label])
    dataset = np.array(dataset)
    print(dataset.shape[1])
    x = dataset[:, :-1]
    y = dataset[:, -1]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15)
    return train_x, test_x, train_y, test_y


def set_validation_dataset():
    list_data = [line.split() for line in
                 open('/Users/gracjankatek/Documents/IIBE/example_test.tsv').read().strip().split('\n')[1:]]
    dataset = []
    print(MHC_dict)
    for entry in list_data:
        vector = get_vector((entry[0], entry[1]))
        dataset.append(vector)
    dataset = np.array(dataset)
    return dataset


def get_result_old(model, train_x, test_x, train_y, test_y):
    classifier = model
    classifier.fit(train_x, train_y)
    y_predict = classifier.predict(test_x)
    target_names = ['positive', 'negative']
    print(classification_report(test_y, y_predict, target_names=target_names))
    print('F1 score: ' + str(f1_score(test_y, y_predict, average='weighted')))
    print('F2 score: ' + str(fbeta_score(test_y, y_predict, average='weighted', beta=0.5)))
    print('MCC score: ' + str(matthews_corrcoef(test_y, y_predict)))
    print('Accuracy score : ' + str(accuracy_score(test_y, y_predict)))
    print('Precision score: ' + str(precision_score(test_y, y_predict, average='weighted')))
    return classifier


def get_result(model, train_x, test_x, train_y, test_y, validation_x):
    classifier = model
    classifier.fit(train_x, train_y)
    y_predict = classifier.predict(test_x)
    scoring_list = ['precision_weighted', 'f1_weighted', 'accuracy', 'f1_micro', 'f1_macro']
    print('--------10---------')
    for scoring in scoring_list:
        print(scoring)
        scores = cross_val_score(classifier, test_x, test_y, cv=10, scoring=scoring)
        print(scores)
        print(scores.mean())
    print('--------5----------')
    for scoring in scoring_list:
        print(scoring)
        scores = cross_val_score(classifier, test_x, test_y, cv=10, scoring=scoring)
        print(scores)
        print(scores.mean())
    data = sqrt(
        (precision_score(test_y, y_predict, average='weighted') + recall_score(test_y, y_predict, average='weighted')))
    print('--------Final---------')
    print('F1 score: ' + str(f1_score(test_y, y_predict, average='weighted')))
    print('F2 score: ' + str(fbeta_score(test_y, y_predict, average='weighted', beta=0.5)))
    print('MCC score: ' + str(matthews_corrcoef(test_y, y_predict)))
    print('Accuracy score : ' + str(accuracy_score(test_y, y_predict)))
    print('Precision score: ' + str(precision_score(test_y, y_predict, average='weighted')))
    print('G-Mean: ' + str(data))
    return classifier


def prediction_plot(cl1, cl2, cl3, x, y):
    roc1 = plot_roc_curve(cl1, x, y)
    roc2 = plot_roc_curve(cl2, x, y)
    roc3 = plot_roc_curve(cl3, x, y)
    plt.figure()
    plt.plot(roc1.fpr, roc1.tpr, label='XGBoost (AUC = %0.2f' % auc(roc1.fpr, roc1.tpr))
    plt.plot(roc2.fpr, roc2.tpr, label='SVM (AUC = %0.2f)' % auc(roc2.fpr, roc2.tpr))
    plt.plot(roc3.fpr, roc3.tpr, label='Decision Tree (AUC = %0.2f)' % auc(roc3.fpr, roc3.tpr))
    plt.legend(loc='best')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()


def get_parameters(clf, x, y):
    y_predict = clf.predict(x)
    data = plot_roc_curve(clf, y, y_predict)
    print('F2 score: ' + str(fbeta_score(y, y_predict, beta=0.5)))
    print('MCC score: ' + str(matthews_corrcoef(y, y_predict)))
    print('G-Mean: ' + str(sqrt(data.fpr + data.tpr)))


def matrix_plot(cl1, cl2, cl3, x, y):
    target_names = ['positive', 'negative']
    display1 = plot_confusion_matrix(cl1, x, y, display_labels=target_names)
    display2 = plot_confusion_matrix(cl2, x, y, display_labels=target_names)
    display3 = plot_confusion_matrix(cl3, x, y, display_labels=target_names)
    display1.ax_.set_title('Confusion matrix XGBoost')
    display2.ax_.set_title('Confusion matrix SVM')
    display3.ax_.set_title('Confusion matrix DecisionTree')
    plt.show()


def tree_plot(tree):
    plt.figure()
    plot_tree(tree, filled=True)
    plt.show()


train_x, test_x, train_y, test_y = set_dataset()
print("-----------XGBoost--------------")
xgb = get_result(XGBClassifier(max_depth=20, n_estimators=100, learning_rate=0.1, n_jobs=4), train_x, test_x, train_y,
                 test_y, set_validation_dataset())
print("-----------SVM--------------")
svc = get_result(SVC(), train_x, test_x, train_y, test_y, set_validation_dataset())
print("-----------Decision Tree--------------")
tree_2 = get_result(DecisionTreeClassifier(), train_x, test_x, train_y, test_y, set_validation_dataset())
prediction_plot(xgb, svc, tree_2, test_x, test_y)
matrix_plot(xgb, svc, tree_2, test_x, test_y)
# tree_plot(tree_2)

