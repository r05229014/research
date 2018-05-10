import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt



def plot_confusion_matrix(cm, classes, normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalizaton can be applied by setting 'normalize=True'.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion_matrix")
    else:
        print("Confusion_matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_my_model(model_path, train, val, test, 
        y_train, y_val, y_test,test_name):
    print("\nplot!!!!!!!!!!!!!!") 
    model = load_model(model_path)
    y_pred_1 = model.predict(train)
    y_pred_1 = y_pred_1.round()
    y_pred_2 = model.predict(val)
    y_pred_2 = y_pred_2.round()
    y_pred_3 = model.predict(test)
    y_pred_3 = y_pred_3.round()

    cnf_matrix_1 = confusion_matrix(y_train, y_pred_1)
    cnf_matrix_2 = confusion_matrix(y_val, y_pred_2)
    cnf_matrix_3 = confusion_matrix(y_test, y_pred_3)
    #np.set_printoptions(precision=2)
    class_names = ['small', 'big']
 
    path = '../img/conf_matrix/' + test_name + '/'
    if not os.path.isdir(path):
        os.mkdir(path)

    plt.figure(1)
    plot_confusion_matrix(cnf_matrix_1, classes=class_names,
                                    title='Confusion matrix, without normalization on '+ r'$\bf{training}$' +' '+ r'$\bf{data}$')
    plt.savefig(path + 'train_data_conf')

    plt.figure(2)
    plot_confusion_matrix(cnf_matrix_2, classes=class_names,
                                   title='Confusion matrix, without normalization on '+ r'$\bf{validataion}$' +' '+ r'$\bf{data}$')
    plt.savefig(path + 'val_data_conf')
    

    plt.figure(3)
    plot_confusion_matrix(cnf_matrix_3, classes=class_names,
                                  title='Confusion matrix, without normalization on '+ r'$\bf{testing}$' +' '+ r'$\bf{data}$')
    plt.savefig(path + 'test_data_conf')

    plt.show()



def load_data(type='nor'):
    path = '/home/ericakcc/Desktop/ericakcc/research/Processed_data/'
    if type == 'ori':
        print('\n\nLoading original data...')
    elif type == 'nor':
        print('\n\nLoading normalized data...')
    else:
        print("Please type 'ori' or 'nor'")

    X_train = np.load(path + 'X_train_' + type + '.npy')
    X_val = np.load(path + 'X_val_' + type + '.npy')
    X_test = np.load(path + 'X_test_' + type + '.npy')

    y_train = np.load(path + 'y_train.npy')
    y_val = np.load(path + 'y_val.npy')
    y_test = np.load(path + 'y_test.npy')

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    test_name = 'test4'
    model_path = '../result/seminar/models/model_'+test_name +'.h5'
    X_train, X_val, X_test, y_train, y_val, y_test = load_data('nor')
    plot_my_model(model_path, X_train, X_val, X_test, y_train, y_val, y_test, test_name)
if __name__ == "__main__":
    main()
