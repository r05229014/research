import numpy as np
import keras 
import itertools
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random


def build_nn_4input():
    print("Build model!!")
    model = Sequential()
    model.add(Dense(DENSE, activation='relu', input_dim=4))
    model.add(Dense(DENSE//2, activation='relu'))
    model.add(Dense(DENSE//4, activation='relu'))
    model.add(Dense(DENSE//8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

def build_nn_3input():
    print("Build model!!")
    model = Sequential()
    model.add(Dense(DENSE, activation='relu', input_dim=3))
    model.add(Dense(DENSE//2, activation='relu'))
    model.add(Dense(DENSE//4, activation='relu'))
    model.add(Dense(DENSE//8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    return model

def train(model, train_x, test_x, train_y, test_y, model_path, csv_path, monitor):
    print("Training...")

    checkpoint = ModelCheckpoint(filepath=model_path, monitor=monitor, 
                                 verbose=1, 
                                 save_best_only=True)
    earlystop = EarlyStopping(monitor=monitor, patience = 20)
    history = model.fit(train_x, train_y,
                        validation_data = [test_x, test_y],
                        batch_size = BATCH_SIZE, epochs = EPOCHS,
                        callbacks = [checkpoint, earlystop],
                        shuffle = True)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    text = open(csv_path,'w')
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow('loss, val_loss, acc, val_acc')
    for i in range(len(loss)):
        s.writerow([loss[i], val_loss[i], acc[i], val_acc[i]])
    text.close()


def data_preprocess(flag):
    dir_ = '../data/'
    all_data = os.listdir(dir_)
    for file in all_data:
        if "ATM" in file:
            all_data.remove(file)

    data = np.empty([1,7])

    for i in range(len(all_data)):
        d = np.load(dir_ + all_data[i])
        print('loading...'+dir_+all_data[i])
        data = np.concatenate([data, d],axis=0)
    data = data[1::]

    # shuffle data and split it to training data, validataion data and testing data
    indices = np.arange(data.shape[0])
    SEED = 666
    random.seed(SEED)
    random.shuffle(indices)
    print(indices)
    data = data[indices]
    nb_testing_samples = int(0.1 * data.shape[0])
    
    test_data = data[0:nb_testing_samples]
    val_data = data[nb_testing_samples:2*nb_testing_samples]
    train_data = data[2*nb_testing_samples:]
    
    print('testing data samples: %s' %test_data.shape[0])
    print('validataion data samples: %s' %val_data.shape[0])
    print('training data samples: %s' %train_data.shape[0])

    sc = StandardScaler()
    # training data
    X_train = train_data[:,[2,3,4,6]]
    X_train = sc.fit_transform(X_train)
    y_train = train_data[:,5]
    y_train = y_train >= 5
    
    # validataion data
    X_val = val_data[:,[2,3,4,6]]
    X_val = sc.fit_transform(X_val)
    y_val = val_data[:,5]
    y_val = y_val >= 5
    
    # testion data
    X_test = test_data[:,[2,3,4,6]]
    X_test = sc.fit_transform(X_test)
    y_test = test_data[:,5]
    y_test = y_test >= 5
    
    if flag == 'm1':
        train_data_m1 = train_data[train_data[:,4]<2500]
        X_train_m1 = train_data_m1[:,[2,3,4,6]]
        X_train_m1 = sc.fit_transform(X_train_m1)
        y_train_m1 = train_data_m1[:,5]
        y_train_m1 = y_train_m1 >= 5
        X_train = X_train_m1
        y_train = y_train_m1

    elif flag == 'm2':
        train_data_m2 = train_data[train_data[:,5]>0]
        X_train_m2 = train_data_m2[:,[2,3,4,6]]
        X_train_m2 = sc.fit_transform(X_train_m2)
        y_train_m2 = train_data_m2[:,5]
        y_train_m2 = y_train_m2 >= 5
        X_train = X_train_m2
        y_train = y_train_m2
    
    elif flag == 'm3':
        X_train = np.load('../x_m3_generate.npy')
        X_train = sc.fit_transform(X_train)
        y_train = np.load('../y_m3_generate.npy')

    else:
        print("\n\nPlot original data!!!!!!!!!\n\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


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
    p = '../img3/conf_matrix/' 
    path = '../img3/conf_matrix/' + test_name + '/'
    if not os.path.isdir(p):
        os.mkdir(p)
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



def main(model_name, csv_name, monitor):
    csv_dir = "../result/seminar_3input/csv_history/"
    if not os.path.isdir(csv_dir):
        os.mkdir(csv_dir)
    model_dir = '../result/seminar_3input/models/'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    csv_path = csv_dir + csv_name
    model_path = model_dir + model_name
    model = build_nn_3input()
    train(model, X_train, X_val, y_train, y_val, model_path, csv_path, monitor)
    print('\n\n Done!!!!')
    plot_my_model(model_path, X_train, X_val, X_test, y_train, y_val, y_test, name)

if __name__ == '__main__' :
    BATCH_SIZE = 128
    EPOCHS = 2000
    DENSE = 256
    MONITOR = 'acc'
    name = 'test2_023'
    csv_name = '/'+name+'.csv'
    model_name = '/model_'+name +'.h5'
    model_path = '../result/seminar_3input/models/' + model_name 
    X_train, X_val, X_test, y_train, y_val, y_test = data_preprocess('haha')
    X_train = X_train[:,[0,2,3]]
    X_val = X_val[:,[0,2,3]]
    X_test = X_test[:,[0,2,3]]
    print(X_train.shape) 
    main(model_name, csv_name, MONITOR)


