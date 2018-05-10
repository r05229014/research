import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# path
dir_ = '../data/'

all_data = os.listdir(dir_)
for file in all_data:
    if "ATM" in file:
        all_data.remove(file)

# loading data
data = np.empty([1,7])

for i in range(len(all_data)):
    d = np.load(dir_+all_data[i])
    print('loading...'+ dir_+all_data[i])
    data = np.concatenate([data, d], axis=0)
data = data[1::]
#data = data[data[:,5]>0]
print("\nThere are %s clouds in the original dataset. \n" %data.shape[0])
print("There are %s clouds' core number is <5, and we call it's small cloud" %data[data[:,5]<5].shape[0])
print("There are %s clouds' core number is >=5, and we call it's big cloud" %data[data[:,5]>=5].shape[0])
#print('All the cloud core is bigger than zero. (no zero core cloud)')

# shuffle data and split it to training data, validataion data and testing data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
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

# # validataion data
X_val = val_data[:,[2,3,4,6]]
X_val = sc.fit_transform(X_val)
y_val = val_data[:,5]
y_val = y_val >= 5

# testion data
X_test = test_data[:,[2,3,4,6]]
X_test = sc.fit_transform(X_test)
y_test = test_data[:,5]
y_test = y_test >= 5

np.save('../Processed_data/X_train_nor.npy', X_train)
np.save('../Processed_data/X_val_nor.npy', X_val)
np.save('../Processed_data/X_test_nor.npy', X_test)
np.save('../Processed_data/y_train.npy', y_train)
np.save('../Processed_data/y_val.npy', y_val)
np.save('../Processed_data/y_test.npy', y_test)



