## ANN neural network to detect the intrusion and classify differenet intrusion 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from joblib import dump, load
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from keras import backend as K

import pandas as pd
import numpy as np
import sys
import keras
import sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, GlobalAveragePooling1D
from keras.utils import to_categorical
import sklearn.preprocessing
from sklearn import metrics
from scipy.stats import zscore
from tensorflow.keras.utils import get_file, plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('/Users/adarshjatti/Desktop/Datasets/kdd_train.csv',low_memory=False)
qp = pd.read_csv('/Users/adarshjatti/Desktop/Datasets/kdd_test.csv',low_memory=False)

df2 = pd.DataFrame().assign(protocol_type=df['protocol_type'], service=df['service'],flag=df['flag'])
qp2 = pd.DataFrame().assign(protocol_type=qp['protocol_type'], service=qp['service'],flag=qp['flag'])
#print(result1)

frames1 = [df2, qp2]
result1 = pd.concat(frames1)
#print("merged 1")
#print(result1)

classlist_df=df.pop('labels')
classlist_qp=qp.pop('labels')
#print(classlist_df)
frames2 = [classlist_df, classlist_qp]
result2 = pd.concat(frames2)
#print("merged 2")
#print(result2)

#Fixing labels for training set
classlist_df_new = []
check1 = ("apache2","back","land","neptune","mailbomb","pod","processtable","smurf","teardrop","udpstorm","worm")
check2 = ("ipsweep","mscan","nmap","portsweep","saint","satan")
check3 = ("buffer_overflow","loadmodule","perl","ps","rootkit","sqlattack","xterm")
check4 = ("ftp_write","guess_passwd","httptunnel","imap","multihop","named","phf","sendmail","Snmpgetattack","spy","snmpguess","warezclient","warezmaster","xlock","xsnoop")

DoSCount=0
ProbeCount=0
U2RCount=0
R2LCount=0
NormalCount=0

for item in result2:
    if item in check1:
        classlist_df_new.append("DOS")
        DoSCount=DoSCount+1
    elif item in check2:
        classlist_df_new.append("Probe")
        ProbeCount=ProbeCount+1
    elif item in check3:
        classlist_df_new.append("U2R")
        U2RCount=U2RCount+1
    elif item in check4:
        classlist_df_new.append("R2L")
        R2LCount=R2LCount+1
    else:
        classlist_df_new.append("Normal")
        NormalCount=NormalCount+1 





result1['labels']=classlist_df_new


print()
print("dataset with independent and dependent values: ")
print(result1)
print()

# one hot encoding of indepenedent inputs 
train = pd.get_dummies(result1, columns = ['protocol_type','service','flag'])






X = train.drop(['labels'],axis=1)
Y = train.labels
Y_count=Y.value_counts()

print("count of each intrusion:")
print(Y_count)
print()

print("dataset after one-hot encoding the independent values: ")
print(train)
print()



print("dataframe with number one-hotcoding the independent values: ")
print(X)
print()

print("dataframe of dependent values:")
print(Y)
print()

# one hot encoding of depenedent inputs 
l_encode = LabelEncoder()
l_encode.fit(Y)
Y = l_encode.transform(Y)
Y = to_categorical(Y)
print("one hot-encding the dependent values:")
print(Y)
print(type(Y))


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

oos_pred=[]

model = Sequential()
model.add(Dense(64, input_dim = 84, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(5, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 15, batch_size = 64)
scores = model.evaluate(X_test, y_test)

for i, m in enumerate(model.metrics_names):
    print("\n%s: %.3f"% (m, scores[i]))

oos_pred.append(scores)
print("Validation score: {}".format(scores))
print()

y_pred=model.predict(X_test)
y_test_class=np.argmax(y_test,axis=1)
y_pred_class=np.argmax(y_pred,axis=1)

from sklearn.metrics import confusion_matrix
confussion_matrix=confusion_matrix(y_test_class, y_pred_class, labels=[0,1,2,3,4])
print(confussion_matrix)

import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

print(plot_confusion_matrix(cm           = confussion_matrix, 
                      normalize    = False,
                      target_names = ["DOS","Normal","Probe","R2L","U2R"],
                      title        = "Confusion Matrix"))

