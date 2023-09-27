import requests
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

frames1 = [df2, qp2]
result1 = pd.concat(frames1)

classlist_df=df.pop('labels')
classlist_qp=qp.pop('labels')
frames2 = [classlist_df, classlist_qp]
result2 = pd.concat(frames2)

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

train = pd.get_dummies(result1, columns = ['protocol_type','service','flag'])

X = train.drop(['labels'],axis=1)
Y = train.labels
Y_count=Y.value_counts()

l_encode = LabelEncoder()
l_encode.fit(Y)
Y = l_encode.transform(Y)
Y = to_categorical(Y)

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



import socket
for i in range(0,20):
    HOST = '127.0.0.1'
    PORT = 8888
    list1=[]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            #print(f'Connected by {addr}')
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                received_data = data.decode()
            
                protocol_type, service, flag = received_data.split(',')
                list1.append(protocol_type)
                list1.append(service)
                list1.append(flag)

                scores = model.evaluate(X_test, y_test)

                for i, m in enumerate(model.metrics_names):
                    print("\n%s: %.3f"% (m, scores[i]))
                oos_pred.append(scores)
                print("Validation score: {}".format(scores))
                
                matching_column_names1=['icmp','tcp','udp']
                matching_column_names2=['IRC','X11','Z39_50','aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain',
                'domain_u','echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames','http','http_2784','http_443',
                'http_8001','imap4','iso_tsap','klogin','kshell','ldap','link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat',
                'nnsp','nntp','ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell','smtp','sql_net','ssh','sunrpc',
                'supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i','uucp','uucp_path','vmnet','whois']
                matching_column_names3=['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']

                coded_list1=[]
                for i in range(len(matching_column_names1)):
                    if(matching_column_names1[i]==list1[0]):
                        coded_list1.append(1)
                    else:
                        coded_list1.append(0)
                for i in range(len(matching_column_names2)):
                    if(matching_column_names2[i]==list1[1]):
                        coded_list1.append(1)
                    else:
                        coded_list1.append(0)
                for i in range(len(matching_column_names3)):
                    if(matching_column_names3[i]==list1[2]):
                        coded_list1.append(1)
                    else:
                        coded_list1.append(0)

                columns = ['protocol_type_icmp', 'protocol_type_tcp', 'protocol_type_udp', 'service_IRC', 'service_X11', 'service_Z39_50', 'service_aol', 
                'service_auth', 'service_bgp', 'service_courier', 'service_csnet_ns', 'service_ctf', 'service_daytime', 'service_discard', 
                'service_domain', 'service_domain_u', 'service_echo', 'service_eco_i', 'service_ecr_i', 'service_efs', 'service_exec', 
                'service_finger', 'service_ftp', 'service_ftp_data', 'service_gopher', 'service_harvest', 'service_hostnames', 'service_http', 
                'service_http_2784', 'service_http_443', 'service_http_8001', 'service_imap4', 'service_iso_tsap', 'service_klogin', 'service_kshell', 
                'service_ldap', 'service_link', 'service_login', 'service_mtp', 'service_name', 'service_netbios_dgm', 'service_netbios_ns', 
                'service_netbios_ssn', 'service_netstat', 'service_nnsp', 'service_nntp', 'service_ntp_u', 'service_other', 'service_pm_dump', 
                'service_pop_2', 'service_pop_3', 'service_printer', 'service_private', 'service_red_i', 'service_remote_job', 'service_rje', 
                'service_shell', 'service_smtp', 'service_sql_net', 'service_ssh', 'service_sunrpc', 'service_supdup', 'service_systat', 'service_telnet',
                'service_tftp_u', 'service_tim_i', 'service_time', 'service_urh_i', 'service_urp_i', 'service_uucp', 'service_uucp_path', 'service_vmnet', 
                'service_whois', 'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR', 'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 
                'flag_SH']

                X_test_new = pd.DataFrame()

                for column in columns:
                    X_test_new[column] = []

                X_test_new.loc[0]=coded_list1

                B=[[0,1,0,0,0]]
                y_test_new=np.array(B)
            
                scores = model.evaluate(X_test_new, y_test_new)


                y_pred=model.predict(X_test_new)
                y_test_class=np.argmax(y_test_new,axis=1)
                y_pred_class=np.argmax(y_pred,axis=1)

                from sklearn.metrics import confusion_matrix
                confussion_matrix=confusion_matrix(y_test_class, y_pred_class, labels=[0,1,2,3,4])
                #print(confusion_matrix)
                def res():
                    for i in range(len(confussion_matrix)):
                        for j in range(len(confussion_matrix[i])):
                            if(confussion_matrix[i][j]==1):
                                return j

                attacks_classification=['DOS','Normal','Probe','R2L','U2R']
                value=res()
                if(value==1):
                    response = f'Access Granted '.encode()
                    print("Access Granted; Connection status:",attacks_classification[value])
                    conn.sendall(response)
                else:
                    response = f'Access Denied '.encode()
                    print("Access Denied; Intrusion type: ",attacks_classification[value])
                    conn.sendall(response)

            






