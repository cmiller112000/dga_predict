"""
This is a complet file to 
import domain names, and extract domains
transform into bigram vectors
build RNN
validae models
predict new dataset by model
"""


import os
import tldextract
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from matplotlib import pyplot as plt

import numpy as np
# fix random seed for reproducibility
np.random.seed(8)

#import data
import pandas as pd
my_list = pd.read_csv("/home/cher/participantTrainData_wfeatures.csv", sep = ',', header =0)

#Load testing data
TestIn = pd.read_csv("/home/cher/participantTestData_wfeatures.csv", sep = ',', header =0)
TestAnsw = pd.read_csv("/home/cher/answerkeyTestData.csv",sep=',',header=0)
Test = TestIn.join(TestAnsw,on='id',rsuffix='answ')
print("training data: ", my_list.describe(include = 'all'))

X0 = my_list.ix[:,'domain_no_tld']
labels0 = my_list.ix[:,'type']

X_domain = [str(x) for x in X0.values.T.tolist()]
labels0 = labels0.values.T.tolist()

print("domain: ", X_domain[0:5])
print("labels: ", labels0[0:5])

X_test = Test.ix[:,'domain_no_tld']
#labels_test = Test.iloc[:,1:2]

X_test = [str(x) for x in X_test.values.T.tolist()]
#labels_test = labels_test.values.T.tolist()[0]

print("domain: ", X_test[0:5])
#print("labels: ", labels_test[0:5])

# Create feature vectors
# Generate a dictionary of valid characters
a00= [ x for x in X_domain if type(x) is str ]
a0=str.join(',',a00)
a1=set(a0)
a2=enumerate(a1)
valid_chars = {x:idx+1 for idx, x in a2 }

a00= [ x for x in X_test if type(x) is str ]
a0=str.join(',',a00)
a1=set(a0)
a2=enumerate(a1)
valid_chars2 = {x:idx+1 for idx, x in a2 }

max_features = len(valid_chars) + 1
maxlen = np.max([len(x) for x in X_domain])

# Convert characters to int and pad
count_vec = [[valid_chars[y] for y in x] for x in X_domain]
count_vec = sequence.pad_sequences(count_vec, maxlen=maxlen)

# Convert labels to 0-1
y = [0 if x == 'non-generated' else 1 for x in labels0]
y_test = [0 if x == 'non-generated' else 1 for x in Test.type]

max_features2 = len(valid_chars2) + 1
maxlen2 = np.max([len(x) for x in X_test])

test_count_vec = [[valid_chars2[y] for y in x] for x in X_test]
test_count_vec = sequence.pad_sequences(test_count_vec, maxlen=maxlen2)

def build_model(max_features):
    """Builds logistic regression model"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
#    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
#    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')
    return model

X_train, X_validate, y_train, y_validate = train_test_split(count_vec, y, test_size=0.2)

best_iter = -1
best_auc = 0.0
out_data = {}
batch_size = 128
max_epoch = 8

if not os.path.isfile("lstm2.model"):
    model = build_model(max_features)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=max_epoch,validation_split=0.2,verbose=2)
    try:
        model.save("lstm2.model")
    except:
        pass
else:
    model = load_model("lstm2.model")

probs_validate = model.predict_proba(X_validate,verbose=2)
print("AUC of Validation:", sklearn.metrics.roc_auc_score(y_validate, probs_validate))

out_data_validate = {'y_pred': probs_validate, "y_validate":y_validate,
            "id":Test.id}

fpr, tpr, _ = roc_curve(y_validate,probs_validate)
with plt.style.context('bmh'):
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.plot(fpr, tpr,
             lw=2, label='y_validate AUC: %s' % auc(fpr,tpr))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.tick_params(axis='both', labelsize=22)
    plt.savefig('lstm_vld_results.png')

f = open("out_data2.txt","w")
f.write("y_pred, y_validate")
f.write("\n")
for i in range(len(probs_validate)):
    f.write("{},{}".format(out_data_validate['y_pred'][i], out_data_validate['y_validate'][i]))
    f.write("\n")
f.close()

print("Start predicting...")
probs = model.predict_proba(test_count_vec,verbose=2)
print("predicted probs:", probs[0:5])
print("AUC of Test:", sklearn.metrics.roc_auc_score(y_test, probs))

out_data = {'predict':probs > .5, 'generate':probs, 'non-generated': 1-probs, 
            'id':Test.id, 'original': Test.domain}

fpr, tpr, _ = roc_curve(y_test,probs)
with plt.style.context('bmh'):
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.plot(fpr, tpr,
             lw=2, label='Test AUC: %s' % auc(fpr,tpr))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.tick_params(axis='both', labelsize=22)
    plt.savefig('lstm_test_results.png')

f = open("out_data3.txt","w")

f.write("predict, generate, non-generated, groundtruth, id, original")
f.write("\n")
for i in range(len(probs)):
    if probs[i] > .5:
        predict="generated"
    else:
        predict="non-generated"
    generate=probs[i][0]
    non_generated = 1-probs[i][0]
    id = Test.id[i]
    groundtruth=Test.type[i]
    original = Test.domain[i]
    f.write("{},{},{},{},{},{}".format(predict, generate, non_generated, groundtruth, id, original))
    f.write("\n")

f.close()


