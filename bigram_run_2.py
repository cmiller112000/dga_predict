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
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation
import sklearn
from sklearn import feature_extraction
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


def ext(domain_name):
    domains =[]           
    try:
                    ext = tldextract.extract(domain_name)
                    if ext.subdomain == "www" or ext.subdomain == '':
                        domains = ext.domain
                    else:
                        domains = '.'.join(ext[:2])
    except:
                    domains = domain_name
    return domains

# Create feature vectors
# Create feature vectors
print("vectorizing data")
ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2, 2))

count_vec = ngram_vectorizer.fit_transform(X_domain)

max_features = count_vec.shape[1]
print("size of X after bigram:", count_vec.shape)

print("example of X", count_vec.todense()[0:5,:])

# Convert labels to 0-1
y = [0 if x == 'non-generated' else 1 for x in labels0]
y_test = [0 if x == 'non-generated' else 1 for x in Test.type]


ngram_vectorizer2 = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2, 2))

test_count_vec = ngram_vectorizer.fit_transform(X_test)

max_features2 = test_count_vec.shape[1]
#test_count_vec.todense()[0:5]
print("test data bigram:", test_count_vec.todense()[0:5])


def build_model(max_features):
    """Builds logistic regression model"""
    model = Sequential()
    model.add(Dense(1, input_dim=max_features, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam')
    return model

X_train, X_validate, y_train, y_validate = train_test_split(count_vec, y, test_size=0.2)

X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.1)
best_iter = -1
best_auc = 0.0
out_data = {}
batch_size = 128
max_epoch = 25

if not os.path.isfile("bigram.model"):
    model = build_model(max_features)
    for ep in range(max_epoch):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=1,verbose=2)
        t_probs = model.predict_proba(X_holdout.todense(),verbose=2)
        t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)
        print('Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc))
        if t_auc > best_auc:
            best_auc = t_auc
            best_iter = ep
        else:
            # No longer improving...break and calc statistics
            if (ep-best_iter) > 2:
                break
    try:
        model.save("bigram.model")
    except:
        pass
else:
    model = load_model("bigram.model")

probs_validate = model.predict_proba(X_validate.todense(),verbose=2)
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
    plt.savefig('bigram_vld_results.png')

f = open("out_data2.txt","w")
f.write("y_pred, y_validate")
f.write("\n")
for i in range(len(probs_validate)):
    f.write("{},{}".format(out_data_validate['y_pred'][i], out_data_validate['y_validate'][i]))
    f.write("\n")
f.close()

print("Start predicting...")
probs = model.predict_proba(test_count_vec.todense(),verbose=2)
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
    plt.savefig('bigram_test_results.png')

f = open("out_data3.txt","w")

f.write("predict,generated,non-generated,groundtruth,id,original")
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


