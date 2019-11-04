from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
            
A = pd.read_csv('info_train.csv')
A = A[:130000]
lbl = A.loc[:,'TARGET'].tolist()
A.replace(np.nan, -1,inplace = True)
A = pd.get_dummies(A, )
B = A.values.tolist()
train_d, train_lbl = B[:100000],lbl[:100000]
test_d, test_lbl = B[100000:130000],lbl[100000:130000]
train_d = train_d[:25000]
test_d = test_d[:25000]
print("Starting PCA Dimensionality Reduction")
scaler = StandardScaler()
scaler.fit(train_d)
train_d = scaler.transform(train_d)
test_d = scaler.transform(test_d)

pca = PCA(.90)
pca.fit(train_d)
train_d = pca.transform(train_d)
test_d = pca.transform(test_d)
print("Loading dataset into Logistic Regression Classifier...")

train_d = train_d[:10000]
test_d = test_d[:10000]
train_lbl = train_lbl[:10000]
test_lbl = test_lbl[:10000]
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_d, train_lbl)
predicted = logisticRegr.predict(test_d)
expected = test_lbl[:10000]

print("Accuracy: ", accuracy_score(expected, predicted))
A = confusion_matrix(expected, predicted)
print(A)
for i in range(len(A)):
    for j in A[i]:
        print(j,end=' ')
    print(' ')
plt.imshow(A, cmap='binary')

#Random Forests with PCA
print("Loading dataset...")
A = pd.read_csv('info_train.csv')
A = A[:175000]
lbl = A.loc[:,'TARGET'].tolist()
A = A.drop(A.columns[[0, 1]],axis=1)
A.replace(np.nan, -1,inplace = True)

A = pd.get_dummies(A, )
B = A.values.tolist()
images, labels = B[:175000],lbl[:175000]
clf = RandomForestClassifier(n_estimators=100)
train_x = images[:25000]
train_y = labels[:25000]
test_x = images[100000:130000]

print("Starting PCA Dimensionality Reduction...")
scaler = StandardScaler()
scaler.fit(train_x)
train_d = scaler.transform(train_x)
test_d = scaler.transform(test_x)

pca = PCA(.90)
pca.fit(train_d)
train_d = pca.transform(train_d)
test_d = pca.transform(test_d)
print("PCA Dimensionality Reduction complete")

print("Loading dataset into Random Forest Classifier...")
clf.fit(train_x, train_y)
expected = labels[100000:130000]
predicted = clf.predict(test_x)

print("Accuracy: ", accuracy_score(expected, predicted))
B = confusion_matrix(expected, predicted)
print(B)
for i in range(len(B)):
    for j in B[i]:
        print(j,end=' ')
    print(' ')

plt.imshow(B, cmap='binary')

