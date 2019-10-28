import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
df = pd.read_csv("file.csv")


X = df.drop('risk', axis=1)
y = df['risk']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print("Accuracy:",sk.metrics.accuracy_score(y_test, y_pred))

a1 = int(input("enter number of curves 9o to 60 :"))
a2 = int(input("enter number of curves 60 to 40 :"))
a3 = int(input("enter number of curves 40 to 20 :"))
a4 = int(input("enter the quality of road\n1-low\n2-medium\n3-good\n:"))


stc = [[a1,a2,a3,a4]]


y_pred= svclassifier.predict(stc)

print("The risk factor is " + y_pred[0])

