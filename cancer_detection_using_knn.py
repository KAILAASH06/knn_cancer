import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


df = pd.read_csv('genecancer.csv')
X = df.drop('Cancer Present',axis=1)
y = df['Cancer Present']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Training samples", len(X_train))
print("Test samples", len(X_test))
scaler = StandardScaler()

#scaled_X_train = scaler.fit_transform(X_train)
#scaled_X_test = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train,y_train)

gene_model = 'knn_model.pkl'
pickle.dump(knn_model, open(gene_model,'wb'))
print("learned knn model created")

#evaluation
y_pred = knn_model.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))