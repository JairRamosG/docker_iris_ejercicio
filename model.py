import pickle
from sklearn.datasets import load_iris
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# loading the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

# Build the model
clf = RandomForestClassifier(n_estimators=10)

# Train the classifier
clf.fit(X_train, y_train)

# Predictions
predicted = clf.predict(X_test)

# Check accuracy
#print(accuracy_score(predicted, y_test))

with open('./model.pkl', 'wb') as model_pkl:
        pickle.dump(clf, model_pkl, protocol=2)
