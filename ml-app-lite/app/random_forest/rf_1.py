import joblib
from sklearn.model_selection import RandomForestClassifier
from sklearn.metrics import classification_report

data = joblib.load("ml_data/prepared_data.joblib")

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

model = RandomForestClassifier(n_estimators=100, random_state=data["config"].random_seed)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


