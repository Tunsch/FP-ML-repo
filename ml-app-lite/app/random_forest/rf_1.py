import joblib
import sys
sys.path.append("../app")

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from app.config import ExperimentConfig

config = ExperimentConfig()

data_path = config.ml_data_dir / "prepared_data.joblib"

if not data_path.exists():
    raise FileNotFoundError(f"Prepared data not found at {data_path}")
data = joblib.load(data_path)

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

model = RandomForestClassifier(n_estimators=100, random_state=data["config"].random_seed)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


