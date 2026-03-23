import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib       #stores and loads the model


#Load dataset
df = pd.read_csv("advertising.csv")

#Convert Timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
df.drop("Timestamp", axis = 1, inplace = True)

#Separate features and target
X = df.drop("Clicked on Ad", axis = 1)
y = df["Clicked on Ad"]

#Drop high-cardinality text column
# = X.drop("Ad Topic Line", axis = 1)

#One-hot encoding (categorial values converted into numerical values)pip 
X = pd.get_dummies(X, drop_first = True)

print("Final feature shape:", X.shape)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Scale numerical values (bringing all values to comparable ranges) min_max scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train Logistic Regression
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)

#Predictions
y_pred = model.predict(X_test)

#Evaluation
print("\nAccuracy:\n", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
joblib.dump(model,"ctr_model.pkl")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(X.columns,"columns.pkl")
print("Successfully pickle files saved")


