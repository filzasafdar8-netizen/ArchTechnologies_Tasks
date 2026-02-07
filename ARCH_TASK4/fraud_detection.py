import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("creditcard.csv")

# Balance classes (simplest approach)
fraud = df[df['Class']==1]
non_fraud = df[df['Class']==0].sample(len(fraud))
data = pd.concat([fraud, non_fraud])

# Features & target
X = data.drop('Class', axis=1)
y = data['Class']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualization
sns.countplot(x='Class', data=data)
plt.title("Fraud vs Legitimate Transactions")
plt.show()