import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Create static/images folder if not exist
os.makedirs("static/images", exist_ok=True)

# Load dataset
df = pd.read_csv('C-section_Dataset.csv')

# Fix categorical values
df['Blood of Pressure'] = df['Blood of Pressure'].replace('low', 'Low')
df['Caesarian'] = df['Caesarian'].replace('yes', 'Yes')

# Convert columns to categorical types to avoid issues with plotting
df['Delivery No'] = df['Delivery No'].astype('category')
df['Blood of Pressure'] = df['Blood of Pressure'].astype('category')
df['Heart Problem'] = df['Heart Problem'].astype('category')
df['Caesarian'] = df['Caesarian'].astype('category')

# Save count plots
sns.countplot(x=df['Delivery No']).get_figure().savefig("static/images/delivery_no.png")
plt.clf()  # Clear the plot
sns.countplot(x=df['Blood of Pressure']).get_figure().savefig("static/images/blood_pressure.png")
plt.clf()
sns.countplot(x=df['Heart Problem']).get_figure().savefig("static/images/heart_problem.png")
plt.clf()
sns.countplot(x=df['Caesarian']).get_figure().savefig("static/images/caesarian.png")
plt.clf()

# Save dataset preview and stats
df.head().to_csv("static/df_head.csv")
df.describe().T.to_csv("static/df_stats.csv")

# Preprocessing
df_dummy = pd.get_dummies(df, drop_first=True)
X = df_dummy.drop(['Caesarian_Yes'], axis=1)
y = df_dummy['Caesarian_Yes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train model
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
# Predict on test data
y_pred = logreg.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy on Test Data:", round(accuracy * 100, 2), "%")

# Optional: Detailed evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
pickle.dump(logreg, open('caesarean.pkl', 'wb'))

print("✅ Model trained and saved as caesarean.pkl")
print("✅ Graphs saved in static/images/")

