import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = {
    'Offer': ['Yes', 'Yes', 'No', 'No', 'No', 'Yes'],
    'Money': ['Yes', 'No', 'Yes', 'No', 'No', 'No'],
    'Meeting': ['No', 'No', 'No', 'Yes', 'Yes', 'Yes'],
    'Actual': ['Spam', 'Spam', 'Spam', 'Not Spam', 'Not Spam', 'Not Spam']
}
df = pd.DataFrame(data)

from sklearn.preprocessing import LabelEncoder
le_offer, le_money, le_meeting, le_target = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()

df['Offer_enc'] = le_offer.fit_transform(df['Offer'])
df['Money_enc'] = le_money.fit_transform(df['Money'])
df['Meeting_enc'] = le_meeting.fit_transform(df['Meeting'])
y = le_target.fit_transform(df['Actual'])

X = df[['Offer_enc', 'Money_enc', 'Meeting_enc']]

model = CategoricalNB()
model.fit(X, y)
y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred, labels=[1, 0]) # 1=Spam, 0=Not Spam

print("--- Spam Detection Confusion Matrix ---")
print(classification_report(y, y_pred, target_names=['Not Spam', 'Spam']))

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Spam', 'Predicted Not Spam'],
            yticklabels=['Actual Spam', 'Actual Not Spam'])
plt.title("Naive Bayes Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()
