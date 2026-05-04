import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load dataset
df = pd.read_csv("spam_ham_dataset.csv", encoding="latin-1")
print(df.columns)  

# 2. Keep only useful columns
df = df[['label', 'text']].copy()
df.columns = ['label', 'message']

# 3. Convert labels to numbers
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# 4. Define features and labels
X = df['message']
y = df['label_num']

# 5. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 7. Print results
print("Train/Test split completed.")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTF-IDF conversion completed.")
print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape:", X_test_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 8. Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# 9. Make predictions
y_pred = nb_model.predict(X_test_tfidf)

# 10. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nNaive Bayes Results")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

from sklearn.svm import LinearSVC

# 11. Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred)

disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=["Ham", "Spam"])
disp_nb.plot()

plt.title("Naive Bayes Confusion Matrix")
plt.show()
