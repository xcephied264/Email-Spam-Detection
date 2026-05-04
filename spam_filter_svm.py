import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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

# 4. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Print results
print("Train/Test split completed.")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTF-IDF conversion completed.")
print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape:", X_test_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.svm import LinearSVC

# 7. Train SVM model
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)

# 8. Predictions
svm_pred = svm_model.predict(X_test_tfidf)

# 9. Evaluation
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)

print("\nSVM Results")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 10. Confusion Matrix for SVM
cm = confusion_matrix(y_test, svm_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot()

plt.title("SVM Confusion Matrix")
plt.show()