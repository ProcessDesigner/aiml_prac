# =======================================
# Experiment 7: Medical Reviews Analysis
# =======================================

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download stopwords (only first run)
nltk.download('stopwords')

# ==========================
# Step 2: Load Dataset
# ==========================
df = pd.read_csv('amazon_health_reviews.csv')

print("🔹 First 5 Rows of Data:")
print(df.head())

# Check for missing values
print("\n🔹 Missing Values per Column:")
print(df.isnull().sum())

# Drop missing reviews
df.dropna(subset=['Review'], inplace=True)

# ==========================
# Step 3: Data Preprocessing
# ==========================
def clean_text(text):
    text = text.lower()                            # Lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)                # Remove non-word characters
    text = re.sub(r'\s+', ' ', text).strip()       # Remove extra spaces
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

df['Clean_Review'] = df['Review'].apply(clean_text)

# ==========================
# Step 4: Encode Sentiment
# ==========================
# Example: Sentiment column contains 'positive', 'negative', 'neutral'
df['Sentiment'] = df['Sentiment'].map({'positive': 1, 'negative': 0, 'neutral': 2})

# ==========================
# Step 5: Feature Extraction
# ==========================
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Clean_Review']).toarray()
y = df['Sentiment']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================
# Step 6: Model Building
# ==========================
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# ==========================
# Step 7: Model Evaluation
# ==========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")
print("\n🔹 Confusion Matrix:\n", cm)
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# ==========================
# Step 8: Visualization
# ==========================
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Sentiment Analysis')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Sentiment distribution
plt.figure(figsize=(5,4))
sns.countplot(x='Sentiment', data=df, palette='Set2')
plt.title('Sentiment Distribution in Health Reviews')
plt.xticks(ticks=[0,1,2], labels=['Negative','Positive','Neutral'])
plt.show()

# ==========================
# Step 9: Example Prediction
# ==========================
sample_review = ["This medicine really helped me recover fast"]
sample_clean = [clean_text(sample_review[0])]
sample_vector = tfidf.transform(sample_clean).toarray()
pred = model.predict(sample_vector)[0]

sentiment_dict = {0: "Negative", 1: "Positive", 2: "Neutral"}
print("\n💬 Sample Review Prediction:", sentiment_dict[pred])
