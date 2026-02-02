import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("spam_Emails_data.csv")

label_col = df.columns[0]
text_col = df.columns[1]

df[text_col] = df[text_col].astype(str)
df = df[df[text_col].str.strip() != ""]
df = df[df[text_col].str.lower() != "nan"]

df[label_col] = df[label_col].astype(str).str.strip().str.lower()
df[label_col] = df[label_col].map({"spam": 1, "ham": 0})
df = df.dropna(subset=[label_col])

x = df[text_col]
y = df[label_col]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state = 20, stratify = y)

x_train = x_train.fillna("")
x_test = x_test.fillna("")


vectorizer = TfidfVectorizer(stop_words = "english",min_df=2,dtype=np.float32)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def predict_spam(email):
    email_vec = vectorizer.transform([email])
    result = model.predict(email_vec)
    return "Spam" if result[0] == 1 else "Not Spam"

print(predict_spam("Congratulations! You have won a ₹10,00,000 cash prize. Call now to claim."))
print(predict_spam("FREE entry in a weekly lucky draw! Text WIN to 56789 now."))

