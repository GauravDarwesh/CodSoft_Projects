import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv('spam.csv', encoding='latin-1')


X = df['v2']  
y = df['v1']  


tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(X)


naive_bayes = MultinomialNB()
naive_bayes.fit(X_tfidf, y)


def classify_message(message):
 
    message_tfidf = tfidf_vectorizer.transform([message])
    
  
    prediction = naive_bayes.predict(message_tfidf)[0]
    
    if prediction == 'spam':
        return "This message is classified as SPAM."
    else:
        return "This message is classified as LEGITIMATE."


message_to_classify = input("Please enter a message to classify: ")
result = classify_message(message_to_classify)
print(result)
