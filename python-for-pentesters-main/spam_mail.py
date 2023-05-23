import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
training_mails = [
    ("Get free money now!!!", "spam"),
    ("Hello, how are you?", "not spam"),
    ("Claim your prize!", "spam"),
    ("Meeting tomorrow", "not spam"),
    ("Buy cheap watches", "spam")
]

# Preprocess the training data
train_documents = [re.sub(r'\W', ' ', mail[0].lower()) for mail in training_mails]
train_labels = [mail[1] for mail in training_mails]

# Initialize the CountVectorizer
vectorizer = CountVectorizer()
train_matrix = vectorizer.fit_transform(train_documents)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(train_matrix, train_labels)

# User input
user_input = input("Enter a mail: ")

# Preprocess the user input
user_input_processed = re.sub(r'\W', ' ', user_input.lower())
user_input_matrix = vectorizer.transform([user_input_processed])

# Predict the label for user input
predicted_label = classifier.predict(user_input_matrix)[0]

# Print the predicted label
print("Predicted Label:", predicted_label)
