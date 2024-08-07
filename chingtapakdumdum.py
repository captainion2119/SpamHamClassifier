import chardet
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split as ttsplit
from sklearn import svm
import pandas as pd
import pickle
import numpy as np
import imaplib
import email
import json

# nltk.download('punkt')

# Detect file encoding
file = "spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
encoding = result['encoding']

# Read the dataset file
df = pd.read_csv(file, encoding=encoding)
message_X = df.iloc[:, 1]  # EmailText column
labels_Y = df.iloc[:, 0]  # Label

# Initialize the stemmer
lstem = LancasterStemmer()

def preprocess(messages):
    processed_messages = []
    for msg in messages:
        # Filter out non-alphabetic characters
        msg = ''.join(filter(lambda ch: ch.isalpha() or ch == " ", msg))
        # Tokenize the messages
        words = word_tokenize(msg)
        # Stem the words
        processed_messages.append(' '.join([lstem.stem(word) for word in words]))
    return processed_messages

message_x = preprocess(message_X)

# Vectorization process
tfvec = TfidfVectorizer(stop_words='english')
x_new = tfvec.fit_transform(message_x).toarray()

# Replace ham and spam labels with 0 and 1 respectively
y_new = np.array(labels_Y.replace(to_replace=['ham', 'spam'], value=[0, 1]))

# Split dataset into training and testing parts
x_train, x_test, y_train, y_test = ttsplit(x_new, y_new, test_size=0.2, shuffle=True)

# Train the SVM classifier
classifier = svm.SVC()
classifier.fit(x_train, y_train)

# Store the classifier and message features for prediction
pickle.dump({'classifier': classifier, 'message_x': message_x, 'tfvec': tfvec},
            open("training_data.pkl", "wb"))

# Streamlit App
st.title("Spam Detector")

# Load classifier and message data
datafile = pickle.load(open("training_data.pkl", "rb"))
message_x = datafile["message_x"]
classifier = datafile["classifier"]
tfvec = datafile["tfvec"]

def connect_email(username, password):
    # Connect to the IMAP server
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(username, password)
    mail.select("inbox")  # Select inbox or another folder
    return mail

def preprocess_message(message):
    # Preprocess the message
    msg = ''.join(filter(lambda ch: ch.isalpha() or ch == " ", message))
    words = word_tokenize(msg)
    stemmed_message = ' '.join([lstem.stem(word) for word in words])
    return stemmed_message

def classify_email(body):
    processed_msg = preprocess_message(body)
    vectorized_msg = tfvec.transform([processed_msg]).toarray()

    # Predict the label
    prediction = classifier.predict(vectorized_msg)[0]
    result = "spam" if prediction == 1 else "ham"
    return result

def fetch_emails(mail):
    email_texts = {}
    try:
        # Fetch emails from inbox
        result, data = mail.search(None, "ALL")  # Fetch all emails
        for num in data[0].split():
            result, data = mail.fetch(num, "(RFC822)")
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)

            # Extract email content (subject and body)
            subject = msg["subject"]
            body = ""

            # Process each part of the message
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                if content_type == "text/plain" and "attachment" not in content_disposition:
                    # Decode text parts
                    try:
                        payload = part.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            body += payload.decode('utf-8', 'ignore')
                        else:
                            body += payload
                    except Exception as e:
                        print(f"Error decoding message: {e}")
                        continue

            if body:
                classification = classify_email(body)
                email_texts[subject] = {'body': body, 'classification': classification}

        # Save emails to file
        with open("emails.json", "w") as f:
            json.dump(email_texts, f)

    except imaplib.IMAP4.error as e:
        print(f"IMAP error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def load_emails_from_file():
    with open("emails.json", "r") as f:
        email_texts = json.load(f)
    return email_texts

# Streamlit UI components
st.sidebar.header("Email Credentials")
username = st.sidebar.text_input("Email", value="talavarakshatha57@gmail.com")
password = st.sidebar.text_input("Password", type="password", value="ugwx rprc mcou lmyg")

if st.sidebar.button("Fetch Emails"):
    mail = connect_email(username, password)
    fetch_emails(mail)
    st.sidebar.success("Emails fetched and saved to file.")

email_texts = load_emails_from_file()
subjects = list(email_texts.keys())

selected_subject = st.selectbox("Select an email", subjects)

if selected_subject:
    email_data = email_texts[selected_subject]
    body = str(email_data['body'])
    classification = email_data['classification']
    result = classify_email(body)
    st.write(f"Subject: {selected_subject}")
    st.write(f"Body: {body}")
    st.write(f"Classification: {classification}")

# Evaluate accuracy
accuracy = classifier.score(x_test, y_test)
st.write(f"Accuracy of the model: {accuracy:.2%}")
