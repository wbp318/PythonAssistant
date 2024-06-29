import PyPDF2
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract text from PDF
def extract_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to read spreadsheet
def read_spreadsheet(file_path):
    return pd.read_excel(file_path)

# Advanced NLP processing
def process_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# Train a classifier
def train_classifier(data, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    print(f"Classifier accuracy: {accuracy}")
    return vectorizer, classifier

# Analyze spreadsheet
def analyze_spreadsheet(df):
    # Basic statistical analysis
    summary = df.describe(include='all')
    
    # Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    
    # Visualization for numeric columns
    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap of Numeric Columns')
        plt.savefig('correlation_heatmap.png')
        plt.close()
    
    # Basic info for non-numeric columns
    non_numeric_info = {col: df[col].value_counts().head() for col in non_numeric_columns}
    
    return summary, non_numeric_info

# Main function
def analyze_document(file_path, file_type):
    if file_type == 'pdf':
        content = extract_pdf_text(file_path)
        processed_content = process_text(content)
        # Here you would use your trained classifier
        # result = classifier.predict(vectorizer.transform([processed_content]))
        return "PDF analysis complete"
    elif file_type == 'spreadsheet':
        df = read_spreadsheet(file_path)
        summary, non_numeric_info = analyze_spreadsheet(df)
        result = f"Spreadsheet analysis complete.\nSummary of numeric columns:\n{summary}\n\nTop values in non-numeric columns:\n"
        for col, counts in non_numeric_info.items():
            result += f"\n{col}:\n{counts}\n"
        return result
    else:
        return "Unsupported file type"

# Simple question answering function
def answer_question(question, document_content):
    # This is a very basic implementation. In a real system, you'd use more advanced NLP techniques.
    processed_question = process_text(question)
    processed_content = process_text(document_content)
    
    # Simple keyword matching
    if all(word in processed_content for word in processed_question.split()):
        return "The document likely contains information relevant to your question."
    else:
        return "I couldn't find information directly related to your question in the document."

# Usage
result = analyze_document('SmokinAssMfs.xlsx', 'spreadsheet')
print(result)

# Example of question answering
# Uncomment these lines if you want to test PDF question answering
# document_content = extract_pdf_text('example.pdf')
# question = "What is the main topic of this document?"
# answer = answer_question(question, document_content)
# print(f"Q: {question}\nA: {answer}")