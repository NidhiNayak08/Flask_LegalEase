from flask import Flask, request, render_template, redirect, url_for
from transformers import pipeline
from PyPDF2 import PdfReader
import re
import os

app = Flask(__name__)

# Function to read a PDF document and extract text
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to clean and process text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text

# Function to split text into overlapping chunks
def split_into_chunks(text, chunk_size=500, overlap=100):
    words = text.split()  # Split text into words/tokens
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = words[start:end]
        chunks.append(' '.join(chunk))
        start += chunk_size - overlap  # Overlap by 'overlap' tokens
    return chunks

def summarize_chunks(chunks, max_length=150, min_length=40):
    summarizer = pipeline('summarization', model='t5-small', framework='pt')  # Use PyTorch
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return summaries


# Function to merge summarized chunks
def merge_summaries(summaries):
    return ' '.join(summaries)

# Function to summarize a document with overlapping chunks
def summarize_document(text):
    cleaned_text = clean_text(text)
    chunks = split_into_chunks(cleaned_text, chunk_size=500, overlap=100)
    chunk_summaries = summarize_chunks(chunks)
    final_summary = merge_summaries(chunk_summaries)
    return final_summary

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file is part of the request
        if 'pdf_file' not in request.files:
            return 'No file part'
        
        file = request.files['pdf_file']
        
        if file.filename == '':
            return 'No selected file'
        
        if file and file.filename.endswith('.pdf'):
            # Save the uploaded PDF
            pdf_path = os.path.join('static', file.filename)
            file.save(pdf_path)
            
            # Extract text from the PDF
            document_text = extract_text_from_pdf(pdf_path)
            
            # Summarize the document
            summary = summarize_document(document_text)
            
            # Remove the saved PDF after summarization
            os.remove(pdf_path)
            
            return render_template('summary.html', summary=summary)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
