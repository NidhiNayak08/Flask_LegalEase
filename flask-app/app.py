from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from PyPDF2 import PdfReader
import re
import os
from rouge_score import rouge_scorer
from bert_score import score
from fpdf import FPDF

app = Flask(__name__)

# Directory to store summaries
SUMMARIES_DIR = 'summaries'

# Ensure the summaries directory exists
if not os.path.exists(SUMMARIES_DIR):
    os.makedirs(SUMMARIES_DIR)

# Load the tokenizer and model for Legal-BERT
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-small-uncased")
model = AutoModelForTokenClassification.from_pretrained("nlpaueb/legal-bert-small-uncased")

# Function to read a PDF document and extract text
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.encode('utf-8', errors='replace').decode('utf-8')  # Ensures UTF-8 encoding

# Function to clean and process text using Legal-BERT
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text

# Function to split text into overlapping recursive chunks
def recursive_chunk(text, chunk_size=600, min_length=150):
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size // 2  # Half overlapping
    return chunks

def summarize_chunks(chunks, max_length=150, min_length=40):
    summarizer = pipeline('summarization', model='t5-small', framework='pt')
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return summaries

# Function to merge summarized chunks
def merge_summaries(summaries):
    return ' '.join(summaries)

# Function to evaluate summary using ROUGE
def evaluate_summary_rouge(reference_summary, generated_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores

# Function to evaluate summary using BERTScore
def evaluate_summary_bert(reference_summary, generated_summary):
    P, R, F1 = score([generated_summary], [reference_summary], lang="en", verbose=True)
    return F1.mean().item()  # Return F1 score

# Function to summarize a document with recursive chunks
def summarize_document(text):
    cleaned_text = clean_text(text)
    chunks = recursive_chunk(cleaned_text, chunk_size=600)  # Adjusted chunk size
    chunk_summaries = summarize_chunks(chunks)
    final_summary = merge_summaries(chunk_summaries)
    return final_summary

# Helper function to save summary as a PDF
def save_summary_as_pdf(summary_text, summary_filename):
    summary_text = summary_text.encode('utf-8', errors='replace').decode('utf-8')  # Ensure UTF-8 encoding
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add content to the PDF
    pdf.multi_cell(0, 10, summary_text)
    
    # Save PDF to the summaries directory
    pdf.output(os.path.join(SUMMARIES_DIR, summary_filename))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ensure the file is uploaded
        if 'pdf_file' not in request.files:
            return 'No file uploaded!', 400
        
        pdf_file = request.files['pdf_file']
        
        if pdf_file.filename == '':
            return 'No selected file', 400
        
        # Save the file to a location (optional) and extract the text
        if pdf_file:
            file_path = os.path.join('static', pdf_file.filename)
            pdf_file.save(file_path)

            print(f"PDF saved at: {file_path}")  # Debugging line to ensure file is saved
            
            # Extract text from the PDF
            document_text = extract_text_from_pdf(file_path)
            
            # Ensure document text is encoded in UTF-8
            document_text = document_text.encode('utf-8', errors='replace').decode('utf-8')
            
            # Summarize the document
            generated_summary = summarize_document(document_text)

            # Example reference summary (replace this with actual human reference summaries)
            reference_summary = "This is the reference summary for the document."  # Update with actual reference summary

            # Evaluate the generated summary
            rouge_scores = evaluate_summary_rouge(reference_summary, generated_summary)
            bert_score = evaluate_summary_bert(reference_summary, generated_summary)

            # Print ROUGE and BERT scores to the terminal
            print("\nROUGE Scores:")
            print(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}")
            print(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}")
            print(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")
            print(f"\nBERT Score (F1): {bert_score:.4f}")

            # Save summary as PDF
            summary_id = len(os.listdir(SUMMARIES_DIR)) + 1
            summary_filename = f'summary_{summary_id}.pdf'
            save_summary_as_pdf(generated_summary, summary_filename)

            # Remove the original PDF file after summarization
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} removed after summarization")  # Debugging line to confirm file removal
            else:
                print(f"File {file_path} not found for deletion")  # Debugging line if file is missing
            
            return render_template('summary.html', summary=generated_summary)
    return render_template('index.html')

@app.route('/mydocs')
def my_docs():
    summaries = os.listdir(SUMMARIES_DIR)
    return render_template('docs.html', summaries=summaries)

@app.route('/mydocs/<summary_name>')
def view_summary(summary_name):
    summary_path = os.path.join(SUMMARIES_DIR, summary_name)
    
    # Check if it's a PDF and send it to the user for download/viewing
    if summary_name.endswith('.pdf'):
        return send_from_directory(SUMMARIES_DIR, summary_name)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
