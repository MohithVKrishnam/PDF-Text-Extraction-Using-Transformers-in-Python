import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch

# Load the summarization model
@st.cache_resource
def load_model():
    summarizer = pipeline("summarization")
    return summarizer

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=1000):
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + '.'
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + '.'
    
    if current_chunk:  # Add the last chunk
        chunks.append(current_chunk)
    
    return chunks

# Function to perform summarization
def summarize_text(text):
    summarizer = load_model()
    chunks = split_text_into_chunks(text, chunk_size=1000)  # Adjust chunk size based on model limit
    summaries = []

    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            st.write(f"Error summarizing chunk: {e}")
    
    # Combine all summaries
    combined_summary = " ".join(summaries)
    return combined_summary

# Function to find the most similar sentence
def find_most_similar_sentence(query, document):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = document.split('.')
    sentence_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])

    similarities = []
    for i, sentence in enumerate(sentences):
        similarity = torch.nn.functional.cosine_similarity(torch.tensor(query_embedding), torch.tensor([sentence_embeddings[i]]))
        similarities.append((sentence, similarity.item()))

    # Sort by similarity
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_similarities[0][0] if sorted_similarities else "No similar sentence found."

# Streamlit App
st.title("PDF Text Analysis with Transformers")
st.write("Upload a PDF file to summarize the text or find similar sentences.")

# PDF file upload
pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

if pdf_file is not None:
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_file)
    
    if len(text) > 0:
        # Display extracted text
        st.subheader("Extracted Text:")
        st.write(text[:1000] + '...')  # Display the first 1000 characters
        
        # Summarize text
        summary = summarize_text(text)
        st.subheader("Summary:")
        st.write(summary)
        
        # Query input for finding similar sentences
        query = st.text_input("Enter a query to find the most similar sentence in the document:")
        
        if query:
            similar_sentence = find_most_similar_sentence(query, text)
            st.subheader("Most Similar Sentence:")
            st.write(similar_sentence)
    else:
        st.write("No text could be extracted from the PDF.")