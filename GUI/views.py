# from io import BytesIO
# import pickle
# from pathlib import Path
# import re
# from urllib.parse import urlparse

# Text-Summarization
# import PyPDF2
import pickle
import requests
# from io import BytesIO
from dotenv import load_dotenv
# from datetime import datetime, time
# from transformers import pipeline
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from django.shortcuts import render
from .models import Paper , RecentPaper 
from django.shortcuts import render, get_object_or_404
from sentence_transformers import SentenceTransformer, util
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from langchain.text_splitter import RecursiveCharacterTextSplitter

#display
import base64

#QA
import os
# import time
# import shutil
# from tqdm import tqdm
# from urllib.parse import urlparse
# from django.core.cache import cache
# from django.http import JsonResponse
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import Chroma
# from .pdf_constants  import CHROMA_SETTINGS
# from langchain.llms import GPT4All, LlamaCpp
from django.shortcuts import render, get_object_or_404
from sentence_transformers import SentenceTransformer, util
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from .ingest import process_documents, does_vectorstore_exist


load_dotenv()


# -------------------------------------------------------Home page------------------------------------------------------------------------------------
def index(request):
    
    """
    Render the index page of the Django web application.
    
    Args:
        request (HttpRequest): The HTTP request made by the user.
    
    Returns:
        HttpResponse: The rendered HTML content of the index page with recent research papers categorized for display.
    steps:
         1. Retrieving recent papers from different categories and passing through the context
    """
    
    ml_papers = RecentPaper.objects.filter(category='cs.CL').order_by("-published_date")[:3]
    nlp_papers = RecentPaper.objects.filter(category='cs.LG').order_by("-published_date")[:3]
    ai_papers = RecentPaper.objects.filter(category='cs.AI').order_by("-published_date")[:3]
    cv_papers = RecentPaper.objects.filter(category='cs.CV').order_by("-published_date")[:3]
    
    context = {
        'ml_papers': ml_papers,
        'nlp_papers': nlp_papers,
        'ai_papers': ai_papers,
        'cv_papers': cv_papers,
    }
    return render(request, 'index.html', context)

# ----------------------------------------------search_papers-------------------------------------------------------------------------------------------
def search_papers(request):
    """
    Search for relevant research papers using a user query and recognized speech, and display recommended papers.
    
    Args:
        request (HttpRequest): The HTTP request made by the user.
    
    Returns:
        HttpResponse: The rendered HTML content showing recommended research papers based on the user's query.
    """
    try:                
        embeddings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "PaperMate_ui", "GUI", "embedded_models", "Embeddings", "Embeddings.pkl")
        sentences_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "PaperMate_ui", "GUI", "embedded_models", "Sentences", "Sentences.pkl")

        if request.method == 'POST':
            # Retrieving user input and recognized speech.
            query = request.POST.get('query', '').strip()
            recognized_text = request.POST.get('recognized_text', '').strip()

            # Combining query and recognized speech if available.
            if recognized_text:
                query += ' ' + recognized_text
        
            # Check if either the query or recognized_text is empty
            if not query:
                return render(request, 'index.html', {'error_message': 'Please enter a query or use speech input.'})
            
            # Load pre-trained SentenceTransformer model
            model = SentenceTransformer('all-MiniLM-L6-v2')
                        
            # Load pre-calculated sentence embeddings
            with open(sentences_path, 'rb') as f:
                sentences_data = pickle.load(f)
            with open(embeddings_path, 'rb') as f:
                embeddings_data = pickle.load(f)
            
            # Generate a prompt template based on the user query
            prompt_template = f"Could you kindly generate top ArXiv paper recommendations based on: '{query}'? Your focus on recent research and relevant papers is greatly appreciated."
            
            # Encoding user query and calculating cosine similarity.
            query_embedding = model.encode([prompt_template])
            cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings_data)[0]
            
            # Get indices of top 4 similar papers
            top_indices = cosine_scores.argsort(descending=True)[:4]
            top_indices = top_indices.cpu().numpy()  # Convert to numpy array
            top_paper_titles = [sentences_data[i.item()] for i in top_indices]  # Access elements using integer indices
            
            # Get paper details from the database
            recommended_papers = Paper.objects.filter(title__in=top_paper_titles)
            
            search_error = len(recommended_papers) == 0
            
            return render(request, 'recommendations.html', {'papers': recommended_papers, 'recommended_papers': recommended_papers, 'search_error': search_error})
            
    except Exception as e:
        print( f"An error occurred: {str(e)}")
        return render(request, 'index.html', {'error_message': f"An error occurred: {str(e)}"})
    
    return render(request, 'index.html')

# ----------------------------------------------Recommendations-----------------------------------------------------------------------------------------
def recommendations(request):
    """
    Retrieve and render all research paper recommendations for display on the recommendations page.
    
    Args:
        request (HttpRequest): The HTTP request made by the user.
    
    Returns:
        HttpResponse: The rendered HTML content displaying all recommended research papers.
    """
    papers = Paper.objects.all()
    return render(request, 'recommendations.html', {'papers': papers})

# -------------------------------------Summarization----------------------------------------------------------------------------------------------------
# -----------------------------------------Display Paper---------------------------------------------------------------------------------------------------
def display(request, paper_id):
    """
    Retrieve and render recommended research paper summaries for display on the summarization page.

    Args:
        request (HttpRequest): The HTTP request made by the user (summarize).
        paper_id (int): Identifier for the specific paper being summarized (corresponding Paper ID).

    Returns:
        HttpResponse: The rendered HTML content displaying the summarized research paper.
    """
    paper = get_object_or_404(Paper, ids=paper_id)
    
    if paper.pdf_content:
        pdf_base64 = base64.b64encode(paper.pdf_content).decode("utf-8")
    else:
        pdf_url = f"https://arxiv.org/pdf/{paper.ids}.pdf"
        try:
            response = requests.get(pdf_url , stream=True)
            response.raise_for_status() 
            pdf_content = response.content
            
            paper.pdf_content = pdf_content
            paper.save()
            
            pdf_base64 = base64.b64encode(pdf_content).decode("utf-8")
        except requests.RequestException as e:
            error_message = f"Oops, something went wrong: {str(e)}"
            context = {
                'search_error': True,
                'error_message': error_message,
            }
            return render(request, 'recommendation.html', context)
        
    context = {
        'pdf_base64': pdf_base64,
        'paper': paper,
    }
    
    return render(request, 'display.html', context)

# ----------------------------------------------------------------------Q&A--------------------------------------------------------------------------------
# ----------------------------------------------About---------------------------------------------------------------------------------------

def about(request):  
    """
    Render the 'About' page of the Django web application.
    
    Args:
        request (HttpRequest): The HTTP request made by the user.
    
    Returns:
        HttpResponse: The rendered HTML content of the 'About' page.
    """
    return render(request, 'about.html')


# -----------------------------------------------architecture---------------------------------------------------
def architecture(request):
    return render(request , 'architecture.html')

# -------------------------------------------------END---------------------------------------------------------------------------------------------------------------