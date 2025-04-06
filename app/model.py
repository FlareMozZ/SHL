# -*- coding: utf-8 -*-
import os
import pandas as pd
import json
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def process_row(row):
    """Process a CSV row into document format for vector storage"""
    # Handle downloads column parsing
    downloads = []
    if pd.notna(row.get('downloads', '')):
        try:
            # Try parsing as JSON first
            downloads = json.loads(row['downloads'].replace("'", '"'))
        except json.JSONDecodeError:
            try:
                # Fallback to comma-separated list
                downloads = [item.strip() for item in row['downloads'].split(',')]
            except:
                downloads = []

    # Extract completion time from assessment_length
    completion_time = row.get('assessment_length', '')
    if isinstance(completion_time, str) and "=" in completion_time:
        completion_time = completion_time.split("=")[-1].strip()

    return {
        "id": row.name,
        "content": f"""
        Product Name: {row.get('name', '')}
        Description: {row.get('description', '')}
        Job Levels: {row.get('job_levels', '')}
        Languages: {row.get('languages', '')}
        Assessment Length: {row.get('assessment_length', '')}
        Test Types: {row.get('test_types', '')}
        Downloads: {downloads}
        """,
        "metadata": {
            "url": row.get('url', ''),
            "remote_testing": row.get('remote_testing', 'No'),
            "completion_time": completion_time,
            "adaptive_support": row.get('adaptive_support', 'No'),
            "test_type": row.get('test_type', ''),
            "duration": completion_time  # Map completion_time to duration
        }
    }

def initialize_components():
    """Initialize all system components"""
    # Load and process data
    df = pd.read_csv("../data/data.csv")
    documents = [process_row(row) for _, row in df.iterrows()]

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Load vector store
    vector_store = FAISS.load_local(
        "../data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Initialize Gemini LLM
    gemini = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",

# Define the system instruction separately
system_instruction = """You are an AI assistant specialized in recommending SHL assessments based on job descriptions and queries. Your task is to analyze the query and recommend the most relevant SHL assessments from our database.

INSTRUCTIONS:
1. Carefully analyze the user's query, which may be a natural language question or job description
2. Consider key factors like:
   - Required technical skills (programming languages, tools)
   - Job level (entry, mid, senior)
   - Desired soft skills or personality traits
   - Time constraints for assessment completion
   - Any specific assessment types mentioned

3. Return EXACTLY 10 most relevant SHL assessments that match the requirements
4. Format your response as a structured JSON with this exact schema:

{
  "recommendations": [
    {
      "assessment_name": "Name of the assessment",
      "url": "Direct URL to the assessment",
      "remote_testing": "Yes/No",
      "adaptive_support": "Yes/No",
      "duration": "Duration in minutes",
      "test_type": "Type of test (e.g., Cognitive, Technical)",
      "relevance_score": Numeric score between 0-1,
      "relevance_explanation": "Brief explanation of why this assessment matches the query"
    },
  ]
}

5. IMPORTANT RULES:
   - ONLY return assessments that match time constraints if specified
   - Sort recommendations by relevance score in descending order
   - Provide succinct but informative relevance explanations
   - Always return valid JSON that can be parsed programmatically
   - Do not include any text outside the JSON object
   - Ensure all required fields are included for each recommendation

The quality of your recommendations will be evaluated using Mean Recall@3 and MAP@3 metrics.""",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Configure retriever
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 10,
            "score_threshold": 0.7
        }
    )

    return RetrievalQA.from_chain_type(
        llm=gemini,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"verbose": True}
    )

# Initialize QA system when module loads
try:
    qa = initialize_components()
except Exception as e:
    raise RuntimeError(f"Failed to initialize QA system: {str(e)}")