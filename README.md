# Customer Support Query Assistant
This project is a full-stack implementation of an intelligent customer support assistant. It combines semantic search using sentence embeddings, unsupervised clustering for data exploration, FAISS for efficient vector similarity search, and LLM-based response generation via Ollama. The user interface is built using Streamlit.

---

## Project Objective
The objective of this project is to streamline and support customer service operations by:
- Retrieving the top 5 most relevant historical query–solution pairs for any user-submitted query.
- Generating a standardized, context-aware response using Ollama’s TinyLLaMA, grounded in similar past cases.
- Improving consistency and accuracy of customer support through contextual understanding and semantic retrieval.

---

## Workflow Overview
### 1. Data Understanding & Preprocessing
- Conducted exploratory analysis to understand structure and key fields.
- Focused on `Category`, `Instruction` (description), and `Response` (solution).
- Preprocessing steps included:
  - Lowercasing text
  - Removing placeholders and special characters
  - Regex-based tokenization
### 2. Text Representation
- Sentence embeddings were generated using the `all-MiniLM-L6-v2` model from SentenceTransformers.
- TF-IDF was applied on cleaned descriptions and solutions to extract top keywords from each cluster.
### 3. Clustering Analysis
- Applied KMeans clustering to uncover patterns and relationships across similar queries and solutions.
- Used the elbow method to determine optimal number of clusters.
- Generated a similarity matrix to evaluate intra- and inter-cluster relationships and validate semantic consistency.
### 4. Vector Search and Indexing
- Description and solution embeddings were stored using FAISS for high-speed retrieval.
- Metadata (description, solution, category) was saved using Pickle for later use during inference.
### 5. LLM-Based Suggestion Generation
- Integrated TinyLLaMA using Ollama's local runtime.
- Once the top 5 matches are retrieved from FAISS, a standardized, LLM-generated response is produced using a structured prompt.
### 6. Streamlit Interface
- Built an interactive UI for users to input queries.
- Displays top 5 similar past queries with solutions, similarity scores, and categories.
- LLM-generated final suggestion is shown below the table as a refined, contextually appropriate reply.

---

## How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Launch the Streamlit application
streamlit run app.py

# Step 3: Ensure Ollama is running TinyLLaMA locally
ollama run tinyllama
```
### Repository Structure
```text
CUSTOMER_SUPPORT_BOT/
├── app.py                         # Streamlit frontend for user interaction
├── clustering.ipynb               # Clustering analysis with elbow method
├── vector_db.ipynb                # Embedding generation and FAISS indexing
├── EDA.ipynb                      # Exploratory Data Analysis
├── cleaned_filtered_ds.xlsx       # Cleaned dataset used in the final application
├── filtered_ds.xlsx               # Initial filtered dataset after column selection
├── cluster_similarity_matrix.xlsx # Matrix showing description-solution cluster mapping
├── customer_query(ds).csv         # Sample input query file (CSV)
├── desc_index.faiss               # FAISS index file of description embeddings
├── desc_metadata.pkl              # Metadata for FAISS (description, solution, category)
├── desc_embeddings.pkl            # Pickled Sentence-BERT embeddings of descriptions
├── soln_embeddings.pkl            # Pickled Sentence-BERT embeddings of solutions
├── requirements.txt               # All Python dependencies
└── README.md                      # Project documentation (you are here)
```

### Dataset
This project leverages the Bitext Customer Support LLM Chatbot Training Dataset, a high-quality dataset developed for customer support automation and LLM fine-tuning.
Structure: Contains fields such as category, instruction (query), and response (solution).
Size: ~26,872 query–response pairs across 27 intents and 10 major categories.
Use Case: Well-suited for information retrieval, intent classification, and RAG (retrieval-augmented generation) tasks.
Access: Available on GitHub and Hugging Face.

### Use Case & Importance of Clustering
1. Clustering was not only used for exploratory data analysis, but also for:
2. Identifying sub-clusters and similarity patterns across and within categories.
3. Extracting meaningful keywords that help define common support themes.
4. Verifying the quality and diversity of query–solution pairs semantically.
5. This unsupervised technique helped build a deeper understanding of customer issue patterns, enhancing both retrieval accuracy and the final LLM-generated output quality.

### Acknowledgements
SentenceTransformers – for semantic text embeddings.
FAISS (Facebook AI) – for scalable and efficient similarity search.
Streamlit – for building the interactive frontend.
Ollama – for enabling local LLM inference using TinyLLaMA.
Bitext Dataset – for providing a robust, structured customer support dataset.
