import streamlit as st
import faiss
import pickle
import numpy as np
import requests
import pandas as pd
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# Load FAISS index and metadata
index = faiss.read_index("desc_index.faiss")
with open("desc_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="Customer Support Query Assistant", layout="wide")
st.title("Customer Support Query Assistant")

query = st.text_input("Enter your customer query:")

if query:
    query_embedding = model.encode([query])
    query_embedding = normalize(query_embedding)

    # Search FAISS for top 10 (to filter duplicates)
    distances, indices = index.search(query_embedding, 10)

    # Collect top 5 unique results
    seen_descriptions = set()
    matches = []

    for i, idx in enumerate(indices[0]):
        description, solution, category = metadata[idx]

        if description in seen_descriptions:
            continue

        sim_score = 1 - distances[0][i] ** 0.5
        matches.append({
            "Match #": len(matches) + 1,
            "Category": category.upper(),
            "Similar Query": description,
            "Stored Solution": solution,
            "Similarity Score": round(sim_score, 4)
        })

        seen_descriptions.add(description)
        if len(matches) == 5:
            break

    # Display match table
    st.subheader("Top 5 Similar Matches")
    st.dataframe(pd.DataFrame(matches), use_container_width=True)

    # Create context for LLM
    context = "\n\n".join([
        f"{i+1}. Query: {m['Similar Query']}\n   Solution: {m['Stored Solution']}"
        for i, m in enumerate(matches)
    ])

    # Refined LLM prompt
    refined_prompt = f"""
You are a customer support assistant for a company. A user has asked the following question:

"{query}"

Based on similar past cases below:
{context}

    Return a **standardized** response using the following exact format:

---
Thank you for reaching out to us. We’ll do our best to help you.

Based on your query, "{query}", here are the steps to resolve the issue:

1. Step 1
2. Step 2
3. Step 3
...

If you face any more issues, feel free to contact our customer service representative. Let me know if I can help you any further. Have a great day!
---

- Your response MUST use this exact structure.
- Do NOT include any conversational variation.
- Keep the steps clear and concise.
- Do NOT invent extra explanations.
"""


    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "tinyllama", "prompt": refined_prompt, "stream": False}
        )

        if response.ok:
            final_response = response.json()["response"]
            st.subheader("LLM-Generated Final Suggestion")
            st.markdown(final_response.strip())
        else:
            st.warning("Ollama did not respond successfully.")

    except Exception as e:
        st.error(f"Error communicating with Ollama: {e}")
