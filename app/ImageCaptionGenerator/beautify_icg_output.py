from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import pickle
import torch
from sentence_transformers import SentenceTransformer

# loading embeddings
f = open('./image_caption_generation/quote_embeddings.pckl', 'rb')
embeddings = pickle.load(f)
f.close()

# loading transoformer
st_model = SentenceTransformer('all-MiniLM-L6-v2')

import json
f = open('./image_caption_generation/quotes.json','rb')
quotes_data = json.load(f)
quoteslist =[]
authors =[]
for quote in quotes_data:
  quoteslist.append(quote['Quote'])
  authors.append(quote['Author'])

def cosine_similarity(element, elements_list):
    similarities = []
    for elem in elements_list:
        dot_product = np.dot(element, elem)
        element_norm = np.linalg.norm(element)
        elem_norm = np.linalg.norm(elem)
        cos_similarity = dot_product / (element_norm * elem_norm)
        similarities.append(cos_similarity)
    return similarities

def get_suggestions(text):
  device = torch.device("cpu")
  input_embedding = st_model.encode(text,convert_to_tensor=True).to(device).numpy()
  sim_scores = cosine_similarity(input_embedding,embeddings)
  df = pd.DataFrame({'quote': quoteslist, 'author': authors, 'similarity scores': sim_scores})
  df.drop_duplicates(subset='quote', inplace=True)
  df_sorted = df.sort_values(by='similarity scores', ascending=False)
  suggestions = []
  top_k =5
  for i in range(top_k):
    suggestions.append((df_sorted.iloc[i].quote,df_sorted.iloc[i].author))
  return suggestions