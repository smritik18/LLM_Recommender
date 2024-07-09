import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Books.csv
books_df = pd.read_csv('data/Books.csv')

# Extract book titles
book_titles = books_df['Book-Title'].tolist()

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode book titles
encoded_titles = embedding_model.encode(book_titles)

# Save embeddings to file
np.save('book_title_embeddings.npy', encoded_titles)
