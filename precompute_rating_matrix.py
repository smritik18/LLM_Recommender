# import pandas as pd
# from scipy.sparse import csr_matrix, save_npz

# # Load Ratings.csv
# ratings_df = pd.read_csv('data/Ratings.csv')

# # Function to create sparse rating matrix
# def create_sparse_rating_matrix(ratings_df):
#     # Assume ratings_df has columns ['User-ID', 'ISBN', 'Book-Rating']
#     user_mapping = {user_id: idx for idx, user_id in enumerate(ratings_df['User-ID'].unique())}
#     book_mapping = {isbn: idx for idx, isbn in enumerate(ratings_df['ISBN'].unique())}
    
#     row_indices = ratings_df['User-ID'].map(user_mapping)
#     col_indices = ratings_df['ISBN'].map(book_mapping)
    
#     rating_values = ratings_df['Book-Rating'].tolist()
    
#     rating_matrix = csr_matrix((rating_values, (row_indices, col_indices)), 
#                                shape=(len(user_mapping), len(book_mapping)))
    
#     return rating_matrix

# # Create and save sparse rating matrix
# rating_matrix = create_sparse_rating_matrix(ratings_df)
# save_npz('rating_matrix.npz', rating_matrix)


import pandas as pd
from scipy.sparse import csr_matrix, save_npz

# Load Ratings.csv
ratings_df = pd.read_csv('data/Ratings.csv')

# Function to create sparse rating matrix
def create_sparse_rating_matrix(ratings_df):
    user_mapping = {user_id: idx for idx, user_id in enumerate(ratings_df['User-ID'].unique())}
    book_mapping = {isbn: idx for idx, isbn in enumerate(ratings_df['ISBN'].unique())}
    
    row_indices = ratings_df['User-ID'].map(user_mapping)
    col_indices = ratings_df['ISBN'].map(book_mapping)
    
    rating_values = ratings_df['Book-Rating'].tolist()
    
    rating_matrix = csr_matrix((rating_values, (row_indices, col_indices)), 
                               shape=(len(user_mapping), len(book_mapping)))
    
    return rating_matrix, user_mapping, book_mapping

# Create and save sparse rating matrix and mappings
rating_matrix, user_mapping, book_mapping = create_sparse_rating_matrix(ratings_df)
save_npz('rating_matrix.npz', rating_matrix)

# Save mappings to disk for later use
pd.DataFrame(list(user_mapping.items()), columns=['User-ID', 'Index']).to_csv('user_mapping.csv', index=False)
pd.DataFrame(list(book_mapping.items()), columns=['ISBN', 'Index']).to_csv('book_mapping.csv', index=False)