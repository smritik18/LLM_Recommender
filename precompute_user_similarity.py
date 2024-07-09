import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load users data
users_df = pd.read_csv('data/Users.csv')

# Preprocess user data
# Handle missing values if any
users_df['Location'].fillna('', inplace=True)  # Replace NaN with empty string for location

# Scale age for similarity calculation
scaler = MinMaxScaler()
users_df['Age_Scaled'] = scaler.fit_transform(users_df[['Age']])

# Function to compute user similarity based on location and scaled age
def compute_user_similarity(users_df):
    # Extract relevant attributes for similarity calculation
    user_attributes = users_df[['Location', 'Age_Scaled']].copy()
    user_attributes.set_index(users_df['User-ID'], inplace=True)
    
    # Calculate similarity based on location (categorical comparison) and scaled age
    location_similarity = cosine_similarity(user_attributes['Location'].str.get_dummies(), dense_output=False)
    age_similarity = cosine_similarity(user_attributes[['Age_Scaled']], dense_output=False)
    
    # Combine location and age similarity
    total_similarity = location_similarity + age_similarity
    
    # Set diagonal to zero (self-similarity)
    np.fill_diagonal(total_similarity, 0)
    
    return total_similarity

# Compute user similarity matrix
user_similarity_matrix = compute_user_similarity(users_df)

# Example to save user similarity matrix to file (you can choose appropriate format)
np.save('user_similarity_matrix.npy', user_similarity_matrix)
