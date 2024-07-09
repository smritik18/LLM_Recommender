import streamlit as st
import os
from llama_cpp import Llama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from transformers import pipeline
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

llm = Llama(model_path='/Users/smritikumari/Desktop/llm/llama/llama-2-7b-chat/llama-2-7b-chat-7B-F4.gguf', n_ctx=2048, n_gpu_layers=1)

# Load precomputed data
encoded_titles = np.load('book_title_embeddings.npy')
rating_matrix = load_npz('rating_matrix.npz')
books_df = pd.read_csv('data/Books.csv')
users_df = pd.read_csv('data/Users.csv')

# Load precomputed mappings and user similarities
user_mapping = pd.read_csv('user_mapping.csv').set_index('User-ID')['Index'].to_dict()
book_mapping = pd.read_csv('book_mapping.csv').set_index('ISBN')['Index'].to_dict()
# user_similarities = np.load('user_similarities.npy')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to retrieve books based on user query
def get_candidate_books(query, encoded_summaries, book_summaries, top_k=1):
    query_embedding = embedding_model.encode([query])[0]
    query_embedding = query_embedding.reshape(1, -1)  
    similarities = cosine_similarity(query_embedding, encoded_summaries)
    similarities = similarities.flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [book_summaries[i] for i in top_indices]

def extract_numbers(s):
    numbers = re.findall(r'\d+', s)
    if numbers:
        return int(numbers[0])
    else:
        return None

def extract_user_id(user_input):
    system_prompt = "Analyze user input and extract User ID from it. \
    - If the user query contains User ID, respond with the User ID \
    - otherwise respond with 'None'"

    response = client.messages.create(model="claude-3-opus-20240229", max_tokens=100, system=system_prompt,
                                      messages=[{"role": "user", "content": user_input}]).content[0].text
    return extract_numbers(response)

def classify_intent(user_input):
    system_prompt = "You are a helper to a chatbot. You are capable of analyzing user queries and determining whether they are asking for recommendation about books or just general information. \
    Analyze the User input and perform Intent Classification as below: \
    Intent Classification: \
    - Is the user seeking a recommendation (e.g., suggest, recommend)? If so, respond with 'recommendation' \
    - Or are they asking for general information?. If so, respond with 'general'"

    response = client.messages.create(model="claude-3-opus-20240229", max_tokens=100, system=system_prompt,
                                      messages=[{"role": "user", "content": user_input}]).content[0].text
    if any(keyword in response.lower() for keyword in ['recommend', 'suggest', 'advice']):
        return "recommendation"
    else:
        return "general"

def collaborative_filtering_recommendation(user_id, candidate_books, rating_matrix, books_df, top_n=3):
    if user_id not in user_mapping:
        print(f"User ID {user_id} is not present in the dataset.")
        return []

    user_index = user_mapping[user_id]
    user_ratings = rating_matrix[user_index]

    candidate_indices = [book_mapping[isbn] for isbn in books_df[books_df['Book-Title'].isin(candidate_books)]['ISBN']]
    unrated_books = [book_index for book_index in candidate_indices if user_ratings[0, book_index] == 0]

    top_books = []
    for book_index in unrated_books:
        book_ratings = rating_matrix[:, book_index].toarray().flatten()
        average_rating = np.mean(book_ratings)
        top_books.append((book_index, average_rating))

    top_books.sort(key=lambda x: x[1], reverse=True)
    top_n_books = [books_df.iloc[books_df.index[books_df['ISBN'] == list(book_mapping.keys())[book_index]]]['ISBN'].values[0] for book_index, _ in top_books[:top_n]]


    return top_n_books




# Example function to parse user query and recommend books
def retrieve_recs(candidate_books, user_id):
    
    recommended_books = collaborative_filtering_recommendation(user_id, candidate_books, rating_matrix, books_df)
    
    # Display recommended books
    books = []
    print("Recommended Books:")
    for isbn in recommended_books:
        books.append(books_df.loc[books_df['ISBN'] == isbn, 'Book-Title'].iloc[0])

    return books

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue =  """
        You are a well-read and friendly book advisor chatbot. You have vast knowledge about books across all genres.
        Respond to the users about their questions on books. Be brief. 
        Respond only to user queries. Do not automatically create user queries. 
        """
    print('RETRIEVED BOOKS: ', retrieved_books)

    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    prompt = f"{string_dialogue} {prompt_input} Assistant: "        
    output = llm(prompt = prompt, temperature = 0.3, max_tokens=1000, stop=["User:"])['choices'][0]['text']
    print('Output: ', output)
    return output

def generate_rec_response(retrieved_books):     
    output = f"Based on your interest, here are some book recommendations:\n{retrieved_books}"
    print('Output: ', output)
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=False):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    intent = classify_intent(prompt)
    print('INTENT :', intent)
    if "recommendation" in intent:
        #get user id
        user_id = extract_user_id(prompt)
        if user_id is None:
            retrieved_books = "None"
        else:
            candidate_books = get_candidate_books(prompt, encoded_titles, books_df['Book-Title'].tolist(), top_k=5)
            retrieved_books = retrieve_recs(candidate_books, user_id)
    else: 
        retrieved_books = "None"

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if (retrieved_books != "None") and (len(retrieved_books)>1):
                response = generate_rec_response(retrieved_books)
            else:
                response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
