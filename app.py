import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
file_path_1 = r'D:\FREELANCE_PROJECTS\product-recommendation\data\All Electronics.csv'
file_path_2 = r'D:\FREELANCE_PROJECTS\product-recommendation\data\Air Conditioners (1).csv'

all_electronics_df = pd.read_csv(file_path_1)
air_conditioners_df = pd.read_csv(file_path_2)

combined_df = pd.concat([all_electronics_df, air_conditioners_df], ignore_index=True)
combined_df.fillna("", inplace=True)
combined_df['text_features'] = combined_df['name'] + " " + combined_df['sub_category']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_df['text_features'])
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_products(product_name, top_n=5):
    idx = combined_df[combined_df['name'].str.contains(product_name, case=False, na=False)].index
    if idx.empty:
        return "Product not found."
    
    sim_scores = list(enumerate(cosine_sim_matrix[idx[0]]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommended_indices = [i[0] for i in sim_scores[1:top_n + 1]]
    
    return combined_df.iloc[recommended_indices][['name', 'sub_category', 'discount_price']]

st.title("Product Recommendation System")

product_name = st.text_input("Enter Product Name:")

if st.button("Recommend"):
    if product_name:
        recommendations = recommend_products(product_name)
        if isinstance(recommendations, str):
            st.write(recommendations)
        else:
            st.write("### Recommended Products:")
            for _, row in recommendations.iterrows():
                st.write(f"- **{row['name']}** ({row['sub_category']}), Price: {row['discount_price']}")
    else:
        st.write("Please enter a product name.")
