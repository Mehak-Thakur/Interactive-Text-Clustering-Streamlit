import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from utils import preprocess_text

st.title("📊 Interactive Text Clustering App")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    text_data = uploaded_file.read().decode("utf-8").split("\n")
    processed_text = [preprocess_text(doc) for doc in text_data if doc.strip() != ""]

    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(processed_text)

    algo = st.selectbox("Choose Clustering Algorithm", ["K-Means", "DBSCAN"])

    if algo == "K-Means":
        k = st.slider("Number of clusters (K)", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)

    else:
        eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5)
        model = DBSCAN(eps=eps, min_samples=2, metric='cosine')
        labels = model.fit_predict(X)

    df = pd.DataFrame({"Text": text_data[:len(labels)], "Cluster": labels})

    st.subheader("📌 Clustered Text Output")
    st.dataframe(df)

    # Visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.title("Text Clustering Visualization")
    st.pyplot(plt)
