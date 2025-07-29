# mall_segmentation_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Title
st.title("🛍️ Customer Segmentation Using K-Means")
st.markdown("Upload customer data and explore clusters based on spending patterns!")

# Load data
df = pd.read_csv("Mall_Customers.csv")


# Display raw data
if st.checkbox("Show raw data"):
    st.write(df)

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Choose number of clusters
k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=5)

# Model and prediction
model = KMeans(n_clusters=k, random_state=0)
df["Cluster"] = model.fit_predict(X)

# Plot
fig, ax = plt.subplots()
for i in range(k):
    cluster = df[df["Cluster"] == i]
    ax.scatter(cluster["Annual Income (k$)"], cluster["Spending Score (1-100)"], label=f"Cluster {i}")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Segments")
ax.legend()

# Display
st.pyplot(fig)
