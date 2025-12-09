import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons
import plotly.express as px

# --- GLOBAL UI STYLING ---
st.set_page_config(page_title="Premium Clustering App", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #0f172a, #1e293b);
    color: white;
}
.card {
    padding: 20px;
    background-color: #1e293b;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
    margin-bottom: 20px;
}
div.stButton > button {
    background-color: #3b82f6;
    color: white;
    padding: 10px 20px;
    border-radius: 10px;
    border: none;
}
div.stButton > button:hover {
    background-color: #60a5fa;
}
/* Tab Styling */
div[data-baseweb="tab-list"] {
    background-color: #1e293b !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
}
div[data-baseweb="tab"] {
    color: white !important;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background-color: #3b82f6 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<h1 style="text-align:center; color:#fff; font-weight:700;">
    Premium Clustering App
</h1>
<p style="text-align:center; color:#cbd5e1;">
    KMeans Prediction | DBSCAN | Outlier Detection | 2D & 3D Visualization
</p>
""", unsafe_allow_html=True)

# --- TAB SETUP ---
tab1, tab2, tab3 = st.tabs([
    "KMeans (Blobs + Outlier)", 
    "DBSCAN (Moons)", 
    "Informasi Model"
])

# ==============================================================================
# TAB 1: KMEANS MAKE BLOBS
# ==============================================================================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("K-Means Clustering - Make Blobs")
    
    st.info("**Informasi Praktikan**")
    st.markdown("""
    **Rentang nilai dataset Make Blobs:**
    * Feature 1: -2.5 s.d 8.0
    * Feature 2: -5.0 s.d 7.0
    
    *Namun rentang fitur bukan batas cluster!*
    **Outlier ditentukan oleh jarak terhadap centroid**
    
    **Custom Outlier Detection:**
    Jika jarak titik baru > (mean jarak cluster + 2 x std) -> **OUTLIER**
    Visualisasi 2D menampilkan lingkaran radius threshold untuk memudahkan analisis.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Dataset Generation ---
    X_blobs, labels_blobs = make_blobs(
        n_samples=150,
        n_features=2,
        centers=3,
        cluster_std=0.5,
        shuffle=True,
        random_state=0
    )

    # --- Clustering Model ---
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X_blobs)
    centers = kmeans.cluster_centers_

    col1, col2 = st.columns([1, 2.5])

    # --- PREDICT PANEL (Left Column) ---
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Predict Cluster")
        
        f1 = st.number_input("Feature 1 (Blobs)", value=0.0, key="b1")
        f2 = st.number_input("Feature 2 (Blobs)", value=0.0, key="b2")
        
        new_point = np.array([[f1, f2]])
        new_label = None
        is_outlier = False
        
        if st.button("Predict (KMeans)", key="predict_blobs"):
            # 1. Predict Label dasar
            new_label = int(kmeans.predict(new_point)[0])
            centroid = centers[new_label]
            
            # 2. Hitung statistik jarak dalam cluster tersebut untuk threshold
            cluster_pts = X_blobs[labels_blobs == new_label]
            distances = np.linalg.norm(cluster_pts - centroid, axis=1)
            threshold = distances.mean() + (2 * distances.std())
            
            # 3. Hitung jarak titik baru ke centroid
            new_dist = np.linalg.norm(new_point - centroid)
            
            # 4. Cek Outlier
            if new_dist > threshold:
                is_outlier = True
                st.error("OUTLIER - Terlalu jauh dari centroid!")
            else:
                st.success(f"Cluster {new_label}")
                st.info(f"Jarak: {new_dist:.3f} | Threshold: {threshold:.3f}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- VISUALIZATION PANEL (Right Column) ---
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("2D Visualization - KMeans")
        
        # Plot dasar dataset
        fig2d = px.scatter(
            x=X_blobs[:, 0],
            y=X_blobs[:, 1],
            color=labels_blobs.astype(str),
            template="plotly_dark",
            title="KMeans Clustering (Make Blobs)"
        )
        
        # Tambah Centroids
        fig2d.add_scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker=dict(size=12, color="white", symbol="x"),
            name="Centroids"
        )
        
        # Tambah Titik Prediksi (jika ada)
        if new_label is not None:
            fig2d.add_scatter(
                x=[f1], y=[f2],
                mode="markers",
                marker=dict(size=22, color="red"),
                name="Predicted Point"
            )
            
            # Gambar Radius Threshold (Lingkaran)
            # Karena plotly shape circle butuh bounding box (x0,y0, x1,y1)
            centroid = centers[new_label]
            # Kita perlu hitung threshold lagi (karena variabel scope) atau ambil dari atas
            cluster_pts = X_blobs[labels_blobs == new_label]
            distances = np.linalg.norm(cluster_pts - centroid, axis=1)
            threshold = distances.mean() + (2 * distances.std())
            
            fig2d.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=centroid[0] - threshold,
                y0=centroid[1] - threshold,
                x1=centroid[0] + threshold,
                y1=centroid[1] + threshold,
                line=dict(color="yellow", width=2, dash="dash"),
            )
            
        st.plotly_chart(fig2d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 3D Visualization
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("3D Visualization - KMeans")
        fig3d = px.scatter_3d(
            x=X_blobs[:, 0],
            y=X_blobs[:, 1],
            z=np.zeros(len(X_blobs)), # Dummy Z axis
            color=labels_blobs.astype(str),
            template="plotly_dark",
            title="3D KMeans Clustering"
        )
        
        if new_label is not None:
             fig3d.add_scatter3d(
                x=[f1], y=[f2], z=[0],
                mode="markers",
                marker=dict(size=12, color="red"),
                name="New Point"
            )
            
        st.plotly_chart(fig3d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 2: DBSCAN MAKE MOONS
# ==============================================================================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("DBSCAN Clustering - Make Moons")
    
    st.info("**Informasi Praktikan**")
    st.markdown("""
    **Rentang dataset Make Moons:**
    * Feature 1: -1.5 s.d 2.5
    * Feature 2: -0.8 s.d 1.5
    
    **DBSCAN otomatis memberi label -1 untuk OUTLIER.**
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Dataset Generation ---
    X_moons, _ = make_moons(n_samples=300, noise=0.07, random_state=0)
    
    # --- Model ---
    db = DBSCAN(eps=0.2, min_samples=5)
    labels_moons = db.fit_predict(X_moons)

    colA, colB = st.columns([1, 2])

    # --- PREDICT PANEL (Left) ---
    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Predict (DBSCAN)")
        
        m1 = st.number_input("Feature 1 (Moons)", value=0.0, key="m1")
        m2 = st.number_input("Feature 2 (Moons)", value=0.0, key="m2")
        new_moon = np.array([[m1, m2]])
        moon_label = None
        
        if st.button("Predict (DBSCAN)", key="predict_moon"):
            # Trik DBSCAN untuk "predict": Masukkan titik baru ke dataset dan fit ulang
            # atau hitung manual neighbors. Di sini kita pakai cara vstack (re-fit).
            stacked = np.vstack([X_moons, new_moon])
            temp_labels = db.fit_predict(stacked)
            moon_label = temp_labels[-1] # Ambil label data terakhir (si new_moon)
            
            if moon_label == -1:
                st.error("OUTLIER / NOISE")
            else:
                st.success(f"Cluster {moon_label}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- VISUALIZATION PANEL (Right) ---
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("2D Visualization - DBSCAN")
        
        fig2_moon = px.scatter(
            x=X_moons[:, 0],
            y=X_moons[:, 1],
            color=labels_moons.astype(str),
            template="plotly_dark",
            title="DBSCAN Clustering (Make Moons)"
        )
        
        if moon_label is not None:
             fig2_moon.add_scatter(
                x=[m1], y=[m2],
                mode="markers",
                marker=dict(size=20, color="red"),
                name="New Point"
            )
            
        st.plotly_chart(fig2_moon, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 3D Visualization
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("3D Visualization - DBSCAN")
        
        fig3_moon = px.scatter_3d(
             x=X_moons[:, 0],
             y=X_moons[:, 1],
             z=np.zeros(len(X_moons)),
             color=labels_moons.astype(str),
             template="plotly_dark",
             title="3D DBSCAN Clustering"
        )
        
        if moon_label is not None:
             fig3_moon.add_scatter3d(
                x=[m1], y=[m2], z=[0],
                mode="markers",
                marker=dict(size=12, color="red"),
                name="New Point"
            )
        
        st.plotly_chart(fig3_moon, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 3: INFORMASI MODEL
# ==============================================================================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Informasi Model, Dataset, dan Pickle")
    
    st.markdown("""
    ### Mengapa Tidak Menggunakan Pickle?
    Dataset sintetis (`make_blobs` dan `make_moons`) dibuat ulang setiap run (kecuali random_state dikunci permanent di luar, tapi secara konsep web app dinamis).
    
    Pickle hanya dipakai jika:
    1. Dataset tetap & konsisten.
    2. Model dilatih di luar Streamlit (misal di Notebook) dan disimpan.
    
    ### Jika Ingin Menggunakan Model di Dataset Asli:
    1. Ambil dataset asli (Kaggle/CSV).
    2. Lakukan EDA untuk melihat pola data (bulat, memanjang, outlier?).
    3. Pilih algoritma:
        * **KMeans**: Cluster bulat / sederhana.
        * **DBSCAN**: Bentuk tidak beraturan / banyak noise.
        * **Hierarchical**: Ingin struktur cluster bertingkat.
    4. Latih model -> Simpan pickle.
    5. Load pickle di Streamlit.
    
    ### Rule of Thumb
    | Pola Data | Algoritma |
    | :--- | :--- |
    | Bulat/rapi | KMeans |
    | Melengkung | DBSCAN |
    | Bertingkat | Hierarchical |
    | Banyak noise | DBSCAN |
    | Tidak tahu jumlah cluster | DBSCAN / Hierarchical |
    
    ### Inti Pembelajaran
    * **Dataset sintetis**: Latih langsung dalam app agar interaktif.
    * **Dataset asli**: Boleh pakai pickle untuk efisiensi.
    * **Model terbaik**: Bergantung bentuk data (*No Free Lunch Theorem*).
    """)
    st.markdown('</div>', unsafe_allow_html=True)