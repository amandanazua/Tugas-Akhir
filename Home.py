import streamlit as st

st.set_page_config(
    page_title="Aplikasi Analisis Reaktor Nuklir",
    page_icon="⚛️",
    layout="wide" # Use "wide" layout for more space
)

st.sidebar.title("Navigasi Aplikasi")
st.sidebar.markdown("""
Pilih halaman dari daftar di bawah ini:
""")

st.title("Selamat Datang di Aplikasi Analisis Reaktor Nuklir")
st.write("""
Aplikasi ini menyediakan tiga fungsi utama:
1.  **Analisis Data Eksploratif (EDA) & Pelatihan Model**: Latih ulang model Deep Learning Anda dan lihat visualisasi data.
2.  **Prediksi Suhu TH-OUT**: Gunakan model yang sudah dilatih untuk memprediksi suhu TH-OUT berdasarkan input TH-IN.
3.  **Real-time Monitoring**: Pantau suhu TH-IN dan TH-OUT secara real-time dengan grafik interaktif.""")

st.info("Pilih halaman dari sidebar di sebelah kiri untuk memulai.")

st.markdown("---")
st.caption("Tugas Akhir Amanda Najwa Perak Azizah - 2110511158")