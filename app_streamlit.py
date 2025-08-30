import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
	page_title="Klasifikasi Tomat",
	page_icon="🍅"
)

df = pd.read_csv("dataset_tomat.csv")

model = joblib.load("model_klasifikasi_tomat.joblib")
scaler = joblib.load("scaler_klasifikasi_tomat.joblib")

st.title("🍅 Klasifikasi Tomat")
st.markdown("Aplikasi machine learning klasifikasi tomat **Ekspor, Lokal Premium** atau untuk **Industri**.")

berat = st.slider("Berat Tomat (gr)", 50, 210, 100)
kekenyalan = st.slider("Kekenyalan Tomat (N)", 3.0, 10.0, 5.5)
kadar_gula = st.slider("Kadar Gula (Bx)", 2.0, 10.0, 3.2)
tebal_kulit = st.slider("Tebal Kulit Tomat (cm)", 0.1, 1.0, 0.5)

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[berat,kekenyalan,kadar_gula,tebal_kulit]], columns=["berat","kekenyalan","kadar_gula","tebal_kulit"])
	
	st.write("**Data Baru :**")
	st.dataframe(data_baru)


	st.write("**Data Baru Scaled :**")
	data_baru_scaled = scaler.transform(data_baru)
	data_baru_scaled = pd.DataFrame(data_baru_scaled, columns=data_baru.columns)
	st.dataframe(data_baru_scaled)



	st.write("**Visualisasi Berat vs Kekenyalan :**")

	ekspor = df[df["grade"]=="Ekspor"]
	lokal_premium = df[df["grade"]=="Lokal Premium"]
	industri = df[df["grade"]=="Industri"]

	fig, ax = plt.subplots(figsize=(6,5))
	ax.scatter(ekspor["berat"], ekspor["kekenyalan"], s=100, alpha=0.7, color="red", label="Ekspor")
	ax.scatter(lokal_premium["berat"], lokal_premium["kekenyalan"], s=100, alpha=0.7, color="green", label="Lokal Premium")
	ax.scatter(industri["berat"], industri["kekenyalan"], s=100, alpha=0.7, color="blue", label="Industri")
	ax.scatter(data_baru["berat"], data_baru["kekenyalan"], s=100, alpha=0.7, color="black", marker="x", label="Data Baru")
	ax.set_xlabel("Berat")
	ax.set_ylabel("Kekenyalan")
	ax.set_title("Berat vs Kekenyalan")
	ax.legend()
	ax.grid(True, linestyle="--", alpha=0.5)
	st.pyplot(fig)

	st.write("**Kadar Gula vs Tebal Kulit :**")


	fig, ax = plt.subplots(figsize=(6,5))
	ax.scatter(ekspor["kadar_gula"], ekspor["tebal_kulit"], s=100, alpha=0.7, color="red", label="Ekspor")
	ax.scatter(lokal_premium["kadar_gula"], lokal_premium["tebal_kulit"], s=100, alpha=0.7, color="green", label="Lokal Premium")
	ax.scatter(industri["kadar_gula"], industri["tebal_kulit"], s=100, alpha=0.7, color="blue", label="Industri")
	ax.scatter(data_baru["kadar_gula"], data_baru["tebal_kulit"], s=100, alpha=0.7, color="black", marker="x", label="Data Baru")
	ax.set_xlabel("Kadar Gula")
	ax.set_ylabel("Tebal Kulit")
	ax.set_title("Kadar Gula vs Tebal Kulit")
	ax.legend()
	ax.grid(True, linestyle="--", alpha=0.5)
	st.pyplot(fig)


	prediksi = model.predict(data_baru_scaled)[0]
	presentase = max(model.predict_proba(data_baru_scaled)[0])
	st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat dengan penuh 🍅 oleh Adi Setiawan")