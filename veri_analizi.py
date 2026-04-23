import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- 1. VERİ YÜKLEME VE ÖN İŞLEME ---
# Veri setini yükle (Dosya adının doğruluğundan emin ol) [cite: 259]
df = pd.read_csv("C:/Python/ML/student_data.csv") 

# Metin verilerini (GP, MS, M, F vb.) sayısal 0 ve 1'lere dönüştür [cite: 235, 260]
df_encoded = pd.get_dummies(df, drop_first=True)

# Hedef Değişken: 10 ve üzeri Geçti (1), altı Kaldı (0) [cite: 47, 89, 118]
y = (df["G3"] >= 10).astype(int)

# --- 2. MODELLERİN EĞİTİLMESİ ---

# A. Akademik Model (G1, G2 sınav notları dahil) [cite: 165, 177]
X_akademik = df_encoded.drop("G3", axis=1)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_akademik, y, test_size=0.2, random_state=42)
model_akademik = RandomForestClassifier(random_state=42).fit(X_train_a, y_train_a)

# B. Sosyal Model (G1 ve G2 notları hariç) [cite: 167, 178]
X_sosyal = X_akademik.drop(["G1", "G2"], axis=1)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sosyal, y, test_size=0.2, random_state=42)
model_sosyal = RandomForestClassifier(random_state=42).fit(X_train_s, y_train_s)

# C. Lite Model (Arayüzdeki 3 ana faktör için özel model) [cite: 200, 262]
X_lite = df[['absences', 'studytime', 'failures']]
model_lite = RandomForestClassifier(random_state=42).fit(X_lite, y)

# --- 3. VELİ DANIŞMANLIK SİSTEMİ (SÖZEL RAPOR) ---
def veli_raporu_olustur(tahmin, veri):
    """Tahmin sonucuna ve öğrenci verisine göre sözel rapor üretir [cite: 159, 272]"""
    if tahmin == 0:
        mesaj = "### ⚠️ ÖĞRENCİ ANALİZ RAPORU (RİSKLİ)\n"
        mesaj += "Öğrencinin bu dersten kalma riski yüksek görünüyor. İşte temel nedenler ve öneriler:\n"
        
        if veri['absences'] > 10:
            mesaj += f"- **Devamsızlık Sorunu:** Öğrenci {veri['absences']} gün devamsızlık yapmış. Okula devamlılığın artırılması başarıyı doğrudan yükseltecektir. [cite: 274]\n"
        
        if veri['studytime'] < 2:
            mesaj += "- **Çalışma Süresi Yetersizliği:** Haftalık çalışma süresi çok düşük. Evde günlük etüt planı oluşturulması önerilir. [cite: 276]\n"
            
        if veri['failures'] > 0:
            mesaj += f"- **Temel Eksiklikler:** Öğrencinin geçmişte {veri['failures']} başarısız dersi var. Geçmiş konular için birebir destek alınması riskin azaltılmasında kritik rol oynar. [cite: 278]\n"
        
        return mesaj
    else:
        return "### ✅ ÖĞRENCİ ANALİZ RAPORU (BAŞARILI)\nÖğrencinin mevcut çalışma düzeni ve sosyal alışkanlıkları başarıyı desteklemektedir. Bu disiplinin korunması tavsiye edilir. [cite: 279]"

# --- 4. STREAMLIT ARAYÜZÜ ---
st.set_page_config(page_title="Öğrenci Başarı Analizi", page_icon="🎓")
st.title("🎓 Öğrenci Başarı Tahmin ve Danışmanlık Sistemi")

# Model Başarı Skorlarını Göster [cite: 203, 281]
col1, col2 = st.columns(2)
col1.metric("Akademik Model Gücü", f"%{model_akademik.score(X_test_a, y_test_a)*100:.1f}")
col2.metric("Sosyal Model Gücü", f"%{model_sosyal.score(X_test_s, y_test_s)*100:.1f}")

st.divider()

# Veri Giriş Alanı [cite: 182, 190]
st.subheader("📊 Veli/Öğretmen Veri Giriş Paneli")
absences = st.slider("Öğrencinin Devamsızlık Sayısı", 0, 30, 5)
studytime = st.selectbox("Haftalık Çalışma Süresi (Kategori)", options=[1, 2, 3, 4], 
                         help="1: <2 saat, 2: 2-5 saat, 3: 5-10 saat, 4: >10 saat")
failures = st.slider("Geçmiş Başarısız Ders Sayısı", 0, 4, 0)

if st.button("🔍 Öğrenci Durumunu Analiz Et"):
    # Lite model ile tahmin yap [cite: 224, 247]
    input_data = [[absences, studytime, failures]]
    tahmin_sonucu = model_lite.predict(input_data)[0]
    
    # Sözel raporu oluştur ve göster [cite: 281]
    rapor = veli_raporu_olustur(tahmin_sonucu, {'absences': absences, 'studytime': studytime, 'failures': failures})
    
    if tahmin_sonucu == 0:
        st.error(rapor)
    else:
        st.success(rapor)