------Öğrenci Başarı Tahmin ve Danışmanlık Sistemi-------
Bu proje, ham veriden başlayarak öğrencilerin akademik başarılarını (geçme/kalma durumlarını) tahmin eden ve velilere/öğretmenlere yönelik stratejik tavsiyeler üreten bir Makine Öğrenmesi (ML) Karar Destek Sistemidir.

-Projenin Amacı
Eğitimcilerin, sınavlar sonuçlanmadan önce risk altındaki öğrencileri tespit etmesini sağlamak ve sadece not odaklı değil, sosyal faktörleri de (devamsızlık, çalışma süresi vb.) içeren bir analiz sunmaktır.

-Öne Çıkan Özellikler

-İkili Model Yapısı: Hem ara sınav notlarını içeren Akademik Model hem de notlar olmadan sadece yaşam tarzı verilerine odaklanan Sosyal Model ile karşılaştırmalı analiz sunar.
-Veli Danışmanlık Sistemi: Modelin teknik tahminlerini, velilerin anlayabileceği somut sözel raporlara ve tavsiyelere dönüştürür.

-İnteraktif Arayüz: Streamlit kütüphanesi kullanılarak geliştirilmiş, anlık veri girişi ve tahmin imkanı sunan web tabanlı panel.

Kullanılan Teknolojiler
-Python 3.x 
-Pandas & NumPy: Veri işleme ve ön işleme 
-Scikit-Learn: ML modelleme (Random Forest, Logistic Regression) 
-Matplotlib & Seaborn: Veri görselleştirme ve EDA 
-Streamlit: Web arayüzü geliştirme 

-Model Performansı

Akademik Model: Sınav notları dahil edildiğinde yüksek doğruluk oranı (Yaklaşık %92).
Sosyal Model: Sadece yaşam tarzı verileriyle erken uyarı imkanı (Yaklaşık %75).

-Dosya Yapısı  :
student-performance-ml/,
data/student.csv  # Ham veri seti [cite: 3, 11],
veri_analizi.py   # Streamlit arayüzü ve model kodları [cite: 11],
notebook.ipynb    # Model geliştirme ve EDA çalışmaları [cite: 11],
README.md         # Proje dökümantasyonu [cite: 11].


-Kurulum ve Çalıştırma/
Gerekli kütüphaneleri yükleyin:

pip install pandas scikit-learn streamlit 

Uygulamayı başlatın:
streamlit run veri_analizi.py
