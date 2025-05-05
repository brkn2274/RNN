# YZM304 Derin Öğrenme Dersi - III. Proje Ödevi: RNN ile Duygu Analizi

**Ad Soyad:** [Çağan Barkın Üstüner]
**Numara:** [22290508]
**Teslim Tarihi:** 07.05.2025

## Giriş (Introduction)

Bu proje, Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği Bölümü YZM304 Derin Öğrenme dersi kapsamında gerçekleştirilmiştir[cite: 1]. Projenin amacı, tekrarlayan sinir ağları (RNN) kullanarak metin verileri üzerinden duygu analizi yapmaktır[cite: 2]. Bu çalışmada, basit metin ifadelerinin olumlu (True) veya olumsuz (False) olarak sınıflandırılması hedeflenmiştir.

Projede iki farklı RNN modeli geliştirilmiş ve karşılaştırılmıştır:
1.  Temel NumPy kütüphanesi kullanılarak sıfırdan implemente edilen bir RNN modeli[cite: 5].
2.  Popüler bir derin öğrenme kütüphanesi (TensorFlow/Keras) kullanılarak implemente edilen bir RNN modeli[cite: 6].

Veri seti olarak, ödev tanımında sağlanan `data.py` içerisindeki basit İngilizce ifadeler ve etiketleri kullanılmıştır[cite: 4]. Çalışmanın temel motivasyonu, RNN mimarisinin temellerini anlamak ve sıfırdan implementasyon ile hazır kütüphane kullanımının avantaj ve dezavantajlarını karşılaştırmaktır.

## Yöntem (Method)

### Veri Seti ve Önişleme

* **Veri Seti:** Çalışmada `data.py` dosyasında tanımlanan `train_data` ve `test_data` sözlükleri kullanılmıştır[cite: 4]. Bu veri seti, kısa metin ifadelerini (key) ve bunlara karşılık gelen ikili duygu etiketlerini (value: True/olumlu, False/olumsuz) içermektedir.
* **Önişleme:**
    * **Sözlük Oluşturma (Vocabulary):** Eğitim ve test verisindeki tüm benzersiz kelimelerden bir sözlük oluşturulmuştur.
    * **Sayısal Temsil (Sequencing):** Her bir metin ifadesi, sözlükteki kelimelerin indekslerinden oluşan bir diziye dönüştürülmüştür.
    * **Dolgu (Padding):** Kütüphane tabanlı RNN modelinin (TensorFlow/Keras) sabit uzunlukta girdi beklemesi nedeniyle, tüm diziler aynı uzunluğa (veri setindeki en uzun dizinin uzunluğu) getirilmek üzere sonlarına 0 değeri eklenerek (post-padding) doldurulmuştur. Sıfırdan implemente edilen model için padding zorunlu olmasa da, karşılaştırma tutarlılığı açısından benzer bir yaklaşım benimsenebilir veya model değişken uzunluktaki dizileri işleyebilecek şekilde tasarlanabilir.

### Model Mimarileri

1.  **Sıfırdan RNN (RNN From Scratch):**
    * Bu model, temel RNN hücre yapısını NumPy kullanarak implemente etmeyi amaçlar[cite: 5].
    * Model, bir girdi katmanı (kelime temsilleri için, örneğin one-hot encoding veya basit indeksleme), tek bir gizli RNN katmanı (tanh aktivasyon fonksiyonu ile) ve ikili sınıflandırma için bir çıktı katmanından (sigmoid aktivasyon fonksiyonu ile) oluşur.
    * Gizli durum (hidden state) her zaman adımında güncellenir ve bir sonraki adıma aktarılır.
    * Eğitim süreci, zaman içinde geri yayılım (Backpropagation Through Time - BPTT) algoritmasının temel prensiplerini içermelidir (Bu taslakta tam implementasyon bulunmamaktadır, ödev kapsamında tamamlanması beklenir).
    * *Seçilen Hiperparametreler:* (Gizli katman boyutu, öğrenme oranı vb. buraya yazılacak)[cite: 6].

2.  **Kütüphane Tabanlı RNN (Library RNN - TensorFlow/Keras):**
    * Bu model, TensorFlow/Keras kütüphanesi kullanılarak oluşturulmuştur[cite: 6].
    * Mimari şu katmanları içerir:
        * `Input`: Sabit uzunluktaki (padding sonrası) dizileri alır.
        * `Embedding`: Kelime indekslerini öğrenilebilir yoğun vektörlere (embedding) dönüştürür. Bu, one-hot encoding'e göre daha verimli bir temsildir.
        * `SimpleRNN`: Keras'ın temel RNN katmanıdır. Gizli durumları hesaplar ve zaman adımları boyunca bilgiyi taşır.
        * `Dense`: Tam bağlı çıktı katmanıdır. Tek bir nöron ve sigmoid aktivasyon fonksiyonu ile ikili sınıflandırma (0 veya 1) olasılığını hesaplar.
    * Model, `adam` optimizasyon algoritması ve `binary_crossentropy` kayıp fonksiyonu ile derlenmiştir. Performans metriği olarak `accuracy` (doğruluk) kullanılmıştır.
    * *Seçilen Hiperparametreler:* (Embedding boyutu, RNN birim sayısı, epoch sayısı, batch boyutu vb. buraya yazılacak)[cite: 6].

## Sonuçlar (Results)

Bu bölümde, her iki modelin eğitim ve test süreçleri sonunda elde edilen performans metrikleri sunulacaktır. Sonuçlar, tablolar ve grafikler kullanılarak görselleştirilecektir[cite: 9].

* **Eğitim Grafikleri (Kütüphane Modeli):** Kütüphane ile eğitilen modelin her epoch'taki eğitim ve doğrulama (validation) kayıp (loss) ve doğruluk (accuracy) değerlerinin grafikleri aşağıdadır.
    * *(Buraya `library_model_training_curves.png` grafiği eklenecek veya referans verilecek)*
* **Performans Metrikleri (Test Seti):** Her iki modelin test veri seti üzerindeki performansları aşağıdaki tabloda özetlenmiştir.
    * *(Buraya Doğruluk, Kayıp, (varsa Precision, Recall, F1-Skor) içeren bir tablo eklenecek)*

    | Model Türü         | Test Kaybı (Loss) | Test Doğruluğu (Accuracy) |
    | :----------------- | :---------------- | :------------------------ |
    | Sıfırdan RNN       | (Hesaplanacak)    | (Hesaplanacak)            |
    | Kütüphane RNN      | (Hesaplandı)      | (Hesaplandı)              |

* **Karmaşıklık Matrisleri (Confusion Matrices):** Test seti üzerindeki sınıflandırma sonuçlarını detaylı gösteren karmaşıklık matrisleri aşağıdadır[cite: 9].
    * *(Buraya `confusion_matrices.png` grafiği eklenecek veya referans verilecek)*
    * *(Matrisler yorumlanacak: True Positive, False Positive, True Negative, False Negative sayıları belirtilecek)*

## Tartışma (Discussion)

Bu bölümde, elde edilen sonuçlar yorumlanacak, iki modelin performansı karşılaştırılacak ve gözlemlenen avantaj/dezavantajlar tartışılacaktır[cite: 9].

* **Performans Karşılaştırması:** Hangi modelin test setinde daha başarılı olduğu (örn. doğruluk açısından) belirtilecek. Olası nedenler (örn. kütüphane optimizasyonları, embedding katmanı kullanımı, sıfırdan implementasyondaki zorluklar) tartışılacak.
* **Model Avantaj/Dezavantajları:**
    * *Sıfırdan RNN:* RNN'in iç işleyişini derinlemesine anlama imkanı sunar. Ancak, implementasyonu karmaşık ve hataya açıktır, optimizasyonu zordur ve genellikle kütüphane implementasyonları kadar verimli çalışmayabilir.
    * *Kütüphane RNN:* Hızlı prototipleme ve geliştirme imkanı sunar. Optimize edilmiş katmanlar (örn. `Embedding`, `SimpleRNN`) ve eğitim döngüleri içerir. Daha az kodla daha karmaşık modeller oluşturulabilir. Dezavantajı, soyutlama nedeniyle iç işleyişin detaylarından uzaklaşılması olabilir.
* **Karşılaşılan Zorluklar ve Sınırlılıklar:** Veri setinin küçüklüğü, sıfırdan RNN implementasyonundaki zorluklar (özellikle geri yayılım), hiperparametre ayarlamanın etkisi gibi konulara değinilebilir.
* **Gelecek Çalışmalar:** Veri setinin büyütülmesi, daha karmaşık RNN mimarilerinin (LSTM, GRU) denenmesi, farklı önişleme tekniklerinin kullanılması gibi potansiyel iyileştirmelerden bahsedilebilir.

## Referanslar (References)

1.  YZM304 Ders Ödevi Tanımı, Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği, 2024-2025 Bahar Dönemi[cite: 1].
2.  Zhou, V. (2019). RNNs from Scratch in Python. Erişim adresi: [https://github.com/vzhou842/rnn-from-scratch](https://github.com/vzhou842/rnn-from-scratch)[cite: 5].
3.  (Eğer kullandıysanız) TensorFlow/Keras Dokümantasyonu: [https://www.tensorflow.org/api_docs/python/tf/keras](https://www.tensorflow.org/api_docs/python/tf/keras)
4.  (Eğer kullandıysanız) NumPy Dokümantasyonu: [https://numpy.org/doc/](https://numpy.org/doc/)
5.  (Eğer kullandıysanız) Matplotlib Dokümantasyonu: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
6.  (Eğer kullandıysanız) Scikit-learn Dokümantasyonu: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
7.  (Teorik açıklamalar için kullandığınız diğer kaynaklar)
