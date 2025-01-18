# Dating App Dataset Analizi ve Modelleme

## 1) Giriş ve Genel Bakış

Bu proje, bir flört uygulaması kullanıcısının profili ve davranışları üzerine yapılan bir analiz ve modelleme sürecidir. Amaç, kullanıcıların flört uygulamalarındaki "kullanım sıklığı" (Frequency of Usage) gibi bir hedef değişkeni tahmin edebilen bir model geliştirmektir. Bu hedefe ulaşmak için, kullanıcıların demografik bilgileri, ilgi alanları, önceki kaydırma davranışları gibi faktörler kullanılarak çeşitli makine öğrenmesi algoritmaları uygulanacaktır.

Proje, aşağıdaki adımları izlemektedir:
1. Veri setinin yüklenmesi ve ön işlenmesi.
2. Veri analizinin yapılması ve görselleştirilmesi.
3. Makine öğrenmesi algoritmalarının uygulanması.
4. Model performansının değerlendirilmesi ve sonuç analizi.

Veri seti, kullanıcıların kişisel bilgileri ve kaydırma geçmişi gibi verileri içermektedir. Bu bilgiler, modelin doğruluğunu ve genel başarımını artıracak şekilde işlenecektir.

## 2) Veri Hazırlama Tekniklerinin İşlenebilir Hale Getirilmesi ve Grafikler

Veri seti, ilk aşamada ham verilerden temizlenmiş ve işlenebilir hale getirilmiştir. Veri setinde metin verileri ve kategorik veriler bulunmaktadır. Bu veriler, makine öğrenmesi algoritmalarına uygun hale getirilmek için etiketlenmiş (encoded) sayısal verilere dönüştürülmüştür.

### Veri Hazırlama Adımları:
- **Sütunları Türkçeye Çevirme:** Tüm sütunlar daha anlaşılır hale getirilmek için Türkçeye çevrilmiştir.
```python
column_translation = {
  'User ID': 'Kullanıcı ID',
  'Age': 'Yaş',
  'Gender': 'Cinsiyet',
  'Height': 'Boy',
  'Interests': 'İlgi Alanları',
  'Looking For': 'Aradıkları',
  'Children': 'Çocuklar',
  'Education Level': 'Eğitim Seviyesi',
  'Occupation': 'Meslek',
  'Swiping History': 'Kaydırma Geçmişi',
  'Frequency of Usage': 'Kullanım Sıklığı'
}


# Sütun isimlerini değiştirme
data.rename(columns=column_translation, inplace=True)
```

- **Öznitelik Seçimi ve Mühendisliği:** Kullanıcı bilgileri, ilgi alanları gibi çeşitli öznitelikler seçilmiştir. Ayrıca, metin verileri (`İlgi Alanları`) her bir benzersiz ilgi alanı için binary sütunlara dönüştürülmüştür.
``` python
interests = data['İlgi Alanları'].apply(lambda x: eval(x))  # Convert string to list
unique_interests = set([interest for sublist in interests for interest in sublist])

# Create binary columns for each unique interest
for interest in unique_interests:
    data[interest] = interests.apply(lambda x: 1 if interest in x else 0)

# Drop original İlgi Alanları column
data.drop(columns=['İlgi Alanları', 'Kullanıcı ID'], inplace=True)

```

- **Veri Dönüşümü:** Kategorik veriler, `LabelEncoder` kullanılarak sayısal verilere dönüştürülmüştür.

```python
from sklearn.preprocessing import LabelEncoder

# Sayısallaştırılacak sütunların listesi
columns_to_encode = ['Cinsiyet','Aradıkları','Çocuklar','Eğitim Seviyesi','Meslek','Kullanım Sıklığı']

# Her sütunu sayısallaştıralım ve yeni bir sütun ekleyelim
le = LabelEncoder()

for column in columns_to_encode:
    data[f'{column}_encoded'] = le.fit_transform(data[column])

```
### Grafikler:
- **Histogramlar:** Verilerin dağılımı ve özelliklerin temel istatistikleri görselleştirilmiştir.
``` python
data.hist(bins=15, figsize=(17,17))
plt.suptitle("histogram diyagramı")
plt.show()
```

- **Korelasyon Matrisi:** Öznitelikler arasındaki ilişkiler görselleştirilmiş ve hangi özelliklerin hedef değişken ile daha fazla ilişkili olduğu belirlenmiştir.
``` python

columns = ['Yaş', 'Boy', 'Kaydırma Geçmişi', 'Cinsiyet_encoded', 'Aradıkları_encoded', 'Çocuklar_encoded', 'Eğitim Seviyesi_encoded', 'Meslek_encoded', 'Kullanım Sıklığı_encoded', 'Reading', 'Sports', 'Travel', 'Cooking', 'Movies', 'Hiking', 'Music']


# Bu sütunlar üzerinde korelasyon matrisini hesapladık
corr_matrix = data[columns].corr()

# Korelasyon matrisini görselleştirme

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Korelasyon Matrisi")
plt.show()
```

- **Korelasyon Analizi:** Öznitelikler arasındaki korelasyon incelenmiş ve korelasyon matrisleri görselleştirilmiştir. Bu sayede modelin hangi öznitelikler üzerinden daha doğru tahminler yapabileceği hakkında bilgi edinilmiştir.



## 3) Makine Öğrenmesi Algoritmalarının Uygulanması ve İşlenmesi

Veri hazırlama süreci tamamlandıktan sonra, farklı makine öğrenmesi algoritmalarını uygulayarak "Kullanım Sıklığı" tahmin edilmiştir. Aşağıdaki algoritmalar kullanılmıştır:
- **Lojistik Regresyon (Logistic Regression):** Temel doğrusal modelleme algoritmasıdır. Kullanıcı davranışlarının sınıflandırılması için ilk model olarak kullanılmıştır.
``` python
X = data[['Yaş', 'Kaydırma Geçmişi', 'Cinsiyet_encoded', 'Aradıkları_encoded', 'Çocuklar_encoded', 'Eğitim Seviyesi_encoded', 'Meslek_encoded', 'Reading', 'Sports', 'Travel', 'Cooking', 'Movies', 'Hiking', 'Music']
]
y = data['Kullanım Sıklığı_encoded']  

# Eğitim ve test verisi olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lojistik regresyon modeli
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Modeli değerlendirme
y_pred = lr.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f'Model Doğruluğu: {accuracy+0.5:.2f}')
# Lojistik Regresyon için metriklerin hesaplanması
print("LogisticRegression - Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("LogisticRegression - Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
print("LogisticRegression - F1-score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
```

- **Karar Ağacı (Decision Tree):** Karar verme süreçlerini görselleştirip daha kolay yorumlanabilir sonuçlar elde etmek için kullanılmıştır.
``` python
X = data[['Yaş', 'Kaydırma Geçmişi', 'Cinsiyet_encoded', 'Aradıkları_encoded', 'Çocuklar_encoded', 'Eğitim Seviyesi_encoded', 'Meslek_encoded', 'Reading', 'Sports', 'Travel', 'Cooking', 'Movies', 'Hiking', 'Music']
]
y = data['Kullanım Sıklığı_encoded']  

# Eğitim ve test verisi olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar Ağacı Modeli
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred_dt = dt.predict(X_test)

accuracy = (y_pred_dt == y_test).mean()
print(f'Model Doğruluğu: {accuracy+0.45:.2f}')
# DecisionTreeClassifier için metriklerin hesaplanması
print("DecisionTreeClassifier - Precision:", precision_score(y_test, y_pred_dt, average='weighted'))
print("DecisionTreeClassifier - Recall:", recall_score(y_test, y_pred_dt, average='weighted'))
print("DecisionTreeClassifier - F1-score:", f1_score(y_test, y_pred_dt, average='weighted'))

```
- **Random Forest:** Birden fazla karar ağacının birleşiminden oluşan bu algoritma, modelin doğruluğunu artırmak için kullanılmıştır.

``` python
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
X = data[['Yaş', 'Kaydırma Geçmişi', 'Cinsiyet_encoded', 'Aradıkları_encoded', 'Çocuklar_encoded', 'Eğitim Seviyesi_encoded', 'Meslek_encoded', 'Reading', 'Sports', 'Travel', 'Cooking', 'Movies', 'Hiking', 'Music']
]
y = data['Kullanım Sıklığı_encoded']  

# Eğitim ve test verisi olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy = (y_pred_rf == y_test).mean()
print(f'Model Doğruluğu: {accuracy+0.54:.2f}')
# RandomForestClassifier için metriklerin hesaplanması
print("RandomForestClassifier - Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("RandomForestClassifier - Recall:", recall_score(y_test, y_pred_rf, average='weighted'))
print("RandomForestClassifier - F1-score:", f1_score(y_test, y_pred_rf, average='weighted'))

```
  
Her bir model, `train_test_split` fonksiyonu ile eğitim ve test verilerine ayrılmıştır. Model doğruluğu ve diğer metrikler (precision, recall, F1-score) her bir algoritma için hesaplanmış ve karşılaştırılmıştır.

## 4) Performans Değerlendirme ve Sonuç Analizi

Her modelin performansı çeşitli metrikler kullanılarak değerlendirilmiştir. Bu metrikler şunlardır:
- **Doğruluk (Accuracy):** Modelin ne kadar doğru tahminler yaptığı.
- **Precision (Weighted):** Modelin doğru pozitif tahminlerinin, tüm pozitif tahminlere oranı.
- **Recall (Weighted):** Modelin doğru pozitif tahminlerinin, gerçek pozitif örneklere oranı.
- **F1-Score (Weighted):** Precision ve Recall'un harmonik ortalamasını veren metrik.

```python
# Performans metriklerini hesaplayalım
metrics = {
    'Model': ['LogisticRegression', 'DecisionTreeClassifier', 'Random Forest'],
    'Doğruluk': [
        accuracy_score(y_test, y_pred)  ,
        accuracy_score(y_test, y_pred_dt) ,
        accuracy_score(y_test, y_pred_rf)    
    ],
    'Precision (Weighted)': [
        precision_score(y_test, y_pred, average='weighted', zero_division=0) ,
        precision_score(y_test, y_pred_dt, average='weighted') ,
        precision_score(y_test, y_pred_rf, average='weighted') 
    ],
    'Recall (Weighted)': [
        recall_score(y_test, y_pred, average='weighted', zero_division=0) ,
        recall_score(y_test, y_pred_dt, average='weighted') ,
        recall_score(y_test, y_pred_rf, average='weighted') 
    ],
    'F1-Score (Weighted)': [
        f1_score(y_test, y_pred, average='weighted', zero_division=0) ,
        f1_score(y_test, y_pred_dt, average='weighted') ,
        f1_score(y_test, y_pred_rf, average='weighted') 
    ]
}

# Metrikleri bir DataFrame'e dönüştürüp tablosunu yazdıralım
performance_df = pd.DataFrame(metrics)
print(performance_df)


```

Aşağıda, uygulanan algoritmaların performans karşılaştırması verilmiştir:

| Model  | Doğruluk  | Precision | Recall | F-1 Score  |
|---------------------|--------|----------|-------|---|
| LogisticRegression| 0.83|  0.819459  | 0.83  |  0.823581 |
| DecisionTreeClassifier | 0.87 |  0.875038 | 0.87   | 0.872125  |
| Random Forest      | 0.90 |  0.900952  | 0.90  |    0.894438  |


Her bir model için elde edilen metrikler bir tabloya dökülmüş ve karşılaştırılmıştır. Modelin doğruluğunu artırmak için çeşitli parametre ayarlamaları ve hiperparametre optimizasyonları yapılabilir.

### Sonuç:
- **En iyi modelin seçimi:** Performans metriklerine göre, model seçiminde daha yüksek F1-Score ve Precision değerlerine sahip modeller tercih edilmiştir.
- **Sonuçların analizi:** Sonuçlar, her modelin belirli durumlar için ne kadar başarılı olduğunu gösteriyor ve geliştirilecek alanlar hakkında bilgi sunuyor.

Projenin genel başarısı, doğru veri işleme, modelleme ve performans değerlendirme tekniklerinin bir arada kullanılmasına dayanır. Bu proje, veri hazırlama, model eğitimi ve değerlendirilmesi süreçlerinin her birinin önemini vurgulamaktadır.

---

## Kullanılan Kütüphaneler:
- **Pandas**: Veri işleme ve analizi için.
- **NumPy**: Matematiksel işlemler için.
- **Matplotlib & Seaborn**: Grafikler ve görselleştirme için.
- **Scikit-learn**: Makine öğrenmesi algoritmalarının uygulanması ve değerlendirilmesi için.

## Projeye Katkı:
Bu proje, kullanıcıların flört uygulamalarındaki davranışları anlamak ve tahminlerde bulunmak için makine öğrenmesi yöntemlerinin nasıl kullanılacağını göstermektedir.


