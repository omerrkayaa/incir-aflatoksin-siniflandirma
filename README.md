# İncir Görüntülerinde Aflatoksin Sınıflandırması

İncir fotoğraflarını **sağlıklı** ve **sağlıksız (aflatoksinli)** olarak ayıran derin öğrenme çalışması. Staj kapsamında yapıldı: veri temizleme, ön işleme, beş CNN mimarisinin transfer learning ile karşılaştırılması, üç farklı yöntemle hiperparametre optimizasyonu ve Grad-CAM ile açıklanabilirlik analizi.

> **Veri seti bu depoda yok.** Fotoğrafları paylaşma iznim olmadığı için depoda yalnız kod, sonuç tabloları ve grafikler var. Model ağırlıkları da boyutları nedeniyle eklenmedi.

## İş akışı

| Adım | Betik | Ne yapıyor |
|---|---|---|
| 1. Bölme | `dataset_split.py` | Sınıfları train / val / test olarak %70 / %15 / %15 böler (sabit tohum). Veri artırılmış (augmented) görüntüler **yalnız eğitime** girer, yalnız train'e düşen orijinallerden türetilmiş olanlar seçilir. Böylece bir test görüntüsünün kopyası eğitimde bulunmaz. |
| 2. Kalite kontrolü | `make_review_sheets.py`, `apply_review_notes.py`, `undo_review_notes.py` | Görüntüleri toplu inceleme sayfalarına dizer. Elle verilen kararlarla (el/parmak görünen, boş arka plan, bozuk, bulanık görüntüler) kötü örnekleri silmeden ayrı klasöre taşır. İşlem geri alınabilir. |
| 3. Ön işleme | `preprocess_dataset.py` | EXIF yönü düzeltme, RGB, merkez kırpma, 224×224 |
| 4. Eğitim | `train_resnet18.py`, `train_resnet50.py`, `train_efficientnet.py`, `train_densenet121.py`, `train_convnext_tiny.py` | ImageNet ön eğitimli modellerin son katmanını iki sınıfa uyarlayıp eğitir. Ortak ayarlar: 3 epoch, Adam, lr 1e-4, weight decay 1e-4, batch 32. |
| 5. Optimizasyon | `optimize_efficientnet_hyperopt.py`, `optimize_efficientnet_optuna.py`, `optimize_efficientnet_pso.py` | EfficientNet-B0 için learning rate (1e-5 – 1e-2), weight decay (1e-6 – 1e-3) ve batch size (32/64/128) araması. Seçim **validation** doğruluğuyla yapılır, test seti yalnız en sonda bir kez kullanılır. |
| 6. Açıklanabilirlik | `generate_gradcam.py` | En iyi üç model için Grad-CAM ve Grad-CAM++ ısı haritaları |
| — | `dataset_stats.py` | Bölme ve temizleme sonrası görüntü sayıları ([dataset_statistics.csv](dataset_statistics.csv)) |

## Veri

| Bölüm | Sağlıklı | Sağlıksız | Toplam |
|---|---|---|---|
| Train | 814 | 1023 (çoğu veri artırılmış) | 1837 |
| Validation | 175 | 26 | 201 |
| Test | 172 | 26 | 198 |

Sınıflar dengesiz olduğu için yalnız doğruluğa bakılmadı; precision, recall, F1 ve karışıklık matrisi de raporlandı ([model_outputs/](model_outputs)).

## Sonuçlar (test seti)

Karışıklık matrisi `[[doğru sağlıklı, yanlış alarm], [kaçırılan sağlıksız, doğru sağlıksız]]` biçimindedir.

| Model | Val doğruluk | Test doğruluk | Karışıklık matrisi |
|---|---|---|---|
| ResNet18 | 0.9900 | 0.9899 | [[170, 2], [0, 26]] |
| ResNet50 | 0.9950 | 1.0000 | [[172, 0], [0, 26]] |
| EfficientNet-B0 | 0.9900 | 0.9949 | [[171, 1], [0, 26]] |
| DenseNet121 | 0.9851 | 0.9848 | [[169, 3], [0, 26]] |
| ConvNeXt-Tiny | 1.0000 | 0.9899 | [[172, 0], [2, 24]] |
| EfficientNet-B0 + Hyperopt (TPE) | 0.9950 | 1.0000 | [[172, 0], [0, 26]] |
| EfficientNet-B0 + Optuna (TPE) | 0.9950 | 1.0000 | [[172, 0], [0, 26]] |
| EfficientNet-B0 + PSO | 0.9900 | 0.9949 | [[172, 0], [1, 25]] |

Optimizasyon bütçesi küçüktü: Hyperopt ve Optuna'da 10'ar, PSO'da 8 deneme (4 parçacık × 2 iterasyon), deneme başına 3 epoch. En iyi parametreler `model_outputs/*/best_params.json` dosyalarında, tüm denemeler `*_trials.csv` dosyalarında.

## Sınırlılıklar: sonuçlara neden temkinli bakılmalı

Test doğruluğunun %99–100 çıkması bu veri setinde problemin çözüldüğü anlamına gelmiyor:

- **Test seti küçük.** Sağlıksız sınıfta yalnız 26 görüntü var. Bir görüntü bile sonucu yaklaşık 4 puan değiştiriyor ve %100 ile %96 arasındaki fark istatistiksel olarak ayırt edilemiyor.
- **Aynı incirin birden çok fotoğrafı olabilir.** Görüntüler saniyeler arayla çekilmiş seriler halinde. Bölme görüntü düzeyinde rastgele yapıldığı için aynı incirin farklı açıları hem eğitimde hem testte bulunabilir. Bu durumda model test sırasında daha önce gördüğü bir inciri tanıyor olabilir. Daha sağlam bir değerlendirme için bölmenin incir ya da çekim serisi düzeyinde yapılması gerekir.
- **Çekim koşulu sınıfla örtüşüyor olabilir.** Sağlıksız örneklerin hepsi tek bir çekim oturumundan geliyor. Sağlıklı örneklerin bir kısmı ise başka günlerde, farklı cihaz adlandırmasıyla çekilmiş. Model hastalığın kendisi yerine ışık, arka plan ya da kamera farkını öğrenmiş olabilir. Grad-CAM incelemesi bu yüzden yapıldı: modelin incire mi yoksa arka plana mı baktığını görmek için. İncelenen 6 örnekle bu soru kesin olarak yanıtlanamıyor.
- **Dış doğrulama yok.** Modeller farklı bir yerden, farklı koşullarda çekilmiş incirlerle denenmedi.

Sonraki adımlar: çekim serisine göre gruplu bölme, farklı oturumlardan sağlıksız örnek toplama, daha fazla epoch ve tekrarlı deneylerle ortalama ± sapma raporlama.

## Çalıştırma

```bash
pip install -r requirements.txt   # CUDA'lı PyTorch için pytorch.org'daki kurulum komutu önerilir
python dataset_split.py
python preprocess_dataset.py
python train_efficientnet.py
python optimize_efficientnet_optuna.py --n-trials 10
python generate_gradcam.py
```

Betikler veri klasörlerini kendi bulundukları klasörde arar (`dataset_split/`, `dataset_processed/` …). Veri seti olmadan çalışmazlar.

Deneyler NVIDIA RTX 3060 Laptop GPU üzerinde yapıldı.
