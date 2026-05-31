# NeuroAnalyse Demo

NeuroAnalyse, 3D T1-weighted MRI görüntülerinden amiloid yükünü Centiloid ölçeğinde tahmin eden web tabanlı bir demo uygulamasıdır.

Bu repo yalnızca demo/runtime paketi olarak hazırlanmıştır. Eğitim kodları, eğitim notebookları, eğitim CSV dosyaları, preprocessing deneyleri ve model geliştirme klasörleri bu pakete dahil değildir.

## İçerik

- FastAPI backend
- Tek dosyalık React frontend
- Demo hasta ve MRI örnekleri
- PDF rapor üretimi
- Late Fusion inference runtime
- En iyi seçilmiş full-brain ve masked-region model checkpointleri

## Model

Arayüz `LateFusion-v1.0` modelini kullanır.

Model iki ayrı 3D ResNet tahminini birleştirir:

- `model_runtime/full_model.ckpt`
- `model_runtime/masked_model.ckpt`

Son Centiloid tahmini full-brain ve masked-region çıktılarının 50/50 ortalaması ile hesaplanır.

## Kurulum

Python 3.11 önerilir.

```bash
git clone https://github.com/elifkk14/neuroanalyse.git
cd neuroanalyse
cd interface/backend
pip install -r requirements.txt
cd ../..
./start.sh
```

Uygulama açıldıktan sonra tarayıcıda şu adresi ziyaret edin:

```text
http://127.0.0.1:8001
```

## Demo Giriş Bilgileri

Klinisyen hesabı:

```text
Username: ayse.yilmaz
Password: Test1234
```

Admin hesabı:

```text
Username: admin@neuroanalyse.local
Password: Admin1234
```

## Notlar

- İlk açılışta demo veritabanı ve örnek raporlar otomatik oluşturulur.
- Yüklenen MRI dosyaları analiz sonrası saklanmaz.
- Bu demo araştırma ve ürün gösterimi amacıyla hazırlanmıştır; klinik tanı aracı olarak kullanılmamalıdır.

