# Helper Function Toolkit

Kode dalam repositori ini ditujukan untuk memudahkan pengguna dalam melakukan tugas pengolahan data. Repositori ini berisi berbagai fungsi yang dapat digunakan untuk melakukan berbagai operasi pada data, termasuk pembuatan plot, tampilan gambar, penghitungan metrik, rekayasa fitur, seleksi fitur, dan berbagai fungsi lainnya.

## Deskripsi

Repositori ini menyediakan berbagai fungsi yang dapat digunakan untuk memudahkan pengguna dalam melakukan tugas pengolahan data. Fungsi-fungsi tersebut dirancang untuk mengurangi kompleksitas tugas yang umum dalam pengolahan data dan memberikan kemudahan bagi pengguna untuk melakukan manipulasi data yang diperlukan.

Beberapa fitur dan fungsi yang dapat ditemukan dalam repositori ini meliputi:

- Pembuatan plot untuk visualisasi data.
- Tampilan gambar dalam format yang sesuai.
- Perhitungan metrik untuk evaluasi model.
- Rekayasa fitur untuk meningkatkan representasi data.
- Seleksi fitur untuk mengidentifikasi fitur yang paling informatif.
- Dan masih banyak lagi.

## Cara Menggunakan

Untuk menggunakan fungsi-fungsi dalam repositori ini, Anda dapat mengikuti langkah-langkah berikut:

1. Pastikan Anda telah mengunduh repositori ini ke komputer Anda.

2. Impor modul yang berisi fungsi yang ingin Anda gunakan. Misalnya, untuk menggunakan fungsi pembuatan plot, Anda dapat mengimpor modul `plotting`:

```python
from helper_function import get_auc
```

3. Anda dapat menggunakan fungsi yang ada dalam modul tersebut dengan memanggilnya menggunakan sintaks `nama_fungsi()`. Misalnya, untuk mendapatkan auc, Anda dapat menggunakan fungsi `get_auc()`:

```python
auc = get_auc(model, model_name, X_train, y_train, X_test, y_test)
```
---

*Dibuat untuk memudahkan pengguna dalam melakukan tugas pengolahan data dengan berbagai fungsi yang tersedia.*
