# modul-belajar

Pustaka **pembelajaran mesin** untuk bahasa [Tenun](https://github.com/TenunLang/Tenun) — ditulis murni dengan Tenun, berjalan di atas bytecode VM. Mencakup aljabar linear, regresi, jaringan saraf dengan propagasi-balik, k-NN, dan k-means.

Tidak ada ketergantungan eksternal: seluruh perhitungan dibangun di atas tipe `desimal`, larik `[]desimal`, dan builtin matematika Tenun (`eksp`, `ln`, `akar`, `tanh`, `pangkat`).

## Pasang

```
tenun add belajar
```

```tenun
impor "belajar";
```

Seluruh fungsi tersedia secara global dengan awalan jelas: `mat_*`, `vek_*` (aljabar linear), `akt_*` (aktivasi), `rugi_*` (fungsi rugi), `jar_*` (jaringan saraf), `reg_*` (regresi), `knn_*`, `kmeans`/`klaster_*`, `stat_*`, `data_*`.

## Contoh: melatih MLP menyelesaikan XOR

```tenun
impor "belajar";

biar X: [][]desimal = [[0.0,0.0], [0.0,1.0], [1.0,0.0], [1.0,1.0]];
biar Y: [][]desimal = [[0.0], [1.0], [1.0], [0.0]];

biar lapisan: []bulat = [2, 6, 1];                       // 2 masukan -> 6 tersembunyi -> 1 keluaran
biar bobot: [][][]desimal = jar_bobot_baru(lapisan, 1.0);
biar bias: [][]desimal = jar_bias_baru(lapisan);

jar_latih(bobot, bias, X, Y, 4000, 0.5);                 // 4000 epok, laju 0.5

untuk i dari 0 sampai 4 {
    cetak(jar_prediksi(bobot, bias, X[i])[0]);           // -> ~0, ~1, ~1, ~0
}
```

## Contoh: regresi linear

```tenun
biar model: []desimal = reg_model_baru(2);               // 2 fitur
reg_linear_latih(model, X, y, 2000, 0.02);
cetak(reg_linear_prediksi(model, [5.0, 5.0]));
```

## Struktur folder

```
modul-belajar/
  tenun.json
  README.md
  src/
    belajar.tenun          titik masuk; mengimpor seluruh bagian
    core/
      matriks.tenun        vektor & matriks: dot, kali, transpos, hadamard
      aktivasi.tenun       sigmoid, relu, tanh, softmax + turunannya
      rugi.tenun           MSE, entropi-silang biner & kategorikal
      jaringan.tenun       MLP: maju, propagasi-balik, latih, prediksi
      regresi.tenun        regresi linear & logistik (gradient descent)
      tetangga.tenun       klasifikasi k-NN
      klaster.tenun        pengklasteran k-means
    utils/
      stat.tenun           rata, varian, stdev, argmaks, normalisasi, akurasi
      data.tenun           one-hot, jarak Euclidean, acak-urut, bagi latih/uji
  examples/
    xor.tenun              MLP belajar XOR
    regresi_linear.tenun   regresi linear
    klasifikasi_knn.tenun  klasifikasi k-NN
    kmeans.tenun           pengklasteran k-means
```

## Acuan API

### Aljabar linear (`core/matriks.tenun`)

| Fungsi | Keterangan |
|---|---|
| `vek_nol(n)`, `vek_isi(n, v)`, `vek_acak(n, skala)` | konstruksi vektor |
| `mat_nol(b, k)`, `mat_acak(b, k, skala)` | konstruksi matriks |
| `vek_dot`, `vek_tambah`, `vek_kurang`, `vek_skalar`, `vek_hadamard`, `vek_jumlah` | operasi vektor |
| `mat_kali_vek`, `mat_kali`, `mat_transpos` | operasi matriks |

### Aktivasi (`core/aktivasi.tenun`)

`akt_sigmoid`, `akt_relu`, `akt_tanh`, `akt_softmax` beserta turunan `akt_d_sigmoid`, `akt_d_relu`, `akt_d_tanh`.

### Fungsi rugi (`core/rugi.tenun`)

`rugi_mse`, `rugi_mse_grad`, `rugi_silang_biner`, `rugi_silang`.

### Jaringan saraf (`core/jaringan.tenun`)

| Fungsi | Keterangan |
|---|---|
| `jar_bobot_baru(lapisan, skala)` | inisialisasi bobot dari arsitektur `[]bulat` |
| `jar_bias_baru(lapisan)` | inisialisasi bias |
| `jar_prediksi(bobot, bias, x)` | propagasi maju, kembalikan keluaran |
| `jar_latih_satu(...)` | satu langkah SGD (propagasi-balik) |
| `jar_latih(bobot, bias, X, Y, epok, laju)` | loop latih penuh |

### Regresi (`core/regresi.tenun`)

`reg_model_baru`, `reg_linear_prediksi`, `reg_linear_latih`, `reg_logistik_prediksi`, `reg_logistik_latih`.

### k-NN & k-means

`knn_klasifikasi(X, y, x, k)` · `kmeans(X, k, iterasi)`, `klaster_terdekat`, `klaster_tugas`.

### Utilitas

`stat_rata`, `stat_varian`, `stat_stdev`, `stat_min`, `stat_maks`, `stat_argmaks`, `stat_normalisasi`, `stat_minmax`, `stat_akurasi` · `data_satu_panas`, `data_jarak`, `data_acak_urut`, `data_batas_latih`.

## Catatan

- Aktivasi jaringan saraf memakai sigmoid pada setiap lapisan; rugi MSE.
- Larik di Tenun bertipe rujukan, sehingga fungsi latih memperbarui bobot di tempat.
- Modul ini berjalan pada VM (`tenun run`). Backend native (`tenun build`) belum mendukung builtin matematika.

## Lisensi

MIT.
