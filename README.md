# Mini Project MLOps — Docker Compose, Grafana, dan MLflow

Dokumentasi ini menjelaskan cara menjalankan stack dengan Docker Compose dan membuka URL Grafana serta MLflow untuk observabilitas dan pelacakan eksperimen.

## Prasyarat
- Docker Desktop dan `docker-compose` terpasang.
- Port kosong: `8000` (API), `9090` (Prometheus), `3000` (Grafana), `5000` (MLflow).
- Repo berada di direktori proyek: `c:\Users\FS-User\Documents\bootcamp\mini-project-mlops` (sesuaikan jika berbeda).

## Menjalankan Stack
1. Buka terminal di root proyek.
2. Jalankan stack:
   - `docker-compose up -d`
3. Cek status kontainer:
   - `docker-compose ps`
4. (Opsional) Ikuti log layanan tertentu:
   - `docker-compose logs -f prometheus grafana`
   - `docker-compose logs -f mlflow api`

## URL
- API (FastAPI): `http://localhost:8000/` dan Swagger UI di `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090/`
- Grafana: `http://localhost:3000/`
- MLflow UI: `http://localhost:5000/`

## Konfigurasi yang Digunakan
- `docker-compose.yml` menjalankan 4 layanan: `api`, `prometheus`, `grafana`, `mlflow`.
- Prometheus scrape target: `host.docker.internal:8000` (lihat `config/prometheus.yml`). Ini men-scrape endpoint `/metrics` dari API.
- Grafana provisioning: `config/grafana/provisioning` (data source dan dashboard dapat diprovide otomatis jika dikonfigurasi).
- MLflow UI memakai backend SQLite di `mlflow/mlflow.db` dan artifacts di `mlruns/`.

## Melihat Data di Grafana
1. Buka `http://localhost:3000/`.
2. Login (default Grafana biasanya `admin/admin`, lalu diminta ganti password). Jika provisioning sudah diatur, data source Prometheus akan siap.
3. Lakukan trafik ke API untuk menghasilkan metrik, misalnya uji endpoint `/predict`
4. Buka dashboard Grafana (atau buat panel baru) dan pilih data source Prometheus.

## Melihat Eksperimen di MLflow
Ada dua skenario tergantung di mana menjalankan training:

- Training di HOST (local):
  - Pastikan `config/config.yaml` berisi `mlflow.tracking_uri: "http://localhost:5000"`.
  - Jalankan script training:
    - `python -c "from src.models.trainer import train_all; train_all()"`
  - Buka `http://localhost:5000/` dan pilih eksperimen `house-price` untuk melihat run dan artifacts.

- Training di dalam kontainer API:
  - Ubah `config/config.yaml` agar `mlflow.tracking_uri: "http://mlflow:5000"` (akses antar-kontainer menggunakan nama layanan Compose).
  - Masuk ke kontainer API dan jalankan training:
    - `docker exec -it house-price-api bash -lc "python -c 'from src.models.trainer import train_all; train_all()'"`
  - Buka `http://localhost:5000/` untuk melihat run.

## Troubleshooting
- Grafana "No data":
  - Pastikan Prometheus target UP di `http://localhost:9090/targets`.
  - Scrape target di `config/prometheus.yml` harus mengarah ke API yang diekspos (`host.docker.internal:8000` atau ganti ke `api:8000`).
  - Restart Prometheus: `docker-compose restart prometheus`.

- MLflow kosong:
  - Verifikasi `config/config.yaml` → `mlflow.tracking_uri` sesuai (HOST: `http://localhost:5000`, kontainer: `http://mlflow:5000`).
  - Jalankan training lagi untuk menghasilkan run baru.
  - Cek port `5000` tidak diblokir.