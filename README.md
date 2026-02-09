# VidCV - AI Video CV Platform

<div align="center">

![VidCV Logo](docs/assets/logo.png)

**Yapay Zeka Destekli Video Özgeçmiş Platformu**

*İşverenler, İş Arayanlar ve Head Hunter'lar İçin*

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org)
[![React Native](https://img.shields.io/badge/React%20Native-0.73+-blue.svg)](https://reactnative.dev)

[Demo](https://vidcv.io) • [Dokümantasyon](docs/) • [API Reference](docs/api/) • [Katkıda Bulun](CONTRIBUTING.md)

</div>

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Teknik Mimari](#-teknik-mimari)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [Proje Yapısı](#-proje-yapısı)
- [API Dokümantasyonu](#-api-dokümantasyonu)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

---

## 🎯 Proje Hakkında

**VidCV**, iş arayanların CV'lerini yapay zeka destekli konuşan video avatar'lara dönüştürmelerini sağlayan, işverenlerin ve head hunter'ların aday değerlendirme süreçlerini devrimci bir şekilde değiştiren yeni nesil bir platformdur.

### Temel Değer Önerisi

| Kullanıcı | Değer |
|-----------|-------|
| **İş Arayanlar** | CV'lerini canlı, etkileyici video formatına dönüştürme. Opsiyonel anonim avatar ile gizlilik korunumu. |
| **İşverenler** | 30 saniyelik video ile aday ön tarama. Zaman tasarrufu ve daha iyi değerlendirme. |
| **Head Hunter'lar** | Aday havuzunu video ile sunma. Premium işe alım süreçlerinde fark yaratma. |
| **İK Firmaları** | White-label çözüm. Kendi markalarınızla AI video CV platformu sunma. |

---

## ✨ Özellikler

### İş Arayan Özellikleri
- 📄 **CV Yükleme & Parsing** - PDF/Word CV yükle, AI otomatik olarak bilgileri çıkarır
- 🎬 **AI Video Avatar Oluşturma** - Kişi fotoğrafını yükler veya hazır anonim avatar seçer
- 🔒 **Anonim Avatar Modu** - Gizlilik isteyenler için AI-üretilmiş profesyonel avatar
- 🌍 **Çoklu Dil Desteği** - Avatar, 30+ dilde CV sunumu yapabilir
- ✏️ **Video Önizleme & Düzenleme** - Oluşturulan videoyu izle, script düzenle
- 📊 **Başvuru Takibi** - Hangi işverenlerin videoyu izlediğini gör

### İşveren Özellikleri
- 📝 **İş İlanı Yayınlama** - Detaylı ilan oluşturma, yetenek gereksinimleri
- 🎥 **Video CV Galeri** - Başvuran adayların video CV'lerini kartlar halinde izleme
- 🤖 **AI Eşleşme Skoru** - İlan gereksinimleri ile aday profilini AI ile eşleştirme
- 🔍 **Filtreleme & Arama** - Şehir, yetenek, deneyim yılı, dil bazlı filtreleme
- 💬 **Doğrudan Mesajlaşma** - Aday ile platform üzerinden iletişim kurma
- 📈 **Analitik Dashboard** - İlan performansı, başvuru istatistikleri

### Head Hunter Özellikleri
- 👥 **Premium Aday Havuzu** - Özel onaylanmış, yüksek profilli aday veritabanı
- 📋 **Toplu Video Sunum** - Birden fazla adayı tek bir linkle işverene sunma
- 🔗 **CRM Entegrasyonu** - Mevcut CRM araçları ile senkronizasyon

---

## 🏗 Teknik Mimari

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────┬─────────────────┬─────────────────┬───────────────────────────┤
│  Web App    │   Mobile App    │   Admin Panel   │  Head Hunter Portal       │
│  (Next.js)  │ (React Native)  │   (Next.js)     │     (Next.js)             │
└──────┬──────┴────────┬────────┴────────┬────────┴─────────┬─────────────────┘
       │               │                 │                  │
       └───────────────┴────────┬────────┴──────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │     API Gateway       │
                    │   (Kong / Nginx)      │
                    │  Rate Limiting, Auth  │
                    └───────────┬───────────┘
                                │
       ┌────────────────────────┼────────────────────────┐
       │                        │                        │
┌──────▼──────┐          ┌──────▼──────┐          ┌──────▼──────┐
│ Auth Service│          │ User Service│          │  CV Service │
│  (Django)   │          │  (Django)   │          │  (Django)   │
└─────────────┘          └─────────────┘          └──────┬──────┘
       │                        │                        │
       │    ┌───────────────────┼───────────────────┐    │
       │    │                   │                   │    │
┌──────▼────▼─┐          ┌──────▼──────┐     ┌─────▼────▼─────┐
│Video Service│          │ Job Service │     │Matching Service│
│  (FastAPI)  │          │  (Django)   │     │   (FastAPI)    │
└─────────────┘          └─────────────┘     └────────────────┘
       │                        │                   │
       └────────────────────────┼───────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼────────┐    ┌─────────▼─────────┐    ┌───────▼────────┐
│ IMAGE PROCESSOR│    │    AI SERVICE     │    │  OLLAMA (LLM)  │
│   (FastAPI)    │    │    (FastAPI)      │    │   (Local AI)   │
│  OCR, Layout   │───►│  LangChain/Graph  │◄───│  No API Keys   │
│   Analysis     │    │  Video Gen, TTS   │    │                │
└────────────────┘    └───────────────────┘    └────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
       ┌────────────────────────┼────────────────────────┐
       │                        │                        │
┌──────▼──────┐          ┌──────▼──────┐          ┌──────▼──────┐
│ PostgreSQL  │          │    Redis    │          │Elasticsearch│
│  + pgvector │          │   (Cache)   │          │  (Search)   │
│ (AI Embed.) │          │  + Celery   │          │             │
└─────────────┘          └─────────────┘          └─────────────┘
        │                        │                       │
        │              ┌─────────▼─────────┐            │
        │              │   MinIO / R2      │            │
        └──────────────│  (Video Storage)  │────────────┘
                       └───────────────────┘
```

### Teknoloji Stack

| Katman | Teknoloji |
|--------|-----------|
| **Backend** | Python 3.11+, Django, DRF, Celery |
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS, Zustand |
| **Mobile** | React Native, Expo, TypeScript |
| **Database** | PostgreSQL + pgvector, Redis, Elasticsearch |
| **AI/ML** | Ollama (Local LLM), LangChain, LangGraph, ChromaDB |
| **Image Processing** | OpenCV, Tesseract/EasyOCR/PaddleOCR, PIL |
| **Video Generation** | HeyGen, D-ID, ElevenLabs TTS |
| **Message Queue** | Kafka, Redis (Celery) |
| **Storage** | MinIO (S3), Cloudflare R2 |
| **Infrastructure** | Docker, Kubernetes, Nginx |
| **CI/CD** | GitHub Actions |

---

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+

### Kurulum

```bash
# 1. Repo'yu klonla
git clone https://github.com/metinemredonmez/img-try.git
cd img-try

# 2. Environment dosyalarını oluştur
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# 3. Docker ile çalıştır (Önerilen)
docker-compose up -d

# VEYA Manuel Kurulum:

# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload

# Frontend (yeni terminal)
cd frontend
npm install
npm run dev

# Mobile (yeni terminal)
cd mobile
npm install
npx expo start
```

### Ortam Değişkenleri

Backend `.env` dosyası:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/vidcv
REDIS_URL=redis://localhost:6379
JWT_SECRET=your-super-secret-key
OPENAI_API_KEY=sk-xxx
HEYGEN_API_KEY=xxx
ELEVENLABS_API_KEY=xxx
```

---

## 📁 Proje Yapısı

```
img-cv/
├── backend/                    # Python FastAPI Backend
│   ├── app/
│   │   ├── api/v1/endpoints/  # API endpoint'leri
│   │   ├── core/              # Config, security, database
│   │   ├── models/            # SQLAlchemy modelleri
│   │   ├── schemas/           # Pydantic şemaları
│   │   ├── services/          # İş mantığı
│   │   └── utils/             # Yardımcı fonksiyonlar
│   ├── tests/                 # Test dosyaları
│   ├── alembic/               # Database migrations
│   └── requirements.txt
│
├── frontend/                   # Next.js Frontend
│   ├── src/
│   │   ├── app/               # Next.js App Router
│   │   ├── components/        # React bileşenleri
│   │   ├── lib/               # Utility fonksiyonlar
│   │   ├── services/          # API servisleri
│   │   ├── store/             # Zustand state management
│   │   └── types/             # TypeScript tipleri
│   └── public/                # Statik dosyalar
│
├── mobile/                     # React Native Mobile App
│   ├── src/
│   │   ├── screens/           # Ekran bileşenleri
│   │   ├── components/        # Paylaşılan bileşenler
│   │   ├── navigation/        # React Navigation
│   │   ├── services/          # API servisleri
│   │   └── store/             # State management
│   └── assets/                # Görseller, fontlar
│
├── ai-pipeline/                # AI Servisleri
│   ├── cv_parser/             # CV parsing modülü
│   ├── image_processor/       # Görüntü İşleme (OCR, Layout)
│   │   ├── ocr.py            # Multi-engine OCR (Tesseract, EasyOCR, PaddleOCR)
│   │   ├── preprocessor.py   # Görüntü ön işleme
│   │   ├── layout_analyzer.py # Belge yapısı analizi
│   │   └── document_processor.py # Ana işlem orkestratörü
│   ├── video_generator/       # Video oluşturma (HeyGen, D-ID)
│   ├── matching_engine/       # AI eşleştirme (pgvector)
│   ├── llm/                   # Ollama, LangChain, LangGraph
│   └── tts/                   # Text-to-Speech (ElevenLabs, local)
│
├── docs/                       # Dokümantasyon
│   ├── api/                   # API dokümantasyonu
│   ├── architecture/          # Mimari dökümanlar
│   └── guides/                # Kullanım kılavuzları
│
├── infrastructure/             # Altyapı dosyaları
│   ├── docker/                # Docker dosyaları
│   ├── k8s/                   # Kubernetes manifests
│   └── scripts/               # Deployment scriptleri
│
└── .github/workflows/          # CI/CD pipelines
```

---

## 📚 API Dokümantasyonu

API dokümantasyonuna erişmek için:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Detaylı Dokümantasyon**: [docs/api/](docs/api/)

### Temel Endpoint'ler

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/api/v1/auth/register` | Yeni kullanıcı kaydı |
| POST | `/api/v1/auth/login` | Kullanıcı girişi |
| POST | `/api/v1/cv/upload` | CV yükleme |
| POST | `/api/v1/video/generate` | Video avatar oluşturma |
| GET | `/api/v1/jobs` | İş ilanları listesi |
| POST | `/api/v1/applications` | İş başvurusu |

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını okuyun.

1. Fork'layın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit'leyin (`git commit -m 'Add amazing feature'`)
4. Push'layın (`git push origin feature/amazing-feature`)
5. Pull Request açın

---

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 📞 İletişim

- **Deep Room AI** - [@deeproomai](https://twitter.com/deeproomai)
- **Email** - info@deeproom.ai
- **Website** - [https://deeproom.ai](https://deeproom.ai)

---

<div align="center">

**Deep Room AI** tarafından ❤️ ile geliştirilmektedir.

</div>
