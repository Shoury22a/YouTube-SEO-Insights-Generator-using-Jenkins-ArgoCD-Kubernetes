# 🚀 YouTube SEO Insights Generator (Enterprise Edition)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-ff4b4b.svg)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/AI-Gemini%202.0%20Flash-orange.svg)](https://deepmind.google/technologies/gemini/)
[![Kubernetes](https://img.shields.io/badge/Orchestration-Kubernetes-326ce5.svg)](https://kubernetes.io/)
[![GitOps](https://img.shields.io/badge/GitOps-ArgoCD-efaf19.svg)](https://argoproj.github.io/cd/)

A professional-grade YouTube SEO tool that generates viral-ready metadata using Google's **Gemini 2.0 Flash**. Designed for high scalability and zero-downtime using a full Enterprise DevOps stack.

---

## 🌟 Key Features

### 🧠 Advanced AI Engine
- **Gemini 2.0 Flash Backend**: High-speed, high-quality generation at zero cost.
- **JSON Native Mode**: Guarantees structured output for titles, tags, and descriptions.
- **Hinglish Support**: Specifically tuned for the Indian creator market, generating conversational Hindi-English mix in Roman script.
- **Smart Summarization**: Handles transcripts up to 25k characters using an automated summarization pipeline.

### 🔍 Robust Competitor Intelligence
- **yt-dlp Integration**: Replaced fragile BeautifulSoup scraping with an industry-standard engine to bypass YouTube's latest bot protections.
- **Keyword Gap Analysis**: AI analyzes competitor metadata to identify hooks and trending keywords.

### 🏗️ Enterprise DevOps Infrastructure
- **Dockerized**: Consistent execution across all environments.
- **Kubernetes Orchestration**: Features self-healing, scaling, and rolling updates for zero downtime.
- **CI/CD via Jenkins**: Automated linting, building, and pushing of images.
- **GitOps with ArgoCD**: Synchronizes your GitHub repository's `k8s/` manifests with the live cluster.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Backend**: Google Generative AI (Gemini 2.0 Flash)
- **Scraper**: yt-dlp
- **CI/CD**: Jenkins, DockerHub
- **Orchestration**: Kubernetes (K8s), ArgoCD

---

## 📐 Architecture Diagram

```mermaid
graph LR
    User([User]) --> Streamlit[Streamlit UI]
    Streamlit --> AI[Gemini 2.0 Flash]
    Streamlit --> Scraper[yt-dlp Scraper]
    
    subgraph DevOps Pipeline
    Jenkins[Jenkins CI] --> Docker[Docker Build]
    Docker --> Registry[DockerHub]
    Registry --> ArgoCD[ArgoCD GitOps]
    ArgoCD --> K8s[Kubernetes Cluster]
    end
    
    GitHub((GitHub)) -- Trigger --> Jenkins
    GitHub -- Sync --> ArgoCD
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- Docker & Kubernetes (Desktop or Minikube)
- A [Google AI Studio](https://aistudio.google.com/app/apikey) API Key.

### 2. Installation
```bash
git clone https://github.com/Shoury22a/YouTube-SEO-Insights-Generator-using-Jenkins-ArgoCD-Kubernetes.git
cd YouTube-SEO-Insights-Generator-using-Jenkins-ArgoCD-Kubernetes
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_api_key_here
```

### 4. Running Locally
```bash
streamlit run app.py
```

---

## ☸️ Kubernetes Deployment

Deploy securely to your local cluster:

1. Create the API secret:
   ```bash
   kubectl create secret generic youtube-seo-secrets --from-literal=google-api-key="your_key"
   ```

2. Apply the manifests:
   ```bash
   kubectl apply -f k8s/
   ```

3. Access the app at: **http://localhost**

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
