# 🚀 YouTube SEO Project: CI/CD Cheat Sheet

## ⛓️ The 4 Stages of Your CI/CD Pipeline
Here is how your industry-standard pipeline is structured:

1.  **STAGE 1: BUILD (GitHub Actions)**
    *   **Action**: GitHub triggers on your push.
    *   **Outcome**: Compiles your Python code and packs it into a **Docker Image**. This ensures your app runs exactly the same everywhere.

2.  **STAGE 2: TEST (GitHub Actions)**
    *   **Action**: (Standard Practice) Running `ruff` for linting or `pytest` for unit tests.
    *   **Outcome**: If a test fails, the pipeline **stops**! This prevents broken code from ever reaching your users.

3.  **STAGE 3: RELEASE (Docker Hub)**
    *   **Action**: The built image is "tagged" and pushed to Docker Hub.
    *   **Outcome**: This creates a permanent version of your app (a "Release"). You can always go back to an older release if something goes wrong.

4.  **STAGE 4: DEPLOY (Argo CD & Kubernetes)**
    *   **Action**: Argo CD detects the new release and tells Kubernetes to "Sync."
    *   **Outcome**: Kubernetes performs a **Rolling Update**, replacing old containers with new ones without any downtime.

---

This guide contains all the commands used to set up and run your pipeline, ordered by their execution flow.

---

## 🛠 Phase 1: Infrastructure Setup (Kubernetes)
Initial setup for the cluster namespaces and application manifests.

```powershell
# Create namespace for Argo CD
kubectl create namespace argocd

# Apply Application Manifests (Deployment & Service)
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

---

## 🏗 Phase 2: CI (Continuous Integration - GitHub Actions)
This happens automatically on every push to `main`. 

1.  **Github Repo Settings** -> Secrets -> Actions:
    *   `DOCKERHUB_USERNAME`: Your Docker Hub ID
    *   `DOCKERHUB_TOKEN`: Security token from Docker Hub
2.  **Manual Docker Build (if needed locally)**:
    ```powershell
    docker build -t yourusername/youtube-seo-app:latest .
    docker push yourusername/youtube-seo-app:latest
    ```

---

## 🐙 Phase 3: CD (Continuous Deployment - Argo CD)
This is what updates your app whenever the Docker image changes.

### 1. Installation (Run once)
```powershell
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

### 2. Access the UI (Most Important)
Run this command in a dedicated terminal and **keep it open**:
```powershell
kubectl port-forward svc/argocd-server -n argocd 8080:443
```
> [!TIP]
> Go to **`https://localhost:8080`** after running this.

### 3. Login Credentials
*   **Username**: `admin`
*   **Initial Password Retrieval**:
    ```powershell
    kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | ForEach-Object { [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($_)) }
    ```

---

## 🔍 Phase 4: Monitoring & Verification
Check if everything is running correctly.

```powershell
# Check Pods (Your App)
kubectl get pods

# Check Services (To see the Port 80 for your App)
kubectl get svc

# Check Logs for Troubleshooting
kubectl logs -f deployment/youtube-seo-app

# Restart App Deployment (Forces a pull of new image)
kubectl rollout restart deployment youtube-seo-app
```

---

## 📖 Summary of "Daily Use" Commands
1.  **Start Access**: `kubectl port-forward svc/argocd-server -n argocd 8080:443`
2.  **Open Website**: [https://localhost:8080](https://localhost:8080)
3.  **Click Sync**!
