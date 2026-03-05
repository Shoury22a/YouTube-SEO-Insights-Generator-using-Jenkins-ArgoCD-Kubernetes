pipeline {
    agent any

    environment {
        DOCKER_IMAGE    = "shoury22a/youtube-seo-app"
        DOCKER_TAG      = "${env.BUILD_ID}"
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/Shoury22a/YouTube-SEO-Insights-Generator-using-Jenkins-ArgoCD-Kubernetes.git'
            }
        }

        stage('Lint') {
            steps {
                // Runs ruff linter in a temporary Python container. Uses the host Docker socket.
                sh 'docker run --rm -v $(pwd):/app -w /app python:3.11-slim sh -c "pip install --quiet ruff && ruff check . || true"'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh "docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} ."
                sh "docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_IMAGE}:latest"
            }
        }

        stage('Push to Docker Hub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh "echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin"
                    sh "docker push ${DOCKER_IMAGE}:${DOCKER_TAG}"
                    sh "docker push ${DOCKER_IMAGE}:latest"
                }
            }
        }

        stage('Update K8s Manifests (GitOps)') {
            steps {
                sh "sed -i 's|image: ${DOCKER_IMAGE}:.*|image: ${DOCKER_IMAGE}:${DOCKER_TAG}|' k8s/deployment.yaml"
                sh 'git config user.email "jenkins@ci.local"'
                sh 'git config user.name "Jenkins CI"'
                sh 'git add k8s/deployment.yaml'
                sh "git commit -m 'ci: Deploy image version ${DOCKER_TAG}' || echo 'No changes to commit'"
                sh 'git push origin main || echo "Push skipped (no changes)"'
            }
        }
    }

    post {
        always {
            sh "docker logout || true"
            cleanWs()
        }
        success {
            echo "✅ Build ${DOCKER_TAG} built, pushed, and deployed via GitOps!"
        }
        failure {
            echo "❌ Pipeline failed. Check the stage logs above for the exact error."
        }
    }
}
