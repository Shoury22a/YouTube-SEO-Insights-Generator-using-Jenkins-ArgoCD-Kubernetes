pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "shoury22a/youtube-seo-app"
        // This is the ID of the credentials you create in Jenkins (Manage Jenkins -> Credentials)
        DOCKER_REGISTRY_CREDENTIALS_ID = "dockerhub-creds"
        K8S_SECRET_NAME = "youtube-seo-secrets"
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/shourya22a/seo-optimization.git'
            }
        }

        stage('Linting') {
            steps {
                sh 'pip install ruff'
                sh 'ruff check .'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    dockerImage = docker.build("${DOCKER_IMAGE}:${env.BUILD_ID}")
                }
            }
        }

        stage('Push to Registry') {
            steps {
                script {
                    docker.withRegistry('', DOCKER_REGISTRY_CREDENTIALS_ID) {
                        dockerImage.push()
                        dockerImage.push('latest')
                    }
                }
            }
        }

        stage('Deploy to K8s (GitOps)') {
            steps {
                script {
                    // In a true GitOps flow, this step updates the k8s manifests in Git.
                    // ArgoCD then detects the change and syncs the cluster.
                    echo "Updating k8s/deployment.yaml with image tag: ${env.BUILD_ID}"
                    sh "sed -i 's|image: .*|image: ${DOCKER_IMAGE}:${env.BUILD_ID}|' k8s/deployment.yaml"
                    
                    // Commit and push back to Git to trigger ArgoCD
                    sh 'git add k8s/deployment.yaml'
                    sh "git commit -m 'Deploying version ${env.BUILD_ID}'"
                    sh 'git push origin main'
                }
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo "Deployment successful! ArgoCD will now sync the cluster."
        }
        failure {
            echo "Deployment failed. Check Jenkins logs for details."
        }
    }
}
