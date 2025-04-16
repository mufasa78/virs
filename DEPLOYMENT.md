# Deployment Guide for Deep Learning Video Image Repair System

This guide provides instructions for deploying the Deep Learning Video Image Repair System using either Flask or Streamlit, with options for local deployment or using Docker.

## Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended for faster processing)
- Docker and Docker Compose (optional, for containerized deployment)

## Local Deployment

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Model Checkpoint

Make sure you have a trained model checkpoint in the `checkpoints` directory. If you don't have a trained model yet, follow the training instructions in the main README.

### 3. Flask Web Application

To run the Flask web application:

```bash
python app.py
```

The application will be available at http://localhost:5000

### 4. Streamlit Application

To run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The application will be available at http://localhost:8501

## Docker Deployment

### 1. Build and Run with Docker Compose

To deploy both Flask and Streamlit applications using Docker Compose:

```bash
docker-compose up -d
```

This will start:
- Flask application at http://localhost:5000
- Streamlit application at http://localhost:8501

### 2. Build and Run Individual Containers

To build and run only the Flask application:

```bash
docker build -f Dockerfile.flask -t video-repair-flask .
docker run -p 5000:5000 -v $(pwd)/checkpoints:/app/checkpoints video-repair-flask
```

To build and run only the Streamlit application:

```bash
docker build -f Dockerfile.streamlit -t video-repair-streamlit .
docker run -p 8501:8501 -v $(pwd)/checkpoints:/app/checkpoints video-repair-streamlit
```

## Production Deployment

For production deployment, consider the following:

### 1. HTTPS Configuration

For Flask, you can use a reverse proxy like Nginx to handle HTTPS:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 2. Authentication

Add authentication to protect your application. For Flask, you can use Flask-Login or similar libraries.

### 3. Cloud Deployment

For cloud deployment, you can use:

- **AWS**: Deploy using Elastic Beanstalk or ECS
- **Google Cloud**: Deploy using App Engine or Cloud Run
- **Azure**: Deploy using App Service or Container Instances
- **Heroku**: Deploy using the Heroku container registry

Example for deploying to Google Cloud Run:

```bash
# Build the container
docker build -f Dockerfile.flask -t gcr.io/your-project/video-repair-flask .

# Push to Google Container Registry
docker push gcr.io/your-project/video-repair-flask

# Deploy to Cloud Run
gcloud run deploy video-repair --image gcr.io/your-project/video-repair-flask --platform managed
```

### 4. Resource Considerations

- Ensure your deployment environment has sufficient GPU resources for model inference
- Configure appropriate memory limits based on your video processing needs
- Consider using a queue system for processing large videos asynchronously

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce the frame size or sequence length in the configuration
2. **Slow Processing**: Ensure GPU acceleration is properly configured
3. **File Upload Issues**: Check the maximum file size configuration in your web server

### Logs

- Flask logs are available in the console or in your web server logs
- Streamlit logs are available in the console
- Docker logs can be viewed with `docker logs <container_id>`

## Monitoring

For production deployments, consider adding:

- Prometheus metrics for monitoring system performance
- Grafana dashboards for visualizing metrics
- Error tracking with Sentry or similar services
