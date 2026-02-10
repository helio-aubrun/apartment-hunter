# Docker Quick Start Guide

Get your Apartment Hunter application running in Docker in under 5 minutes!

## Prerequisites

Install Docker Desktop:
- **Windows**: https://docs.docker.com/desktop/install/windows-install/
- **Mac**: https://docs.docker.com/desktop/install/mac-install/
- **Linux**: https://docs.docker.com/desktop/install/linux-install/

## Step 1: Build and Run

Open a terminal in the project directory and run:

```bash
docker-compose up -d
```

That's it! The application is now running.

## Step 2: Access the Application

Open your browser and go to:

**http://localhost:5000**

## Step 3: Use the Application

### Via Web Interface:
1. Click "Load Dataset"
2. Click "Train Selected Models"
3. Wait for training to complete
4. Use "Make Predictions" to test

### Via API:
```bash
# Check status
curl http://localhost:5000/api/status

# Load data
curl -X POST http://localhost:5000/api/load-data \
  -H "Content-Type: application/json" \
  -d "{\"filepath\": \"clean_dataset.csv\", \"target\": \"buy_price\"}"

# Train models
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d "{\"models\": [\"RandomForest\"]}"
```

## Useful Commands

### View logs
```bash
docker-compose logs -f
```

### Stop application
```bash
docker-compose down
```

### Restart application
```bash
docker-compose restart
```

### Rebuild after code changes
```bash
docker-compose up -d --build
```

## Troubleshooting

### "Port 5000 is already in use"
```bash
# Stop other services using port 5000, or change port in docker-compose.yml:
ports:
  - "8080:5000"  # Use 8080 instead
```

### "Cannot connect to Docker daemon"
- Make sure Docker Desktop is running
- On Windows: Check Docker Desktop icon in system tray

### "Container unhealthy"
```bash
# Check logs
docker-compose logs

# Restart
docker-compose restart
```

## Next Steps

- Read [README_DOCKER.md](README_DOCKER.md) for detailed documentation
- Read [README_FLASK.md](README_FLASK.md) for API documentation
- Check logs: `docker-compose logs -f`

## That's It!

Your ML model serving application is now running in a container! 🚀