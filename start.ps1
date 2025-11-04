# Local RAG Quick Start Script for Windows
# This script helps you set up and run the Local RAG application

Write-Host "🚀 Local RAG System - Quick Start" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
$dockerRunning = docker info 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Docker is running" -ForegroundColor Green

# Check if docker-compose exists
if (Test-Path "docker-compose.yml") {
    Write-Host "✅ Found docker-compose.yml" -ForegroundColor Green
} else {
    Write-Host "❌ docker-compose.yml not found in current directory" -ForegroundColor Red
    exit 1
}

# Create .env file if it doesn't exist
if (-not (Test-Path "server\.env")) {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item "server\.env.example" "server\.env"
    Write-Host "✅ Created server\.env (you can customize it)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting services with Docker Compose..." -ForegroundColor Yellow
Write-Host ""

# Start docker-compose
docker-compose up -d

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Services started successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📝 Next steps:" -ForegroundColor Cyan
    Write-Host "   1. Wait ~30 seconds for services to initialize"
    Write-Host "   2. Pull Ollama models (first time only):"
    Write-Host "      docker exec -it ollama ollama pull mxbai-embed-large"
    Write-Host "      docker exec -it ollama ollama pull llama3.2"
    Write-Host "   3. Open your browser to: http://localhost:8501"
    Write-Host ""
    Write-Host "🛠️  Useful commands:" -ForegroundColor Cyan
    Write-Host "   View logs:        docker-compose logs -f"
    Write-Host "   Stop services:    docker-compose down"
    Write-Host "   Restart:          docker-compose restart"
    Write-Host ""
} else {
    Write-Host "❌ Failed to start services" -ForegroundColor Red
    exit 1
}
