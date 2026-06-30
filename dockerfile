# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose BentoML port
EXPOSE 3000

# Start BentoML Service
CMD ["bentoml", "serve", "service:StudentDropoutService", "--host", "0.0.0.0", "--port", "3000"]
