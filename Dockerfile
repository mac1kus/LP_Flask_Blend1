
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install GLPK before installing requirements
RUN apt-get update && apt-get install -y glpk-utils

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port for Flask
ENV PORT=5000
EXPOSE $PORT

# Run the app
CMD ["python", "app.py"]
