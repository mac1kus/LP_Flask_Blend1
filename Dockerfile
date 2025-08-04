FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install GLPK and verify installation
RUN apt-get update && \
    apt-get install -y glpk-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "GLPK installed:" && \
    which glpsol && \
    glpsol --version

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Use Render's PORT environment variable
ENV PORT=10000
EXPOSE $PORT

# Run the app
CMD ["python", "app.py"]
