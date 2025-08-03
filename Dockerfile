FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y glpk-utils

COPY . .

ENV PORT=5000
EXPOSE $PORT

CMD ["python", "app.py"]
