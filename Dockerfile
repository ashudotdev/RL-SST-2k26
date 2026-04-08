FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Install package in editable mode
RUN pip install -e .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
