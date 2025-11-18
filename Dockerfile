FROM python:3.12-slim

WORKDIR /app
ENV PYDONTWRITEBYTECODE=1

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

EXPOSE 5000

CMD ["python3", "app.py"]
