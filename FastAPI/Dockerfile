FROM python:3.12-slim
RUN groupadd -r groupfastapi && useradd -r -g groupfastapi userfastapi
RUN pip install --upgrade pip
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
USER userfastapi
CMD ["python", "main.py"]
