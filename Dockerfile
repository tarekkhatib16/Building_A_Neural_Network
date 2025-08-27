FROM python:3.13-slim

WORKDIR /app 

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/

COPY src/ /app/src/

COPY model_store/ /app/model_store/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]