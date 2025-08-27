# Use Python 3.13 slim base image for smaller size and faster builds
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first for better Docker layer caching
# This allows pip install to be cached if requirements.txt hasn't changed
COPY requirements.txt .

# Install Python dependencies without caching pip files to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files into the container
COPY app.py /app/
COPY src/ /app/src/
COPY model_store/ /app/model_store/

# Expose port 8000 for the web application
EXPOSE 8000

# Start the application using uvicorn ASGI server
# Bind to all interfaces (0.0.0.0) on port 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]