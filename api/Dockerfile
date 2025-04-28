FROM python:3.13-alpine


WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model code
COPY . /app/api/
RUN mkdir -p /app/model

# Copy the model file (this will be replaced in the workflow)
COPY ../model/mnist_pytorch.pt /app/model/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]