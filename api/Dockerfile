FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory
RUN mkdir -p /app/model

# Copy API code
COPY . /app/

# Add model directory to Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Copy model module
COPY ../model/model.py /app/model/

# The actual model file will be downloaded from Azure ML during startup
# We'll add a script to handle this

# Create script to download model at startup
RUN echo '#!/bin/bash\n\
if [ ! -f /app/model/mnist_pytorch.pt ]; then\n\
  echo "Downloading model from Azure ML..."\n\
  # In production, you would add code here to download the model from Azure ML\n\
  # For now, we\'ll assume the model is mounted or copied in during deployment\n\
  echo "Model file not found. Using placeholder or mounted model."\n\
fi\n\
\n\
# Start the API\n\
exec uvicorn app:app --host 0.0.0.0 --port 8000\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Run the application
CMD ["/app/start.sh"]