FROM python:alpine

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

# Copy model module from parent directory
COPY ../model/model.py /app/model/

VOLUME /app/model

# Create a startup script
RUN echo '#!/bin/bash\n\
\n\
# Check if model file exists\n\
if [ ! -f /app/model/mnist_pytorch.pt ]; then\n\
  echo "Model file not found, checking for mounted model..."\n\
  # Will be mounted at runtime\n\
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