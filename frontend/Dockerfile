FROM nginx:alpine

COPY . /usr/share/nginx/html/

# Replace API URL at runtime using environment variable
# Make sure to escape any special characters to avoid sed errors
RUN echo '#!/bin/sh\n\
# Ensure API_URL is set\n\
if [ -z "$API_URL" ]; then\n\
  echo "Warning: API_URL environment variable is not set. Using default value."\n\
  API_URL="http://localhost:8000"\n\
fi\n\
\n\
# Replace the API URL in the script.js file\n\
echo "Setting API URL to: $API_URL"\n\
sed -i "s|https://your-aci-instance.azurecontainer.io|$API_URL|g" /usr/share/nginx/html/script.js\n\
\n\
# Start nginx in foreground\n\
exec nginx -g "daemon off;"\n\
' > /docker-entrypoint.sh

RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/bin/sh", "/docker-entrypoint.sh"]