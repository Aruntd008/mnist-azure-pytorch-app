FROM nginx:alpine

COPY . /usr/share/nginx/html/

# Replace API URL at runtime using environment variable
RUN echo 'sed -i "s|https://your-aci-instance.azurecontainer.io|$API_URL|g" /usr/share/nginx/html/script.js && nginx -g "daemon off;"' > /docker-entrypoint.sh

RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/bin/sh", "/docker-entrypoint.sh"]