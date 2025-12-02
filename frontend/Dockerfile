# frontend/Dockerfile
# Use a lightweight Nginx image to serve the static UI files
FROM nginx:alpine

# Remove default Nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy the custom config and the static files
COPY default.conf /etc/nginx/conf.d/
COPY index.html /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/
COPY script.js /usr/share/nginx/html/

# The Nginx image automatically runs the web server (CMD is not needed)