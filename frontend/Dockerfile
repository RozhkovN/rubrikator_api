# Этап сборки
FROM oven/bun:latest AS builder

WORKDIR /app

# Принимаем переменные окружения как build arguments
ARG VITE_API_URL=http://localhost:8800/

# Экспортируем как переменные окружения для bun
ENV VITE_API_URL=$VITE_API_URL

COPY . .

RUN bun install --frozen-lockfile
RUN bun run build

# Этап деплоя с nginx
FROM nginx:alpine

# Копируем собранные файлы фронта
COPY --from=builder /app/dist /usr/share/nginx/html

# Копируем конфигурацию nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Создаем директорию для Let's Encrypt сертификатов
RUN mkdir -p /etc/nginx/ssl /var/www/certbot

EXPOSE 3800

CMD ["nginx", "-g", "daemon off;"]