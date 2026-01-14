# Этап сборки
FROM node:20-alpine AS builder

WORKDIR /app

# Принимаем переменные окружения как build arguments
ARG VITE_API_URL=http://80.253.19.93:8800

# Экспортируем как переменные окружения для vite
ENV VITE_API_URL=$VITE_API_URL

# Копируем package файлы
COPY package.json package-lock.json* bun.lock* ./

# Устанавливаем зависимости
RUN npm ci || npm install

# Копируем остальные файлы (исключая node_modules)
COPY tsconfig*.json vite.config.ts index.html ./
COPY public ./public
COPY src ./src

# Собираем проект с использованием npx
RUN npx tsc -b && npx vite build

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