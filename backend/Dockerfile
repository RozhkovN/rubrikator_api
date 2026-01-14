# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаем директории для данных и моделей
RUN mkdir -p data models

# Открываем порт
EXPOSE 8800

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PORT=8800

# Команда запуска
CMD ["python", "run_api.py"]
