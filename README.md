# Анализатор Жалоб - Рубрикатор

Система автоматической классификации жалоб по 20 рубрикаторам с использованием ML.

## Архитектура

**Гибридный подход:**
- Sentence Transformers (paraphrase-multilingual-mpnet-base-v2) для семантического анализа
- Анализ ключевых слов для уточнения категории
- Взвешенная комбинация методов для максимальной точности

## Структура проекта

```
rubrikator_api/
├── config/
│   └── rubrics.py          # Описание всех 20 рубрикаторов
├── src/
│   ├── model.py            # Класс классификатора
│   ├── preprocessor.py     # Предобработка текста
│   └── utils.py            # Вспомогательные функции
├── scripts/
│   ├── generate_data.py    # Генерация синтетических данных
│   ├── train.py            # Обучение/подготовка модели
│   └── predict.py          # Тестирование классификации
├── data/                   # Данные для обучения
├── models/                 # Сохраненные модели
├── requirements.txt
└── README.md
```

## Установка

```bash
# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Установить зависимости
pip install -r requirements.txt
```

## Использование

### 1. Генерация обучающих данных
```bash
python scripts/generate_data.py
```

### 2. Обучение модели
```bash
python scripts/train.py
```

### 3. Классификация жалобы
```bash
python scripts/predict.py "Текст жалобы"
```

## Ожидаемая точность

- Базовая модель: 75-85%
- С ключевыми словами: 85-92%
- После fine-tuning: 92-97%

## Roadmap

- [x] Базовая архитектура
- [x] Sentence Transformers классификация
- [x] Анализ ключевых слов
- [ ] REST API (FastAPI)
- [ ] База данных (PostgreSQL)
- [ ] Веб-интерфейс
- [ ] Continuous learning
