"""
FastAPI приложение для классификации жалоб.

Запуск:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8800
"""

import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import logging

from src.model import ComplaintClassifier

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="Рубрикатор Жалоб API",
    description="API для автоматической классификации жалоб по 20 рубрикаторам",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная для классификатора
classifier: Optional[ComplaintClassifier] = None


# Модели данных
class ComplaintRequest(BaseModel):
    """Запрос на классификацию жалобы"""
    text: str = Field(..., description="Текст жалобы", min_length=10)
    top_k: int = Field(1, description="Количество топ результатов", ge=1, le=5)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Банк заблокировал мою карту без предупреждения согласно 161-ФЗ",
                "top_k": 1
            }
        }


class PredictionItem(BaseModel):
    """Один результат классификации"""
    rubric_id: int
    rubric_name: str
    short_name: str
    confidence: float


class ComplaintResponse(BaseModel):
    """Ответ с результатами классификации"""
    text: str
    best_match: PredictionItem
    all_predictions: Optional[List[PredictionItem]] = None


class HealthResponse(BaseModel):
    """Ответ health check"""
    status: str
    model_loaded: bool


class TrainRequest(BaseModel):
    """Запрос на обучение модели"""
    model_name: str = Field(
        "paraphrase-multilingual-mpnet-base-v2",
        description="Название модели Sentence Transformers"
    )
    use_keywords: bool = Field(True, description="Использовать ли анализ ключевых слов")
    keyword_weight: float = Field(0.3, description="Вес ключевых слов (0-1)", ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "paraphrase-multilingual-mpnet-base-v2",
                "use_keywords": True,
                "keyword_weight": 0.3
            }
        }


class TrainResponse(BaseModel):
    """Ответ на запрос обучения"""
    status: str
    message: str
    model_path: str


# События жизненного цикла
@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте приложения"""
    global classifier
    
    logger.info("🚀 Запуск API сервера...")
    
    try:
        model_path = root_dir / "models" / "classifier.pkl"
        
        if not model_path.exists():
            logger.warning(f"⚠️  Модель не найдена: {model_path}")
            logger.warning("Используйте POST /train для обучения модели")
            classifier = None
        else:
            classifier = ComplaintClassifier()
            classifier.load(str(model_path))
            logger.info("✅ Классификатор загружен успешно")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        logger.warning("API запущен без модели. Используйте POST /train для обучения")
        classifier = None


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка ресурсов при остановке"""
    logger.info("🛑 Остановка API сервера...")


# Эндпоинты
@app.get("/", tags=["General"])
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Рубрикатор Жалоб API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Проверка состояния API"""
    return {
        "status": "ok",
        "model_loaded": classifier is not None
    }


@app.post("/classify", response_model=ComplaintResponse, tags=["Classification"])
async def classify_complaint(request: ComplaintRequest):
    """
    Классификация жалобы.
    
    Принимает текст жалобы и возвращает наиболее подходящий рубрикатор.
    
    - **text**: Текст жалобы (минимум 10 символов)
    - **top_k**: Количество топ результатов (по умолчанию 1)
    
    Возвращает:
    - **best_match**: Лучшее совпадение с рубрикатором
    - **all_predictions**: Все топ-k результатов (если top_k > 1)
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Классификатор не загружен. Сервер не готов к работе."
        )
    
    try:
        # Классифицируем жалобу
        result = classifier.predict(
            text=request.text,
            top_k=request.top_k,
            return_scores=False
        )
        
        # Формируем ответ
        predictions = []
        for pred in result['predictions']:
            predictions.append(PredictionItem(
                rubric_id=pred['rubric_id'],
                rubric_name=pred['rubric_name'],
                short_name=pred.get('short_name', ''),
                confidence=round(pred['confidence'], 4)
            ))
        
        response = ComplaintResponse(
            text=request.text,
            best_match=predictions[0],
            all_predictions=predictions if request.top_k > 1 else None
        )
        
        logger.info(f"✅ Классифицировано: {predictions[0].short_name} ({predictions[0].confidence:.2%})")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Ошибка классификации: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при классификации: {str(e)}"
        )


@app.post("/classify/batch", tags=["Classification"])
async def classify_batch(complaints: List[str]):
    """
    Пакетная классификация жалоб.
    
    Принимает список текстов жалоб и возвращает результаты для каждой.
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Классификатор не загружен"
        )
    
    if len(complaints) > 100:
        raise HTTPException(
            status_code=400,
            detail="Максимум 100 жалоб за один запрос"
        )
    
    try:
        results = classifier.predict_batch(complaints, top_k=1)
        
        responses = []
        for result in results:
            pred = result['best_match']
            responses.append({
                "text": result['text'],
                "rubric_id": pred['rubric_id'],
                "rubric_name": pred['rubric_name'],
                "short_name": pred.get('short_name', ''),
                "confidence": round(pred['confidence'], 4)
            })
        
        return {"results": responses, "count": len(responses)}
        
    except Exception as e:
        logger.error(f"❌ Ошибка пакетной классификации: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при классификации: {str(e)}"
        )


@app.post("/train", response_model=TrainResponse, tags=["Model Management"])
async def train_model(request: TrainRequest):
    """
    Обучение (подготовка) модели классификатора.
    
    Создает векторные представления рубрикаторов и сохраняет модель.
    После обучения модель автоматически загружается в memory.
    
    - **model_name**: Название модели Sentence Transformers
    - **use_keywords**: Использовать ли анализ ключевых слов
    - **keyword_weight**: Вес ключевых слов (0-1)
    
    Примечание: Процесс может занять несколько минут при первом запуске
    (загрузка модели из интернета).
    """
    global classifier
    
    try:
        logger.info("🚀 Начинаем обучение модели...")
        logger.info(f"   Модель: {request.model_name}")
        logger.info(f"   Ключевые слова: {request.use_keywords}")
        logger.info(f"   Вес ключевых слов: {request.keyword_weight}")
        
        # Создаем новый классификатор с заданными параметрами
        new_classifier = ComplaintClassifier(
            model_name=request.model_name,
            use_keywords=request.use_keywords,
            keyword_weight=request.keyword_weight
        )
        
        # Путь для сохранения модели
        model_path = root_dir / "models" / "classifier.pkl"
        
        # Обучаем (создаем эмбеддинги)
        new_classifier.train(save_path=str(model_path))
        
        # Заменяем глобальный классификатор на новый
        classifier = new_classifier
        
        logger.info("✅ Обучение завершено успешно")
        
        return TrainResponse(
            status="success",
            message="Модель успешно обучена и загружена",
            model_path=str(model_path)
        )
        
    except Exception as e:
        logger.error(f"❌ Ошибка обучения модели: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обучении модели: {str(e)}"
        )


@app.get("/model/info", tags=["Model Management"])
async def get_model_info():
    """
    Получение информации о текущей загруженной модели.
    
    Возвращает параметры и статус модели.
    """
    if classifier is None:
        return {
            "loaded": False,
            "message": "Модель не загружена"
        }
    
    return {
        "loaded": True,
        "model_name": classifier.model_name,
        "use_keywords": classifier.use_keywords,
        "keyword_weight": classifier.keyword_weight,
        "semantic_weight": classifier.semantic_weight,
        "rubrics_count": len(classifier.rubrics)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8800))
    uvicorn.run(app, host="0.0.0.0", port=port)
