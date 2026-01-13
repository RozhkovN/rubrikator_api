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


# События жизненного цикла
@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте приложения"""
    global classifier
    
    logger.info("🚀 Запуск API сервера...")
    
    try:
        classifier = ComplaintClassifier()
        model_path = root_dir / "models" / "classifier.pkl"
        
        if not model_path.exists():
            logger.error(f"❌ Модель не найдена: {model_path}")
            logger.error("Запустите сначала: python scripts/train.py")
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        classifier.load(str(model_path))
        logger.info("✅ Классификатор загружен успешно")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        raise


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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8800))
    uvicorn.run(app, host="0.0.0.0", port=port)
