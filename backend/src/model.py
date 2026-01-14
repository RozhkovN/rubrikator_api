"""
Основная модель классификатора жалоб.

Использует гибридный подход:
1. Sentence Transformers для семантического анализа
2. Анализ ключевых слов для уточнения
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.rubrics import RUBRICS, get_rubric_by_id
from src.preprocessor import normalize_text, calculate_keyword_score


class ComplaintClassifier:
    """Классификатор жалоб на основе Sentence Transformers"""
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        use_keywords: bool = True,
        keyword_weight: float = 0.3
    ):
        """
        Инициализация классификатора.
        
        Args:
            model_name: название модели Sentence Transformers
            use_keywords: использовать ли анализ ключевых слов
            keyword_weight: вес ключевых слов (0-1), остальное - семантика
        """
        self.model_name = model_name
        self.use_keywords = use_keywords
        self.keyword_weight = keyword_weight
        self.semantic_weight = 1 - keyword_weight
        
        self.model: Optional[SentenceTransformer] = None
        self.rubric_embeddings: Optional[np.ndarray] = None
        self.rubrics = RUBRICS
        
        print(f"🔧 Инициализация классификатора...")
        print(f"   Модель: {model_name}")
        print(f"   Ключевые слова: {'Да' if use_keywords else 'Нет'}")
        if use_keywords:
            print(f"   Веса: семантика={self.semantic_weight:.1f}, ключевые слова={self.keyword_weight:.1f}")
    
    def load_model(self):
        """Загрузка модели Sentence Transformers"""
        if self.model is None:
            print(f"📥 Загрузка модели {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("✓ Модель загружена")
    
    def prepare_rubric_texts(self) -> List[str]:
        """
        Подготовка текстовых представлений рубрикаторов.
        
        Returns:
            Список текстов для векторизации
        """
        texts = []
        for rubric in self.rubrics:
            # Объединяем описание и ключевые слова для лучшего представления
            text = f"{rubric['description']}. Ключевые слова: {', '.join(rubric['keywords'])}"
            texts.append(text)
        return texts
    
    def train(self, save_path: str = "models/classifier.pkl"):
        """
        Подготовка классификатора (создание эмбеддингов рубрикаторов).
        
        Args:
            save_path: путь для сохранения обученной модели
        """
        print("\n🎯 Подготовка классификатора...")
        
        # Загружаем модель, если не загружена
        self.load_model()
        
        # Создаем текстовые представления рубрикаторов
        print("📝 Подготовка описаний рубрикаторов...")
        rubric_texts = self.prepare_rubric_texts()
        
        # Создаем эмбеддинги
        print("🔄 Создание векторных представлений...")
        self.rubric_embeddings = self.model.encode(
            rubric_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Сохраняем модель
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.rubric_embeddings,
                'model_name': self.model_name,
                'use_keywords': self.use_keywords,
                'keyword_weight': self.keyword_weight
            }, f)
        
        print(f"✓ Классификатор сохранен в {save_path}")
        print(f"✓ Создано {len(self.rubric_embeddings)} эмбеддингов рубрикаторов")
    
    def load(self, load_path: str = "models/classifier.pkl"):
        """
        Загрузка обученного классификатора.
        
        Args:
            load_path: путь к сохраненной модели
        """
        print(f"📂 Загрузка классификатора из {load_path}...")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.rubric_embeddings = data['embeddings']
        self.model_name = data['model_name']
        self.use_keywords = data.get('use_keywords', True)
        self.keyword_weight = data.get('keyword_weight', 0.3)
        self.semantic_weight = 1 - self.keyword_weight
        
        # Загружаем модель для предсказаний
        self.load_model()
        
        print("✓ Классификатор загружен")
    
    def _calculate_semantic_scores(self, text: str) -> np.ndarray:
        """
        Расчет семантических оценок через cosine similarity.
        
        Args:
            text: текст жалобы
            
        Returns:
            Массив оценок для каждого рубрикатора
        """
        # Создаем эмбеддинг текста
        text_embedding = self.model.encode([text], convert_to_numpy=True)
        
        # Считаем cosine similarity со всеми рубрикаторами
        similarities = cosine_similarity(text_embedding, self.rubric_embeddings)[0]
        
        # Нормализуем в диапазон [0, 1]
        # Cosine similarity уже в [-1, 1], но обычно положительный
        scores = (similarities + 1) / 2
        
        return scores
    
    def _calculate_keyword_scores(self, text: str) -> np.ndarray:
        """
        Расчет оценок на основе ключевых слов.
        
        Args:
            text: текст жалобы
            
        Returns:
            Массив оценок для каждого рубрикатора
        """
        scores = np.zeros(len(self.rubrics))
        
        for i, rubric in enumerate(self.rubrics):
            scores[i] = calculate_keyword_score(text, rubric['keywords'])
        
        return scores
    
    def predict(
        self,
        text: str,
        top_k: int = 3,
        return_scores: bool = True
    ) -> Dict:
        """
        Классификация жалобы.
        
        Args:
            text: текст жалобы
            top_k: количество топ результатов
            return_scores: возвращать ли детальные оценки
            
        Returns:
            Словарь с результатами классификации
        """
        if self.model is None or self.rubric_embeddings is None:
            raise ValueError("Модель не загружена. Используйте load() или train()")
        
        # Нормализуем текст
        text_normalized = normalize_text(text)
        
        # Рассчитываем семантические оценки
        semantic_scores = self._calculate_semantic_scores(text)
        
        # Рассчитываем оценки по ключевым словам
        if self.use_keywords:
            keyword_scores = self._calculate_keyword_scores(text)
            # Комбинируем оценки
            final_scores = (
                self.semantic_weight * semantic_scores +
                self.keyword_weight * keyword_scores
            )
        else:
            keyword_scores = np.zeros(len(self.rubrics))
            final_scores = semantic_scores
        
        # Находим топ-k результатов
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        # Формируем результаты
        predictions = []
        for idx in top_indices:
            rubric = self.rubrics[idx]
            predictions.append({
                'rubric_id': rubric['id'],
                'rubric_name': rubric['description'],  # Используем полное описание
                'short_name': rubric['name'],  # Краткое название для справки
                'confidence': float(final_scores[idx]),
                'semantic_score': float(semantic_scores[idx]) if return_scores else None,
                'keyword_score': float(keyword_scores[idx]) if return_scores else None
            })
        
        result = {
            'text': text,
            'predictions': predictions,
            'best_match': predictions[0] if predictions else None
        }
        
        return result
    
    def predict_batch(
        self,
        texts: List[str],
        top_k: int = 1
    ) -> List[Dict]:
        """
        Пакетная классификация жалоб.
        
        Args:
            texts: список текстов жалоб
            top_k: количество топ результатов
            
        Returns:
            Список результатов для каждого текста
        """
        results = []
        for text in texts:
            result = self.predict(text, top_k=top_k, return_scores=False)
            results.append(result)
        return results
