"""
Основная модель классификатора жалоб.

Использует гибридный подход:
1. Sentence Transformers для семантического анализа
2. Анализ ключевых слов для уточнения
3. Примеры жалоб для улучшения эмбеддингов
4. Приоритетные и негативные ключевые слова
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.rubrics import RUBRICS, get_rubric_by_id
from config.response_templates import get_response_template
from src.preprocessor import (
    normalize_text, 
    calculate_keyword_score,
    calculate_advanced_keyword_score,
    extract_law_references,
    extract_organization_mentions
)


class ComplaintClassifier:
    """Классификатор жалоб на основе Sentence Transformers"""
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        use_keywords: bool = True,
        keyword_weight: float = 0.35,
        use_examples: bool = True
    ):
        """
        Инициализация классификатора.
        
        Args:
            model_name: название модели Sentence Transformers
            use_keywords: использовать ли анализ ключевых слов
            keyword_weight: вес ключевых слов (0-1), остальное - семантика
            use_examples: использовать ли примеры для эмбеддингов
        """
        self.model_name = model_name
        self.use_keywords = use_keywords
        self.keyword_weight = keyword_weight
        self.semantic_weight = 1 - keyword_weight
        self.use_examples = use_examples
        
        self.model: Optional[SentenceTransformer] = None
        self.rubric_embeddings: Optional[np.ndarray] = None
        self.example_embeddings: Optional[Dict[int, np.ndarray]] = None
        self.rubrics = RUBRICS
        
        print(f"🔧 Инициализация классификатора...")
        print(f"   Модель: {model_name}")
        print(f"   Ключевые слова: {'Да' if use_keywords else 'Нет'}")
        print(f"   Примеры: {'Да' if use_examples else 'Нет'}")
        if use_keywords:
            print(f"   Веса: семантика={self.semantic_weight:.2f}, ключевые слова={self.keyword_weight:.2f}")
    
    def load_model(self):
        """Загрузка модели Sentence Transformers"""
        if self.model is None:
            print(f"📥 Загрузка модели {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("✓ Модель загружена")
    
    def prepare_rubric_texts(self) -> List[str]:
        """
        Подготовка текстовых представлений рубрикаторов.
        Улучшенная версия с примерами.
        
        Returns:
            Список текстов для векторизации
        """
        texts = []
        for rubric in self.rubrics:
            # Базовое описание
            parts = [rubric['description']]
            
            # Добавляем ключевые слова
            parts.append(f"Ключевые слова: {', '.join(rubric['keywords'])}")
            
            # Добавляем приоритетные ключевые слова
            if 'priority_keywords' in rubric and rubric['priority_keywords']:
                parts.append(f"Важные признаки: {', '.join(rubric['priority_keywords'])}")
            
            # Добавляем примеры для лучшего понимания контекста
            if self.use_examples and 'examples' in rubric and rubric['examples']:
                examples_text = " | ".join(rubric['examples'][:3])  # Берем до 3 примеров
                parts.append(f"Примеры обращений: {examples_text}")
            
            text = ". ".join(parts)
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
        
        # Создаем эмбеддинги рубрикаторов
        print("🔄 Создание векторных представлений рубрикаторов...")
        self.rubric_embeddings = self.model.encode(
            rubric_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Создаем эмбеддинги примеров для каждого рубрикатора
        self.example_embeddings = {}
        if self.use_examples:
            print("🔄 Создание векторных представлений примеров...")
            for rubric in self.rubrics:
                if 'examples' in rubric and rubric['examples']:
                    embeddings = self.model.encode(
                        rubric['examples'],
                        convert_to_numpy=True
                    )
                    self.example_embeddings[rubric['id']] = embeddings
        
        # Сохраняем модель
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.rubric_embeddings,
                'example_embeddings': self.example_embeddings,
                'model_name': self.model_name,
                'use_keywords': self.use_keywords,
                'keyword_weight': self.keyword_weight,
                'use_examples': self.use_examples
            }, f)
        
        print(f"✓ Классификатор сохранен в {save_path}")
        print(f"✓ Создано {len(self.rubric_embeddings)} эмбеддингов рубрикаторов")
        print(f"✓ Создано {len(self.example_embeddings)} наборов эмбеддингов примеров")
    
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
        self.example_embeddings = data.get('example_embeddings', {})
        self.model_name = data['model_name']
        self.use_keywords = data.get('use_keywords', True)
        self.keyword_weight = data.get('keyword_weight', 0.35)
        self.use_examples = data.get('use_examples', True)
        self.semantic_weight = 1 - self.keyword_weight
        
        # Загружаем модель для предсказаний
        self.load_model()
        
        print("✓ Классификатор загружен")
    
    def _calculate_semantic_scores(self, text: str) -> np.ndarray:
        """
        Расчет семантических оценок через cosine similarity.
        Улучшенная версия с учетом примеров.
        
        Args:
            text: текст жалобы
            
        Returns:
            Массив оценок для каждого рубрикатора
        """
        # Создаем эмбеддинг текста
        text_embedding = self.model.encode([text], convert_to_numpy=True)
        
        # Считаем cosine similarity с описаниями рубрикаторов
        rubric_similarities = cosine_similarity(text_embedding, self.rubric_embeddings)[0]
        
        # Нормализуем в диапазон [0, 1]
        scores = (rubric_similarities + 1) / 2
        
        # Добавляем сравнение с примерами
        if self.use_examples and self.example_embeddings:
            for rubric in self.rubrics:
                rubric_id = rubric['id']
                idx = rubric_id - 1  # ID начинаются с 1
                
                if rubric_id in self.example_embeddings:
                    example_emb = self.example_embeddings[rubric_id]
                    example_sim = cosine_similarity(text_embedding, example_emb)[0]
                    
                    # Берем максимальное сходство с примерами
                    max_example_sim = (np.max(example_sim) + 1) / 2
                    
                    # Средняя оценка между описанием и лучшим примером
                    # С небольшим бонусом за примеры (они более конкретные)
                    scores[idx] = 0.4 * scores[idx] + 0.6 * max_example_sim
        
        return scores
    
    def _calculate_keyword_scores(self, text: str) -> np.ndarray:
        """
        Расчет оценок на основе ключевых слов.
        Улучшенная версия с приоритетами и негативными словами.
        
        Args:
            text: текст жалобы
            
        Returns:
            Массив оценок для каждого рубрикатора
        """
        scores = np.zeros(len(self.rubrics))
        
        for i, rubric in enumerate(self.rubrics):
            score, _ = calculate_advanced_keyword_score(
                text,
                rubric['keywords'],
                rubric.get('priority_keywords', []),
                rubric.get('negative_keywords', [])
            )
            scores[i] = score
        
        return scores
    
    def _apply_rule_based_adjustments(self, text: str, scores: np.ndarray) -> np.ndarray:
        """
        Применяет правила на основе специфических паттернов.
        
        Args:
            text: текст жалобы
            scores: текущие оценки
            
        Returns:
            Скорректированные оценки
        """
        text_lower = text.lower()
        adjusted_scores = scores.copy()
        
        # Правило 1: Если есть 161-ФЗ - это скорее всего блокировка (ID=2)
        if '161-фз' in text_lower or '161 фз' in text_lower:
            adjusted_scores[1] += 0.25  # Бонус для ID=2
            adjusted_scores[0] -= 0.1   # Штраф для ID=1
        
        # Правило 2: Если есть 115-ФЗ без управляющего/адвоката - это ID=3
        if ('115-фз' in text_lower or '115 фз' in text_lower):
            if 'управляющ' not in text_lower and 'адвокат' not in text_lower:
                adjusted_scores[2] += 0.2  # Бонус для ID=3
        
        # Правило 3: Коллекторы - явный признак
        if 'коллектор' in text_lower:
            adjusted_scores[4] += 0.3  # Бонус для ID=5
            adjusted_scores[3] -= 0.15  # Штраф для ФССП
        
        # Правило 4: ФССП/пристав - явный признак
        if 'фссп' in text_lower or 'пристав' in text_lower:
            if 'коллектор' not in text_lower:
                adjusted_scores[3] += 0.25  # Бонус для ID=4
        
        # Правило 5: Мошенники + Росфинмониторинг + звонок = ID=12
        if 'росфинмониторинг' in text_lower:
            if 'звон' in text_lower or 'позвонил' in text_lower or 'мошенник' in text_lower:
                adjusted_scores[11] += 0.3  # ID=12
            if 'доверенность' in text_lower:
                adjusted_scores[12] += 0.3  # ID=13
            if 'письмо' in text_lower and ('оплат' in text_lower or 'штраф' in text_lower):
                adjusted_scores[13] += 0.3  # ID=14
            if 'сотрудник' in text_lower and 'подтвердить' in text_lower:
                adjusted_scores[14] += 0.3  # ID=15
        
        # Правило 6: Госуслуги + взлом
        if 'госуслуг' in text_lower or 'епгу' in text_lower:
            if 'взлом' in text_lower or 'взломал' in text_lower:
                if 'полици' in text_lower or 'мвд' in text_lower or 'заявлени' in text_lower:
                    adjusted_scores[17] += 0.25  # ID=18 - обратился в полицию
                else:
                    adjusted_scores[16] += 0.25  # ID=17 - просто взлом
            if 'доверенность' in text_lower:
                adjusted_scores[12] += 0.25  # ID=13
        
        # Правило 7: Кредит + не брал/мошенники
        if 'кредит' in text_lower:
            if 'не брал' in text_lower or 'не оформлял' in text_lower or 'без согласия' in text_lower:
                adjusted_scores[15] += 0.3  # ID=16
        
        # Правило 8: Межведомственная комиссия
        if 'межведомственн' in text_lower and 'комисси' in text_lower:
            adjusted_scores[18] += 0.4  # ID=19
        
        # Правило 9: Ошибочный перевод
        if 'ошибочн' in text_lower and 'перевод' in text_lower:
            adjusted_scores[19] += 0.35  # ID=20
        if 'перепутал' in text_lower and ('номер' in text_lower or 'реквизит' in text_lower):
            adjusted_scores[19] += 0.25  # ID=20
        
        # Правило 10: Казино
        if 'казино' in text_lower:
            if 'реквизит' in text_lower or 'пополнени' in text_lower:
                adjusted_scores[9] += 0.25  # ID=10
            if 'выплат' in text_lower or 'вывод' in text_lower or 'не выплач' in text_lower:
                adjusted_scores[10] += 0.25  # ID=11
        
        # Правило 11: Финансовый/конкурсный управляющий
        if 'управляющ' in text_lower:
            if 'финансов' in text_lower or 'конкурсн' in text_lower or 'арбитражн' in text_lower:
                adjusted_scores[7] += 0.35  # ID=8
        
        # Правило 12: Адвокат
        if 'адвокат' in text_lower and ('запрос' in text_lower or '63-фз' in text_lower):
            adjusted_scores[8] += 0.35  # ID=9
        
        # Ограничиваем значения
        adjusted_scores = np.clip(adjusted_scores, 0, 1)
        
        return adjusted_scores
    
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
            combined_scores = (
                self.semantic_weight * semantic_scores +
                self.keyword_weight * keyword_scores
            )
        else:
            keyword_scores = np.zeros(len(self.rubrics))
            combined_scores = semantic_scores
        
        # Применяем правила
        final_scores = self._apply_rule_based_adjustments(text, combined_scores)
        
        # Находим топ-k результатов
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        # Формируем результаты
        predictions = []
        for idx in top_indices:
            rubric = self.rubrics[idx]
            rubric_id = rubric['id']
            predictions.append({
                'rubric_id': rubric_id,
                'rubric_name': rubric['description'],  # Используем полное описание
                'short_name': rubric['name'],  # Краткое название для справки
                'response_template': get_response_template(rubric_id),  # Шаблон ответа
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
