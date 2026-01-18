"""
Основная модель классификатора жалоб.

Использует гибридный подход:
1. Sentence Transformers для семантического анализа (с ONNX ускорением)
2. Анализ ключевых слов для уточнения
3. Примеры жалоб для улучшения эмбеддингов
4. Приоритетные и негативные ключевые слова
5. Взаимоисключающие правила между рубриками
6. Умная фильтрация дополнительных вариантов
7. LRU-кэширование для ускорения повторных запросов
"""

import os
import pickle
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from collections import OrderedDict
import threading
import time

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


# ============================================================================
# LRU Cache для эмбеддингов
# ============================================================================

class EmbeddingCache:
    """Thread-safe LRU кэш для эмбеддингов текстов"""
    
    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 3600):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _hash_text(self, text: str) -> str:
        """Создаёт хэш текста для ключа кэша"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Получить эмбеддинг из кэша"""
        key = self._hash_text(text)
        with self.lock:
            if key in self.cache:
                # Проверяем TTL
                if time.time() - self.timestamps[key] < self.ttl_seconds:
                    # Перемещаем в конец (LRU)
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return self.cache[key]
                else:
                    # TTL истёк, удаляем
                    del self.cache[key]
                    del self.timestamps[key]
            self.misses += 1
            return None
    
    def set(self, text: str, embedding: np.ndarray):
        """Сохранить эмбеддинг в кэш"""
        key = self._hash_text(text)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.maxsize:
                    # Удаляем самый старый
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
                self.cache[key] = embedding.copy()
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Очистить кэш"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict:
        """Статистика кэша"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'size': len(self.cache),
                'maxsize': self.maxsize,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': f"{hit_rate:.2%}"
            }


# ============================================================================
# Взаимоисключающие правила между рубриками
# ============================================================================

# Группы взаимоисключающих рубрик (если одна выбрана с высоким confidence,
# другие из группы получают штраф)
MUTUALLY_EXCLUSIVE_GROUPS = [
    # Банковские блокировки vs общие жалобы на банки
    {1, 2, 3},  # ID: 1-жалоба на банк, 2-блокировка 161-ФЗ, 3-нарушение 115-ФЗ
    
    # ФССП vs Коллекторы
    {4, 5},  # ID: 4-ФССП, 5-коллекторы
    
    # Управляющий vs Адвокат
    {8, 9},  # ID: 8-управляющий, 9-адвокат
    
    # Казино: реквизиты vs жалоба
    {10, 11},  # ID: 10-реквизиты казино, 11-жалоба на казино
    
    # Мошенничество от имени Росфинмониторинга (разные типы)
    {12, 13, 14, 15},  # ID: 12-звонок, 13-доверенность, 14-письмо, 15-проверка сотрудника
    
    # Взлом Госуслуг (с полицией и без)
    {17, 18},  # ID: 17-взлом, 18-взлом+полиция
]

# Пороги для фильтрации дополнительных вариантов
MIN_CONFIDENCE_THRESHOLD = 0.35  # Минимальный порог для показа варианта
MIN_GAP_THRESHOLD = 0.15  # Минимальная разница между лучшим и следующим


class ComplaintClassifier:
    """Классификатор жалоб на основе Sentence Transformers с ONNX ускорением"""
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        use_keywords: bool = True,
        keyword_weight: float = 0.35,
        use_examples: bool = True,
        use_onnx: bool = True,
        cache_size: int = 1000,
        cache_ttl: int = 3600
    ):
        """
        Инициализация классификатора.
        
        Args:
            model_name: название модели Sentence Transformers
            use_keywords: использовать ли анализ ключевых слов
            keyword_weight: вес ключевых слов (0-1), остальное - семантика
            use_examples: использовать ли примеры для эмбеддингов
            use_onnx: использовать ли ONNX ускорение (2-3x быстрее)
            cache_size: размер LRU кэша для эмбеддингов
            cache_ttl: время жизни кэша в секундах
        """
        self.model_name = model_name
        self.use_keywords = use_keywords
        self.keyword_weight = keyword_weight
        self.semantic_weight = 1 - keyword_weight
        self.use_examples = use_examples
        self.use_onnx = use_onnx
        
        self.model: Optional[SentenceTransformer] = None
        self.rubric_embeddings: Optional[np.ndarray] = None
        self.example_embeddings: Optional[Dict[int, np.ndarray]] = None
        self.rubrics = RUBRICS
        
        # Инициализация кэша
        self.embedding_cache = EmbeddingCache(maxsize=cache_size, ttl_seconds=cache_ttl)
        
        # Статистика производительности
        self.stats = {
            'total_predictions': 0,
            'avg_time_ms': 0,
            'total_time_ms': 0
        }
        
        print(f"🔧 Инициализация классификатора...")
        print(f"   Модель: {model_name}")
        print(f"   ONNX ускорение: {'Да' if use_onnx else 'Нет'}")
        print(f"   Ключевые слова: {'Да' if use_keywords else 'Нет'}")
        print(f"   Примеры: {'Да' if use_examples else 'Нет'}")
        print(f"   Кэш: {cache_size} элементов, TTL {cache_ttl}с")
        if use_keywords:
            print(f"   Веса: семантика={self.semantic_weight:.2f}, ключевые слова={self.keyword_weight:.2f}")
    
    def load_model(self):
        """Загрузка модели Sentence Transformers с опциональным ONNX бэкендом"""
        if self.model is None:
            print(f"📥 Загрузка модели {self.model_name}...")
            
            # Попробуем загрузить с ONNX
            if self.use_onnx:
                try:
                    # ONNX Runtime для ускорения
                    self.model = SentenceTransformer(
                        self.model_name,
                        backend="onnx"
                    )
                    print("✓ Модель загружена с ONNX ускорением")
                except Exception as e:
                    print(f"⚠️  ONNX недоступен: {e}")
                    print("   Загрузка стандартной модели...")
                    self.model = SentenceTransformer(self.model_name)
                    print("✓ Модель загружена (без ONNX)")
            else:
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
        
        # Очищаем кэш после загрузки новой модели
        self.embedding_cache.clear()
        
        print("✓ Классификатор загружен")
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Получить эмбеддинг текста с использованием кэша.
        
        Args:
            text: текст для эмбеддинга
            
        Returns:
            Эмбеддинг текста
        """
        # Проверяем кэш
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached
        
        # Создаём новый эмбеддинг
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        
        # Сохраняем в кэш
        self.embedding_cache.set(text, embedding)
        
        return embedding
    
    def _calculate_semantic_scores(self, text: str) -> np.ndarray:
        """
        Расчет семантических оценок через cosine similarity.
        Улучшенная версия с учетом примеров и кэшированием.
        
        Args:
            text: текст жалобы
            
        Returns:
            Массив оценок для каждого рубрикатора
        """
        # Получаем эмбеддинг текста (с кэшированием)
        text_embedding = self._get_text_embedding(text).reshape(1, -1)
        
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
        Усиленная версия с дополнительными правилами.
        
        Args:
            text: текст жалобы
            scores: текущие оценки
            
        Returns:
            Скорректированные оценки
        """
        text_lower = text.lower()
        adjusted_scores = scores.copy()
        
        # ============================================================
        # БЛОК 1: Законы и нормативные акты
        # ============================================================
        
        # Правило 1: 161-ФЗ - блокировка карты/счета (ID=2)
        if '161-фз' in text_lower or '161 фз' in text_lower or '161фз' in text_lower:
            adjusted_scores[1] += 0.35  # Бонус для ID=2
            adjusted_scores[0] -= 0.15  # Штраф для ID=1 (общая жалоба на банк)
            adjusted_scores[2] -= 0.1   # Штраф для ID=3 (115-ФЗ)
        
        # Правило 2: 115-ФЗ без управляющего/адвоката - нарушение 115-ФЗ (ID=3)
        if '115-фз' in text_lower or '115 фз' in text_lower or '115фз' in text_lower:
            if 'управляющ' not in text_lower and 'адвокат' not in text_lower:
                adjusted_scores[2] += 0.3  # Бонус для ID=3
                adjusted_scores[1] -= 0.1  # Штраф для ID=2 (161-ФЗ)
        
        # Правило 2.1: 230-ФЗ - коллекторы (ID=5)
        if '230-фз' in text_lower or '230 фз' in text_lower:
            adjusted_scores[4] += 0.3  # Бонус для ID=5
        
        # Правило 2.2: 63-ФЗ - адвокат (ID=9)
        if '63-фз' in text_lower or '63 фз' in text_lower:
            adjusted_scores[8] += 0.35  # Бонус для ID=9
        
        # ============================================================
        # БЛОК 2: Организации и органы
        # ============================================================
        
        # Правило 3: Коллекторы - явный признак (ID=5)
        if 'коллектор' in text_lower or 'взыскатель' in text_lower:
            adjusted_scores[4] += 0.35  # Бонус для ID=5
            adjusted_scores[3] -= 0.2   # Штраф для ФССП
            adjusted_scores[0] -= 0.1   # Штраф для общей жалобы на банк
        
        # Правило 4: ФССП/пристав - явный признак (ID=4)
        if 'фссп' in text_lower or 'пристав' in text_lower or 'исполнительн' in text_lower:
            if 'коллектор' not in text_lower:
                adjusted_scores[3] += 0.3  # Бонус для ID=4
                adjusted_scores[4] -= 0.15  # Штраф для коллекторов
        
        # Правило 4.1: ФНС/налоговая (ID=7)
        if 'фнс' in text_lower or 'налогов' in text_lower:
            if 'уклон' in text_lower or 'неуплат' in text_lower or 'серая' in text_lower or 'конверт' in text_lower:
                adjusted_scores[6] += 0.35  # Бонус для ID=7
        
        # Правило 4.2: ФАС/антимонопольный (ID=6)
        if 'фас' in text_lower or 'антимонопольн' in text_lower or 'монополи' in text_lower:
            adjusted_scores[5] += 0.35  # Бонус для ID=6
        
        # ============================================================
        # БЛОК 3: Мошенничество от имени Росфинмониторинга
        # ============================================================
        
        if 'росфинмониторинг' in text_lower or 'рфм' in text_lower:
            # ID=12: Звонок от мошенников
            if any(w in text_lower for w in ['звон', 'позвонил', 'мошенник', 'безопасн', 'перевести']):
                adjusted_scores[11] += 0.4  # ID=12
                # Штрафы для похожих рубрик
                adjusted_scores[12] -= 0.1
                adjusted_scores[13] -= 0.1
                adjusted_scores[14] -= 0.1
            
            # ID=13: Поддельная доверенность на Госуслугах
            if 'доверенность' in text_lower:
                adjusted_scores[12] += 0.4  # ID=13
                adjusted_scores[11] -= 0.1
            
            # ID=14: Фишинговое письмо об оплате
            if 'письмо' in text_lower and any(w in text_lower for w in ['оплат', 'штраф', 'пени', 'комисси', 'лицензи']):
                adjusted_scores[13] += 0.4  # ID=14
                adjusted_scores[11] -= 0.1
            
            # ID=15: Проверка сотрудника
            if any(w in text_lower for w in ['подтвердить', 'является ли', 'проверить сотрудник', 'работает ли']):
                adjusted_scores[14] += 0.4  # ID=15
                adjusted_scores[11] -= 0.15
        
        # ============================================================
        # БЛОК 4: Госуслуги и взлом
        # ============================================================
        
        if 'госуслуг' in text_lower or 'епгу' in text_lower or 'esia' in text_lower:
            # ID=17 vs ID=18: взлом с полицией или без
            if any(w in text_lower for w in ['взлом', 'взломал', 'взломали', 'несанкциониров', 'украли данные']):
                if any(w in text_lower for w in ['полици', 'мвд', 'заявлени', 'уголовн', 'возбужд']):
                    adjusted_scores[17] += 0.35  # ID=18 - обратился в полицию
                    adjusted_scores[16] -= 0.2   # Штраф для ID=17
                else:
                    adjusted_scores[16] += 0.35  # ID=17 - просто взлом
                    adjusted_scores[17] -= 0.15  # Штраф для ID=18
            
            # ID=13: Доверенность на Госуслугах
            if 'доверенность' in text_lower:
                adjusted_scores[12] += 0.3  # ID=13
        
        # ============================================================
        # БЛОК 5: Кредиты и мошенничество
        # ============================================================
        
        if 'кредит' in text_lower or 'займ' in text_lower or 'займа' in text_lower:
            # ID=16: Кредит оформлен мошенниками
            if any(phrase in text_lower for phrase in ['не брал', 'не оформлял', 'без согласия', 'без ведома', 'мошенник', 'не подписывал']):
                adjusted_scores[15] += 0.4  # ID=16
                adjusted_scores[0] -= 0.15   # Штраф для общей жалобы на банк
            # ID=1: Обычная жалоба на кредит
            elif any(phrase in text_lower for phrase in ['навязал', 'страховк', 'процент', 'условия']):
                adjusted_scores[0] += 0.2  # ID=1
        
        # ============================================================
        # БЛОК 6: Специфические темы
        # ============================================================
        
        # Правило: Межведомственная комиссия (ID=19)
        if 'межведомственн' in text_lower and 'комисси' in text_lower:
            adjusted_scores[18] += 0.5  # ID=19 - очень специфичная тема
        
        # Правило: Ошибочный перевод (ID=20)
        if any(phrase in text_lower for phrase in ['ошибочн', 'случайно перевел', 'перепутал']):
            if any(w in text_lower for w in ['перевод', 'перевел', 'отправил', 'реквизит', 'номер']):
                adjusted_scores[19] += 0.4  # ID=20
        if 'сбп' in text_lower and any(w in text_lower for w in ['ошиб', 'не тому', 'неправильн']):
            adjusted_scores[19] += 0.3  # ID=20
        
        # Правило: Казино (ID=10 vs ID=11)
        if 'казино' in text_lower or 'азартн' in text_lower or 'ставк' in text_lower:
            # ID=10: Реквизиты казино (информирование)
            if any(w in text_lower for w in ['реквизит', 'пополнени', 'карта для', 'сообщаю']):
                adjusted_scores[9] += 0.35  # ID=10
                adjusted_scores[10] -= 0.15  # Штраф для ID=11
            # ID=11: Жалоба на казино (не выплачивают)
            elif any(w in text_lower for w in ['выплат', 'вывод', 'не выплач', 'заблокиров', 'обманули']):
                adjusted_scores[10] += 0.35  # ID=11
                adjusted_scores[9] -= 0.15   # Штраф для ID=10
        
        # Правило: Финансовый/конкурсный управляющий (ID=8)
        if 'управляющ' in text_lower:
            if any(w in text_lower for w in ['финансов', 'конкурсн', 'арбитражн', 'банкротств']):
                adjusted_scores[7] += 0.4  # ID=8
                adjusted_scores[8] -= 0.15  # Штраф для адвоката
        
        # Правило: Адвокат (ID=9)
        if 'адвокат' in text_lower:
            if any(w in text_lower for w in ['запрос', '63-фз', 'доверител', 'юридическ']):
                adjusted_scores[8] += 0.4  # ID=9
                adjusted_scores[7] -= 0.15  # Штраф для управляющего
        
        # ============================================================
        # БЛОК 7: Негативные правила (исключения)
        # ============================================================
        
        # Если явно упоминаются мошенники - штраф для обычных жалоб
        if 'мошенник' in text_lower and 'росфинмониторинг' not in text_lower:
            adjusted_scores[0] -= 0.1  # ID=1
            adjusted_scores[1] -= 0.1  # ID=2
            adjusted_scores[2] -= 0.1  # ID=3
        
        # Ограничиваем значения
        adjusted_scores = np.clip(adjusted_scores, 0, 1)
        
        return adjusted_scores
    
    def _apply_mutual_exclusion(self, scores: np.ndarray) -> np.ndarray:
        """
        Применяет взаимоисключающие правила между рубриками.
        Если одна рубрика из группы имеет высокий score, другие получают штраф.
        
        Args:
            scores: текущие оценки
            
        Returns:
            Скорректированные оценки
        """
        adjusted_scores = scores.copy()
        
        for group in MUTUALLY_EXCLUSIVE_GROUPS:
            # Получаем индексы (ID - 1)
            indices = [rubric_id - 1 for rubric_id in group]
            
            # Находим лидера в группе
            group_scores = [(idx, adjusted_scores[idx]) for idx in indices]
            group_scores.sort(key=lambda x: x[1], reverse=True)
            
            if len(group_scores) >= 2:
                leader_idx, leader_score = group_scores[0]
                second_idx, second_score = group_scores[1]
                
                # Если лидер значительно впереди, штрафуем остальных
                if leader_score > 0.5 and leader_score - second_score > 0.1:
                    for idx, score in group_scores[1:]:
                        # Штраф пропорционален разнице
                        penalty = (leader_score - score) * 0.3
                        adjusted_scores[idx] = max(0, adjusted_scores[idx] - penalty)
        
        return adjusted_scores
    
    def _filter_predictions(self, predictions: List[Dict], best_confidence: float) -> List[Dict]:
        """
        Фильтрует дополнительные варианты для уменьшения ложных срабатываний.
        
        Args:
            predictions: список предсказаний
            best_confidence: confidence лучшего результата
            
        Returns:
            Отфильтрованный список
        """
        if len(predictions) <= 1:
            return predictions
        
        filtered = [predictions[0]]  # Лучший результат всегда включаем
        
        for pred in predictions[1:]:
            confidence = pred['confidence']
            gap = best_confidence - confidence
            
            # Условия для включения дополнительного варианта:
            # 1. Confidence выше минимального порога
            # 2. Разрыв с лидером не слишком большой
            # 3. Если лидер очень уверен (>0.7), требуем меньший gap
            
            if confidence >= MIN_CONFIDENCE_THRESHOLD:
                if best_confidence > 0.7:
                    # Лидер уверен - требуем gap < 0.1
                    if gap < 0.1:
                        filtered.append(pred)
                elif best_confidence > 0.5:
                    # Средняя уверенность - требуем gap < MIN_GAP_THRESHOLD
                    if gap < MIN_GAP_THRESHOLD:
                        filtered.append(pred)
                else:
                    # Низкая уверенность - показываем больше вариантов
                    if gap < MIN_GAP_THRESHOLD * 1.5:
                        filtered.append(pred)
        
        return filtered
    
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
        rule_adjusted_scores = self._apply_rule_based_adjustments(text, combined_scores)
        
        # Применяем взаимоисключающие правила
        final_scores = self._apply_mutual_exclusion(rule_adjusted_scores)
        
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
        
        # Фильтруем дополнительные варианты для уменьшения ложных срабатываний
        if len(predictions) > 1:
            predictions = self._filter_predictions(predictions, predictions[0]['confidence'])
        
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
        Пакетная классификация жалоб с оптимизированной обработкой.
        
        Args:
            texts: список текстов жалоб
            top_k: количество топ результатов
            
        Returns:
            Список результатов для каждого текста
        """
        if self.model is None or self.rubric_embeddings is None:
            raise ValueError("Модель не загружена. Используйте load() или train()")
        
        # Получаем эмбеддинги для всех текстов (батч)
        # Сначала проверяем кэш
        embeddings = []
        texts_to_encode = []
        text_indices = []
        
        for i, text in enumerate(texts):
            cached = self.embedding_cache.get(text)
            if cached is not None:
                embeddings.append((i, cached))
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
        
        # Кодируем только те, которых нет в кэше
        if texts_to_encode:
            new_embeddings = self.model.encode(texts_to_encode, convert_to_numpy=True)
            for i, (text_idx, emb) in enumerate(zip(text_indices, new_embeddings)):
                embeddings.append((text_idx, emb))
                # Сохраняем в кэш
                self.embedding_cache.set(texts_to_encode[i], emb)
        
        # Сортируем по исходному индексу
        embeddings.sort(key=lambda x: x[0])
        embeddings = [emb for _, emb in embeddings]
        
        # Классифицируем каждый текст
        results = []
        for i, text in enumerate(texts):
            result = self.predict(text, top_k=top_k, return_scores=False)
            results.append(result)
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """Получить статистику кэша"""
        return self.embedding_cache.stats()
    
    def clear_cache(self):
        """Очистить кэш эмбеддингов"""
        self.embedding_cache.clear()
