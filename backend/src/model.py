"""
Основная модель классификатора жалоб.

Использует гибридный подход:
1. Sentence Transformers для семантического анализа
2. Анализ ключевых слов для уточнения
3. Примеры жалоб для улучшения эмбеддингов
4. Приоритетные и негативные ключевые слова
5. Взаимоисключающие правила между рубриками
6. LRU кэширование для ускорения
7. Динамическая фильтрация дополнительных вариантов
"""

import os
import pickle
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from collections import OrderedDict
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


# ===================== КОНФИГУРАЦИЯ ФИЛЬТРАЦИИ =====================

# Минимальный порог confidence для дополнительных вариантов
MIN_CONFIDENCE_THRESHOLD = 0.25

# Максимальный разрыв между лучшим и вторым результатом (если больше - не показываем второй)
MAX_GAP_THRESHOLD = 0.35

# Минимальный confidence для основного результата, чтобы считать его уверенным
HIGH_CONFIDENCE_THRESHOLD = 0.65


# ===================== ВЗАИМОИСКЛЮЧАЮЩИЕ ПРАВИЛА =====================
# Если выбрана одна рубрика, другие из этого списка получают штраф
MUTUALLY_EXCLUSIVE_GROUPS = [
    # Блокировки: 161-ФЗ vs 115-ФЗ
    {2, 3},
    
    # ФССП vs Коллекторы
    {4, 5},
    
    # Казино: реквизиты vs жалоба
    {10, 11},
    
    # Мошенничество от имени РФМ: разные типы
    {12, 13, 14, 15},
    
    # Госуслуги: взлом vs взлом+полиция
    {17, 18},
    
    # Управляющий vs Адвокат
    {8, 9},
]


class LRUCache:
    """LRU кэш для эмбеддингов текстов"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_text(self, text: str) -> str:
        """Создаёт хэш текста для ключа кэша"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Получить эмбеддинг из кэша"""
        key = self._hash_text(text)
        if key in self.cache:
            self.hits += 1
            # Перемещаем в конец (недавно использованный)
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """Добавить эмбеддинг в кэш"""
        key = self._hash_text(text)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Удаляем самый старый элемент
                self.cache.popitem(last=False)
            self.cache[key] = embedding
    
    def get_stats(self) -> Dict:
        """Статистика кэша"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}"
        }
    
    def clear(self):
        """Очистка кэша"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class ComplaintClassifier:
    """Классификатор жалоб на основе Sentence Transformers"""
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        use_keywords: bool = True,
        keyword_weight: float = 0.35,
        use_examples: bool = True,
        cache_size: int = 1000,
        use_onnx: bool = False
    ):
        """
        Инициализация классификатора.
        
        Args:
            model_name: название модели Sentence Transformers
            use_keywords: использовать ли анализ ключевых слов
            keyword_weight: вес ключевых слов (0-1), остальное - семантика
            use_examples: использовать ли примеры для эмбеддингов
            cache_size: размер LRU кэша для эмбеддингов
            use_onnx: использовать ли ONNX Runtime для ускорения
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
        
        # LRU кэш для эмбеддингов
        self.embedding_cache = LRUCache(max_size=cache_size)
        
        print(f"🔧 Инициализация классификатора...")
        print(f"   Модель: {model_name}")
        print(f"   Ключевые слова: {'Да' if use_keywords else 'Нет'}")
        print(f"   Примеры: {'Да' if use_examples else 'Нет'}")
        print(f"   Кэш: {cache_size} элементов")
        print(f"   ONNX: {'Да' if use_onnx else 'Нет'}")
        if use_keywords:
            print(f"   Веса: семантика={self.semantic_weight:.2f}, ключевые слова={self.keyword_weight:.2f}")
    
    def load_model(self):
        """Загрузка модели Sentence Transformers с опциональной ONNX оптимизацией"""
        if self.model is None:
            print(f"📥 Загрузка модели {self.model_name}...")
            
            # Пробуем загрузить с ONNX backend для ускорения
            if self.use_onnx:
                try:
                    # Проверяем доступность ONNX Runtime
                    import onnxruntime
                    print("   🚀 Используем ONNX Runtime для ускорения...")
                    
                    # Загружаем модель с ONNX backend
                    self.model = SentenceTransformer(
                        self.model_name,
                        backend="onnx"
                    )
                    print("✓ Модель загружена с ONNX оптимизацией")
                except ImportError:
                    print("   ⚠️ ONNX Runtime не установлен, используем стандартный backend")
                    self.model = SentenceTransformer(self.model_name)
                    print("✓ Модель загружена (стандартный режим)")
                except Exception as e:
                    print(f"   ⚠️ Ошибка ONNX: {e}, используем стандартный backend")
                    self.model = SentenceTransformer(self.model_name)
                    print("✓ Модель загружена (стандартный режим)")
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
        
        print("✓ Классификатор загружен")
    
    def _calculate_semantic_scores(self, text: str) -> np.ndarray:
        """
        Расчет семантических оценок через cosine similarity.
        Улучшенная версия с учетом примеров и кэшированием.
        
        Args:
            text: текст жалобы
            
        Returns:
            Массив оценок для каждого рубрикатора
        """
        # Проверяем кэш
        cached_embedding = self.embedding_cache.get(text)
        if cached_embedding is not None:
            text_embedding = cached_embedding
        else:
            # Создаем эмбеддинг текста
            text_embedding = self.model.encode([text], convert_to_numpy=True)
            # Сохраняем в кэш
            self.embedding_cache.put(text, text_embedding)
        
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
        Улучшенная версия с дополнительными правилами.
        
        Args:
            text: текст жалобы
            scores: текущие оценки
            
        Returns:
            Скорректированные оценки
        """
        text_lower = text.lower()
        adjusted_scores = scores.copy()
        
        # ===================== ПРАВИЛА ДЛЯ ЗАКОНОВ =====================
        
        # Правило 1: Если есть 161-ФЗ - это скорее всего блокировка (ID=2)
        if '161-фз' in text_lower or '161 фз' in text_lower or '161фз' in text_lower:
            adjusted_scores[1] += 0.30  # Бонус для ID=2
            adjusted_scores[0] -= 0.15   # Штраф для ID=1
            adjusted_scores[2] -= 0.10   # Штраф для ID=3 (115-ФЗ)
        
        # Правило 2: Если есть 115-ФЗ без управляющего/адвоката - это ID=3
        if ('115-фз' in text_lower or '115 фз' in text_lower or '115фз' in text_lower):
            if 'управляющ' not in text_lower and 'адвокат' not in text_lower:
                adjusted_scores[2] += 0.25  # Бонус для ID=3
                adjusted_scores[1] -= 0.10  # Штраф для ID=2 (161-ФЗ)
        
        # Правило 2.1: 230-ФЗ - коллекторы
        if '230-фз' in text_lower or '230 фз' in text_lower:
            adjusted_scores[4] += 0.25  # Бонус для ID=5
        
        # ===================== ПРАВИЛА ДЛЯ ВЗЫСКАНИЯ =====================
        
        # Правило 3: Коллекторы - явный признак
        if 'коллектор' in text_lower or 'коллекторск' in text_lower:
            adjusted_scores[4] += 0.35  # Бонус для ID=5
            adjusted_scores[3] -= 0.20  # Штраф для ФССП
            adjusted_scores[0] -= 0.10  # Штраф для обычной жалобы на банк
        
        # Правило 4: ФССП/пристав - явный признак
        if 'фссп' in text_lower or 'пристав' in text_lower or 'исполнительн' in text_lower:
            if 'коллектор' not in text_lower:
                adjusted_scores[3] += 0.30  # Бонус для ID=4
                adjusted_scores[4] -= 0.15  # Штраф для коллекторов
        
        # ===================== ПРАВИЛА ДЛЯ МОШЕННИЧЕСТВА =====================
        
        # Правило 5: Мошенники + Росфинмониторинг
        if 'росфинмониторинг' in text_lower or 'рфм' in text_lower:
            # Звонят мошенники от имени РФМ
            if any(word in text_lower for word in ['звон', 'позвонил', 'мошенник', 'безопасн', 'перевести']):
                adjusted_scores[11] += 0.35  # ID=12
                # Штрафуем смежные
                adjusted_scores[12] -= 0.10
                adjusted_scores[13] -= 0.10
                adjusted_scores[14] -= 0.10
            
            # Доверенность на Госуслугах
            if 'доверенность' in text_lower and ('госуслуг' in text_lower or 'появил' in text_lower):
                adjusted_scores[12] += 0.35  # ID=13
                adjusted_scores[11] -= 0.10
            
            # Письмо об оплате
            if 'письмо' in text_lower and any(word in text_lower for word in ['оплат', 'штраф', 'пени', 'комисс', 'лиценз']):
                adjusted_scores[13] += 0.35  # ID=14
                adjusted_scores[11] -= 0.10
            
            # Проверка сотрудника
            if any(word in text_lower for word in ['является ли', 'подтвердить', 'проверить']) and 'сотрудник' in text_lower:
                adjusted_scores[14] += 0.35  # ID=15
                adjusted_scores[11] -= 0.10
        
        # ===================== ПРАВИЛА ДЛЯ ГОСУСЛУГ =====================
        
        # Правило 6: Госуслуги + взлом
        if 'госуслуг' in text_lower or 'епгу' in text_lower or 'есиа' in text_lower:
            if any(word in text_lower for word in ['взлом', 'взломал', 'взломали', 'получили доступ', 'несанкционированн']):
                # Обратился в полицию
                if any(word in text_lower for word in ['полици', 'мвд', 'заявлени', 'уголовн', 'возбужден']):
                    adjusted_scores[17] += 0.35  # ID=18 - обратился в полицию
                    adjusted_scores[16] -= 0.15  # Штраф для просто взлома
                else:
                    adjusted_scores[16] += 0.30  # ID=17 - просто взлом
                    adjusted_scores[17] -= 0.10  # Небольшой штраф для полиции
            
            # Доверенность на Госуслугах
            if 'доверенность' in text_lower:
                adjusted_scores[12] += 0.30  # ID=13
        
        # ===================== ПРАВИЛА ДЛЯ КРЕДИТОВ =====================
        
        # Правило 7: Кредит + не брал/мошенники
        if 'кредит' in text_lower or 'займ' in text_lower:
            if any(phrase in text_lower for phrase in ['не брал', 'не оформлял', 'без согласия', 'без моего', 'на мое имя', 'мошенник']):
                adjusted_scores[15] += 0.35  # ID=16
                adjusted_scores[0] -= 0.15  # Штраф для обычной жалобы на банк
        
        # ===================== ПРАВИЛА ДЛЯ СПЕЦИАЛЬНЫХ СЛУЧАЕВ =====================
        
        # Правило 8: Межведомственная комиссия
        if 'межведомственн' in text_lower and 'комисси' in text_lower:
            adjusted_scores[18] += 0.45  # ID=19 - очень специфичный случай
        
        # Правило 9: Ошибочный перевод
        if any(phrase in text_lower for phrase in ['ошибочн', 'случайно перевел', 'перепутал', 'не тому', 'неправильн']):
            if 'перевод' in text_lower or 'перевел' in text_lower:
                adjusted_scores[19] += 0.40  # ID=20
                # Штрафуем несвязанные рубрики
                adjusted_scores[11] -= 0.15  # Мошенничество
        
        # Правило 10: Казино - реквизиты vs жалоба
        if 'казино' in text_lower or 'букмекер' in text_lower or 'ставк' in text_lower:
            # Реквизиты для пополнения
            if any(word in text_lower for word in ['реквизит', 'пополнени', 'карт', 'счет']):
                if 'выплат' not in text_lower and 'вывод' not in text_lower:
                    adjusted_scores[9] += 0.30  # ID=10
                    adjusted_scores[10] -= 0.15
            
            # Жалоба на невыплату
            if any(word in text_lower for word in ['выплат', 'вывод', 'не выплач', 'заблокировал', 'не могу вывести']):
                adjusted_scores[10] += 0.30  # ID=11
                adjusted_scores[9] -= 0.15
        
        # Правило 11: Финансовый/конкурсный управляющий
        if 'управляющ' in text_lower:
            if any(word in text_lower for word in ['финансов', 'конкурсн', 'арбитражн', 'банкротств']):
                adjusted_scores[7] += 0.40  # ID=8
                adjusted_scores[8] -= 0.15  # Штраф для адвоката
        
        # Правило 12: Адвокат
        if 'адвокат' in text_lower:
            if any(word in text_lower for word in ['запрос', '63-фз', '63 фз', 'доверител']):
                adjusted_scores[8] += 0.40  # ID=9
                adjusted_scores[7] -= 0.15  # Штраф для управляющего
        
        # ===================== ПРАВИЛА ДЛЯ АНТИМОНОПОЛЬНОГО/НАЛОГОВ =====================
        
        # Правило 13: ФАС / антимонопольное
        if 'фас' in text_lower or 'антимонопольн' in text_lower or 'монопол' in text_lower:
            adjusted_scores[5] += 0.35  # ID=6
        
        # Правило 14: ФНС / налоги
        if 'фнс' in text_lower or 'налог' in text_lower:
            if any(word in text_lower for word in ['уклонен', 'неуплат', 'серая', 'конверт']):
                adjusted_scores[6] += 0.35  # ID=7
        
        # Ограничиваем значения
        adjusted_scores = np.clip(adjusted_scores, 0, 1)
        
        return adjusted_scores
    
    def _apply_mutual_exclusion(self, scores: np.ndarray) -> np.ndarray:
        """
        Применяет взаимоисключающие правила между рубриками.
        Если одна рубрика явно лидирует, смежные получают штраф.
        
        Args:
            scores: текущие оценки
            
        Returns:
            Скорректированные оценки
        """
        adjusted_scores = scores.copy()
        
        for group in MUTUALLY_EXCLUSIVE_GROUPS:
            # Находим индексы рубрик в группе
            indices = [rubric_id - 1 for rubric_id in group]
            
            # Находим лидера в группе
            group_scores = [(idx, adjusted_scores[idx]) for idx in indices]
            group_scores.sort(key=lambda x: x[1], reverse=True)
            
            if len(group_scores) >= 2:
                leader_idx, leader_score = group_scores[0]
                second_idx, second_score = group_scores[1]
                
                # Если лидер значительно впереди, штрафуем остальных в группе
                if leader_score > HIGH_CONFIDENCE_THRESHOLD and (leader_score - second_score) > 0.15:
                    for idx, score in group_scores[1:]:
                        # Штраф пропорционален разнице
                        penalty = min(0.20, (leader_score - score) * 0.3)
                        adjusted_scores[idx] = max(0, adjusted_scores[idx] - penalty)
        
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
        rule_adjusted_scores = self._apply_rule_based_adjustments(text, combined_scores)
        
        # Применяем взаимоисключающие правила
        final_scores = self._apply_mutual_exclusion(rule_adjusted_scores)
        
        # Находим топ-k результатов
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        # Формируем результаты с фильтрацией
        predictions = []
        best_score = final_scores[top_indices[0]] if len(top_indices) > 0 else 0
        
        for i, idx in enumerate(top_indices):
            rubric = self.rubrics[idx]
            rubric_id = rubric['id']
            score = float(final_scores[idx])
            
            # Для дополнительных вариантов (не первый) применяем фильтрацию
            if i > 0:
                # Фильтр 1: Минимальный порог confidence
                if score < MIN_CONFIDENCE_THRESHOLD:
                    continue
                
                # Фильтр 2: Максимальный разрыв с лидером
                gap = best_score - score
                if gap > MAX_GAP_THRESHOLD:
                    continue
                
                # Фильтр 3: Если лидер очень уверенный, строже фильтруем остальных
                if best_score > HIGH_CONFIDENCE_THRESHOLD and score < MIN_CONFIDENCE_THRESHOLD + 0.10:
                    continue
            
            predictions.append({
                'rubric_id': rubric_id,
                'rubric_name': rubric['description'],  # Используем полное описание
                'short_name': rubric['name'],  # Краткое название для справки
                'response_template': get_response_template(rubric_id),  # Шаблон ответа
                'confidence': score,
                'semantic_score': float(semantic_scores[idx]) if return_scores else None,
                'keyword_score': float(keyword_scores[idx]) if return_scores else None
            })
        
        result = {
            'text': text,
            'predictions': predictions,
            'best_match': predictions[0] if predictions else None,
            'filtered_count': top_k - len(predictions)  # Сколько вариантов отфильтровано
        }
        
        return result
    
    def _filter_predictions(
        self,
        predictions: List[Dict],
        min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
        max_gap: float = MAX_GAP_THRESHOLD
    ) -> List[Dict]:
        """
        Фильтрует дополнительные варианты предсказаний.
        
        Args:
            predictions: список предсказаний
            min_confidence: минимальный порог confidence
            max_gap: максимальный разрыв с лидером
            
        Returns:
            Отфильтрованный список
        """
        if not predictions:
            return predictions
        
        best_score = predictions[0]['confidence']
        filtered = [predictions[0]]  # Лидер всегда остаётся
        
        for pred in predictions[1:]:
            score = pred['confidence']
            
            # Проверяем минимальный порог
            if score < min_confidence:
                continue
            
            # Проверяем разрыв с лидером
            if (best_score - score) > max_gap:
                continue
            
            # Если лидер очень уверенный, строже фильтруем
            if best_score > HIGH_CONFIDENCE_THRESHOLD and score < min_confidence + 0.10:
                continue
            
            filtered.append(pred)
        
        return filtered
    
    def get_cache_stats(self) -> Dict:
        """Возвращает статистику кэша"""
        return self.embedding_cache.get_stats()
    
    def clear_cache(self):
        """Очищает кэш эмбеддингов"""
        self.embedding_cache.clear()
    
    def predict_batch(
        self,
        texts: List[str],
        top_k: int = 1
    ) -> List[Dict]:
        """
        Пакетная классификация жалоб.
        Оптимизированная версия с batch-обработкой эмбеддингов.
        
        Args:
            texts: список текстов жалоб
            top_k: количество топ результатов
            
        Returns:
            Список результатов для каждого текста
        """
        if not texts:
            return []
        
        # Разделяем на кэшированные и некэшированные
        uncached_texts = []
        uncached_indices = []
        cached_embeddings = {}
        
        for i, text in enumerate(texts):
            cached = self.embedding_cache.get(text)
            if cached is not None:
                cached_embeddings[i] = cached
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Batch-обработка некэшированных текстов
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                convert_to_numpy=True,
                show_progress_bar=len(uncached_texts) > 10
            )
            
            # Кэшируем новые эмбеддинги
            for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                orig_idx = uncached_indices[i]
                cached_embeddings[orig_idx] = embedding.reshape(1, -1)
                self.embedding_cache.put(text, embedding.reshape(1, -1))
        
        # Обрабатываем каждый текст
        results = []
        for i, text in enumerate(texts):
            result = self.predict(text, top_k=top_k, return_scores=False)
            results.append(result)
        
        return results
