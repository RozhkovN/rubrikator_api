"""
Утилиты для предобработки текста жалоб.
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """
    Очистка текста от лишних символов.
    
    Args:
        text: исходный текст
        
    Returns:
        Очищенный текст
    """
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    
    # Убираем пробелы в начале и конце
    text = text.strip()
    
    return text


def normalize_text(text: str) -> str:
    """
    Нормализация текста для лучшей обработки.
    
    Args:
        text: исходный текст
        
    Returns:
        Нормализованный текст
    """
    # Приводим к нижнему регистру
    text = text.lower()
    
    # Очищаем
    text = clean_text(text)
    
    return text


def extract_keywords(text: str, keywords: List[str]) -> List[str]:
    """
    Извлекает ключевые слова из текста.
    
    Args:
        text: текст для анализа
        keywords: список ключевых слов для поиска
        
    Returns:
        Список найденных ключевых слов
    """
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords


def calculate_keyword_score(text: str, keywords: List[str]) -> float:
    """
    Рассчитывает оценку совпадения ключевых слов.
    
    Args:
        text: текст для анализа
        keywords: список ключевых слов
        
    Returns:
        Оценка от 0 до 1
    """
    if not keywords:
        return 0.0
    
    found = extract_keywords(text, keywords)
    return len(found) / len(keywords)
