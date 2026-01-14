"""
Утилиты для предобработки текста жалоб.
"""

import re
from typing import List, Dict, Tuple


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


def calculate_advanced_keyword_score(
    text: str,
    keywords: List[str],
    priority_keywords: List[str] = None,
    negative_keywords: List[str] = None
) -> Tuple[float, Dict]:
    """
    Расчет продвинутой оценки ключевых слов с учетом приоритетов.
    
    Args:
        text: текст для анализа
        keywords: обычные ключевые слова
        priority_keywords: приоритетные ключевые слова (дают больший вес)
        negative_keywords: негативные ключевые слова (снижают оценку)
        
    Returns:
        Tuple из (оценка, детали)
    """
    text_lower = text.lower()
    
    # Подсчет обычных ключевых слов
    found_keywords = []
    for kw in keywords:
        if kw.lower() in text_lower:
            found_keywords.append(kw)
    
    # Подсчет приоритетных ключевых слов
    found_priority = []
    if priority_keywords:
        for kw in priority_keywords:
            if kw.lower() in text_lower:
                found_priority.append(kw)
    
    # Подсчет негативных ключевых слов
    found_negative = []
    if negative_keywords:
        for kw in negative_keywords:
            if kw.lower() in text_lower:
                found_negative.append(kw)
    
    # Расчет базовой оценки
    base_score = len(found_keywords) / max(len(keywords), 1)
    
    # Бонус за приоритетные слова (каждое дает +0.2, максимум 0.4)
    priority_bonus = min(len(found_priority) * 0.2, 0.4) if priority_keywords else 0
    
    # Штраф за негативные слова (каждое снижает на 0.15)
    negative_penalty = len(found_negative) * 0.15 if negative_keywords else 0
    
    # Итоговая оценка
    final_score = base_score + priority_bonus - negative_penalty
    final_score = max(0.0, min(1.0, final_score))  # Ограничиваем [0, 1]
    
    details = {
        'found_keywords': found_keywords,
        'found_priority': found_priority,
        'found_negative': found_negative,
        'base_score': base_score,
        'priority_bonus': priority_bonus,
        'negative_penalty': negative_penalty
    }
    
    return final_score, details


def check_exact_phrases(text: str, phrases: List[str]) -> List[str]:
    """
    Проверяет наличие точных фраз в тексте.
    Более надежно, чем поиск отдельных слов.
    
    Args:
        text: текст для анализа
        phrases: список фраз для поиска
        
    Returns:
        Список найденных фраз
    """
    text_lower = text.lower()
    found = []
    
    for phrase in phrases:
        # Ищем фразу с учетом границ слов
        pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.append(phrase)
    
    return found


def extract_law_references(text: str) -> List[str]:
    """
    Извлекает упоминания законов (ФЗ) из текста.
    
    Args:
        text: текст для анализа
        
    Returns:
        Список найденных ссылок на законы
    """
    # Паттерны для поиска ФЗ
    patterns = [
        r'\d{1,3}-ФЗ',  # 115-ФЗ, 161-ФЗ
        r'№\s*\d{1,3}-ФЗ',  # № 115-ФЗ
        r'закон\w*\s+(?:от\s+)?\d{2}\.\d{2}\.\d{4}',  # закон от 01.01.2001
        r'федеральн\w+\s+закон\w*\s+(?:от\s+)?\d{2}\.\d{2}\.\d{4}',  # федеральный закон от...
    ]
    
    found = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        found.extend(matches)
    
    return list(set(found))


def extract_organization_mentions(text: str) -> Dict[str, List[str]]:
    """
    Извлекает упоминания организаций из текста.
    
    Args:
        text: текст для анализа
        
    Returns:
        Словарь с найденными организациями по категориям
    """
    text_lower = text.lower()
    
    orgs = {
        'banks': [],
        'government': [],
        'other': []
    }
    
    # Банки
    bank_patterns = [
        r'сбербанк\w*', r'втб', r'альфа-банк\w*', r'тинькофф\w*', 
        r'газпромбанк\w*', r'банк\w*'
    ]
    
    # Государственные органы
    gov_patterns = [
        r'росфинмониторинг\w*', r'фссп\w*', r'фнс\w*', r'фас\w*',
        r'центральн\w+\s+банк\w*', r'банк\s+росси\w*', r'госуслуг\w*',
        r'полици\w*', r'мвд', r'прокуратур\w*'
    ]
    
    for pattern in bank_patterns:
        matches = re.findall(pattern, text_lower)
        orgs['banks'].extend(matches)
    
    for pattern in gov_patterns:
        matches = re.findall(pattern, text_lower)
        orgs['government'].extend(matches)
    
    # Убираем дубликаты
    for key in orgs:
        orgs[key] = list(set(orgs[key]))
    
    return orgs
