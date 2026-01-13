"""
Скрипт для тестирования API.

Использование:
    python test_api.py
"""

import requests
import json

# URL API
API_URL = "http://localhost:8800"


def test_health():
    """Тест health check"""
    print("\n" + "="*60)
    print("🔍 Проверка состояния API...")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Статус: {response.status_code}")
    print(f"Ответ: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")


def test_classify(text: str):
    """Тест классификации одной жалобы"""
    print("\n" + "="*60)
    print("📝 Тест классификации жалобы")
    print("="*60)
    print(f"Текст: {text}\n")
    
    response = requests.post(
        f"{API_URL}/classify",
        json={
            "text": text,
            "top_k": 1
        }
    )
    
    print(f"Статус: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ РЕЗУЛЬТАТ:")
        print(f"   Рубрикатор: {result['best_match']['rubric_name']}")
        print(f"   Краткое: {result['best_match']['short_name']}")
        print(f"   Уверенность: {result['best_match']['confidence']:.2%}")
    else:
        print(f"❌ Ошибка: {response.text}")


def test_classify_top3(text: str):
    """Тест классификации с топ-3 результатами"""
    print("\n" + "="*60)
    print("📝 Тест классификации с топ-3")
    print("="*60)
    print(f"Текст: {text}\n")
    
    response = requests.post(
        f"{API_URL}/classify",
        json={
            "text": text,
            "top_k": 3
        }
    )
    
    print(f"Статус: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ ЛУЧШЕЕ СОВПАДЕНИЕ:")
        print(f"   {result['best_match']['rubric_name']}")
        print(f"   Уверенность: {result['best_match']['confidence']:.2%}")
        
        if result.get('all_predictions'):
            print(f"\n📊 ВСЕ РЕЗУЛЬТАТЫ:")
            for i, pred in enumerate(result['all_predictions'], 1):
                print(f"   {i}. {pred['short_name']}")
                print(f"      Уверенность: {pred['confidence']:.2%}")
    else:
        print(f"❌ Ошибка: {response.text}")


def test_batch():
    """Тест пакетной классификации"""
    print("\n" + "="*60)
    print("📦 Тест пакетной классификации")
    print("="*60)
    
    complaints = [
        "Банк заблокировал мою карту без предупреждения",
        "Коллекторы звонят по 30 раз в день",
        "Ошибочно перевел деньги через СБП не тому человеку"
    ]
    
    response = requests.post(
        f"{API_URL}/classify/batch",
        json=complaints
    )
    
    print(f"Статус: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Обработано жалоб: {result['count']}\n")
        for i, item in enumerate(result['results'], 1):
            print(f"{i}. {item['short_name']}")
            print(f"   Текст: {item['text'][:60]}...")
            print(f"   Уверенность: {item['confidence']:.2%}\n")
    else:
        print(f"❌ Ошибка: {response.text}")


def main():
    """Основная функция"""
    print("\n" + "🚀 ТЕСТИРОВАНИЕ API РУБРИКАТОРА ЖАЛОБ")
    print("="*60)
    
    try:
        # 1. Health check
        test_health()
        
        # 2. Простая классификация
        test_classify(
            "Банк Сбербанк заблокировал мою карту без объяснения причин согласно 161-ФЗ"
        )
        
        # 3. Классификация с топ-3
        test_classify_top3(
            "Мне позвонили якобы из Росфинмониторинга и требуют перевести деньги"
        )
        
        # 4. Пакетная классификация
        test_batch()
        
        print("\n" + "="*60)
        print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
        print("="*60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Не удается подключиться к API")
        print("Убедитесь, что сервер запущен: python run_api.py\n")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}\n")


if __name__ == "__main__":
    main()
