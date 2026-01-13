"""
Скрипт для запуска API сервера.

Использование:
    python run_api.py
"""

import uvicorn

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8800))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
