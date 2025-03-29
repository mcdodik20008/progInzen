# tools/test_block.py
def get_ui_meta():
    return {
        "id": "test_block",
        "name": "🚀 Пример блока",
        "description": "Это тестовый блок для демонстрации UI.",
        "parameters": {
            "message": {"type": "str", "default": "Привет, мир!"},
            "repeat": {"type": "int", "default": 3}
        }
    }

def main(message, repeat):
    print(message * int(repeat))

