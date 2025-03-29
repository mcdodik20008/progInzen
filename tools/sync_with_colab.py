import os
import re
import requests
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

# https://console.cloud.google.com/apis/dashboard?inv=1&invt=AbszfQ&project=behavior-454613

# Путь к папке с датасетами
DATASETS_DIR = "../ready_datasets"
# Путь к папке для сохранения моделей
MODELS_DIR = "../ready_models"
# ID папки на Google Диске для хранения датасетов
DATASETS_FOLDER_ID = '1fELxRYgDr0rXHfxFAfSRCZYk1lSdL9i2'
# ID папки на Google Диске для хранения моделей
MODELS_FOLDER_ID = '1sZEC1gBB1Xhj5HIQQk4z_LGQmvKbj2PI'
# Ссылка на Colab-ноутбук
COLAB_NOTEBOOK_URL = 'https://colab.research.google.com/drive/1ulKwTdbEa8ASsh0exagl0fcYBqL1uq7n'
SCOPES = ['https://www.googleapis.com/auth/drive']


# Аутентификация и создание сервиса для работы с Google Drive API
def authenticate_gdrive():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('_credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('drive', 'v3', credentials=creds)
    return service


# Получение пути к последнему датасету
def get_latest_dataset_path():
    files = [
        f for f in os.listdir(DATASETS_DIR)
        if f.startswith("v") and f.endswith(".npz")
    ]
    if not files:
        raise FileNotFoundError("Нет датасетов в папке ready_datasets/")
    versions = sorted([int(re.findall(r"v(\d+)", f)[0]) for f in files], reverse=True)
    filename = f"v{versions[0]}.npz"
    return os.path.join(DATASETS_DIR, filename), versions[0]


# Загрузка файла на Google Диск
def upload_to_gdrive(service, file_path, folder_id):
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')


# Скачивание файла с Google Диска
def download_from_gdrive(service, file_id, dest_path):
    request = service.files().get_media(fileId=file_id)
    with open(dest_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

def get_ui_meta():
    return {
        "id": "sync_colab",
        "name": "☁️ Синхронизация с Google Colab",
        "description": "Загрузка датасета на Google Drive и выгрузка модели обратно.",
        "order": 4,
        "parameters": {
            "only_download": {"type": "bool", "default": "False"},
        }
    }

# Основная функция
def main(only_download: bool = False):
    # Аутентификация
    service = authenticate_gdrive()
    # Получение последнего датасета
    dataset_path, version = get_latest_dataset_path()
    if not only_download:
        print(f"[📦] Загружаем датасет: {dataset_path}")

        # Загрузка датасета на Google Диск

        dataset_id = upload_to_gdrive(service, dataset_path, DATASETS_FOLDER_ID)
        print(f"[🌍] Датасет загружен на Google Диск: https://drive.google.com/file/d/{dataset_id}/view?usp=sharing")

        # Инструкция по запуску Colab
        print("\n=== Откройте этот Colab-ноутбук для обучения модели ===")
        print(f"{COLAB_NOTEBOOK_URL}")
        print(
            "\nПосле завершения обучения, убедитесь, что файлы модели и отчёта сохранены в соответствующей папке на Google Диске.")

    # Путь к папке с моделью на Google Диске
    model_folder_name = f"v{version}"
    model_folder_query = f"'{MODELS_FOLDER_ID}' in parents and name = '{model_folder_name}' and mimeType = 'application/vnd.google-apps.folder'"
    results = service.files().list(q=model_folder_query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])

    if not items:
        print(f"[!] Папка для модели версии {version} не найдена на Google Диске.")
        return

    model_folder_id = items[0]['id']

    # Скачивание модели и отчёта
    results = service.files().list(q=f"'{model_folder_id}' in parents", spaces='drive',
                                   fields='files(id, name)').execute()
    items = results.get('files', [])

    if not items:
        print(f"[!] Файлы модели и отчёта не найдены в папке версии {version}.")
        return

    os.makedirs(os.path.join(MODELS_DIR, model_folder_name), exist_ok=True)

    for item in items:
        file_name = item['name']
        file_id = item['id']
        dest_path = os.path.join(MODELS_DIR, model_folder_name, file_name)
        download_from_gdrive(service, file_id, dest_path)
        print(f"[⬇️] Файл {file_name} загружен в {dest_path}")

if __name__ == "__main__":
    main()
