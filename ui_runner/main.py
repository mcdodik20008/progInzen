# Обновим main.py с сортировкой по полю "order" и учётом стилей
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import importlib.util
import os
import inspect
from typing import List

app = FastAPI()
app.mount("/static", StaticFiles(directory="ui_runner/static"), name="static")
templates = Jinja2Templates(directory="ui_runner/templates")

BLOCKS_DIRS  = ["core", "tools", "inference", ]
loaded_blocks = {}


def recursive_find_blocks_from_dirs(dirs: List[str]):
    blocks = []
    for base_path in dirs:
        for root, _, files in os.walk(base_path):
            for filename in files:
                if filename.endswith(".py"):
                    path = os.path.join(root, filename)
                    name = os.path.relpath(path, base_path).replace(os.sep, ".")[:-3]  # dotted path without .py
                    spec = importlib.util.spec_from_file_location(name, path)
                    module = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(module)
                        if hasattr(module, "get_ui_meta") and inspect.isfunction(module.get_ui_meta):
                            meta = module.get_ui_meta()
                            meta["module"] = module
                            meta["__path"] = path
                            meta["order"] = meta.get("order", 999)
                            blocks.append(meta)
                    except Exception as e:
                        print(f"[⚠️] Не удалось загрузить {path}: {e}")
    return sorted(blocks, key=lambda m: m["order"])


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    blocks = recursive_find_blocks_from_dirs(BLOCKS_DIRS)
    return templates.TemplateResponse("index.html", {"request": request, "blocks": blocks})


@app.get("/block/{block_id}", response_class=HTMLResponse)
def block_detail(request: Request, block_id: str):
    module = loaded_blocks.get(block_id)
    if not module:
        return RedirectResponse("/")
    meta = module.get_ui_meta()
    return templates.TemplateResponse("block.html", {"request": request, "block": meta})


@app.post("/run/{block_id}")
async def run_block(block_id: str, request: Request):
    form = await request.form()
    module = loaded_blocks.get(block_id)
    if not module:
        return RedirectResponse("/")

    kwargs = {key: form[key] for key in form}
    meta = module.get_ui_meta()
    params_info = meta.get("parameters", {})
    for key, param_meta in params_info.items():
        if param_meta["type"] == "int":
            kwargs[key] = int(kwargs[key])
        elif param_meta["type"] == "float":
            kwargs[key] = float(kwargs[key])
        # string — по умолчанию

    try:
        output = module.main(**kwargs)
        result = f"✅ Запуск {block_id} завершён"
    except Exception as e:
        result = f"❌ Ошибка: {e}"

    return templates.TemplateResponse("result.html", {
        "request": request,
        "block_id": block_id,
        "result": result,
        "args": kwargs
    })
