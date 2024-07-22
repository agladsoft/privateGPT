import os
import logging
from private_gpt.constants import FILES_DIR

from private_gpt.settings.settings import settings
from private_gpt.ui.ui import PrivateGptUi, CHUNK_SIZE, CHUNK_OVERLAP
from fastapi import APIRouter, Request, UploadFile, HTTPException, File


# Not authentication or authorization required to get the health status.
files_router = APIRouter()


def save_files(file: UploadFile, dict_form, is_upload: bool = True) -> str:
    if file.filename is None or file.filename == "":
        raise FileNotFoundError("Файл не был загружен")
    file_location = f"{FILES_DIR}/{dict_form['uuid']}_{file.filename}"
    if not is_upload:
        path = f"{FILES_DIR}/{dict_form['uuid_return']}_{file.filename}"
        os.remove(path) if os.path.exists(path) else None
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    return file_location


def handle_active_status(service, file, dict_form):
    if dict_form["uuid_return"] == '':
        file_name = save_files(file, dict_form, is_upload=True)
        if service:
            return service.upload_file([file_name], CHUNK_SIZE, CHUNK_OVERLAP, dict_form["uuid"])
        return "Загружен документ",
    else:
        file_name = save_files(file, dict_form, is_upload=False)
        if service:
            return service.update_file([file_name], CHUNK_SIZE, CHUNK_OVERLAP, dict_form["uuid"], dict_form["uuid_return"])
        return "Обновлен документ",


def handle_obsolete_status(service, file, dict_form):
    path = f"{FILES_DIR}/{dict_form['uuid']}_{file.filename}"
    os.remove(path) if os.path.exists(path) else None
    if service:
        return service.delete_file([dict_form["uuid"]])
    return "Удален документ",


@files_router.post("/upload_files", tags=["Files"])
def ingest(request: Request, files: UploadFile = File(...)):
    logging.info(f"Files {files}")
    file_name = files.filename
    if settings().ui.enabled:
        service = request.state.injector.get(PrivateGptUi)
    else:
        service = None
    dict_form = request._form._dict
    logging.info(dict_form)

    status = "success"
    try:
        if dict_form["status"] == "Действующий":
            message = handle_active_status(service, files, dict_form)
        elif dict_form["status"] == "Утратил силу":
            message = handle_obsolete_status(service, files, dict_form)
        else:
            logging.error(f"details: {{'file': '{file_name}', 'message': 'Неизвестный статус', 'status': {status}}}")
            save_files(files, dict_form, is_upload=True)
            return {"file": file_name, "message": "Неизвестный статус", "status": status}
    except Exception as ex:
        status = "fail"
        logging.error(f"Exception is {ex}, "
                      f"details: {{'file': '{file_name}', 'message': {ex}, 'status': {status}}}")
        return {"file": file_name, "message": str(ex), "status": status}
    logging.info(f"details: {{'file': '{file_name}', 'message': {message[0]}, 'status': {status}}}")
    return {"file": file_name, "message": message[0], "status": status}
