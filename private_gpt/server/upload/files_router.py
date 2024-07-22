import os
import logging
from typing import List
from private_gpt.constants import FILES_DIR
from private_gpt.settings.settings import settings
from private_gpt.ui.ui import PrivateGptUi, CHUNK_SIZE, CHUNK_OVERLAP
from fastapi import APIRouter, Request, UploadFile, HTTPException, File


# Not authentication or authorization required to get the health status.
files_router = APIRouter()


def save_files(files: List[UploadFile], dict_form, is_upload: bool = True) -> List[str]:
    file_locations = []
    for file in files:
        if file.filename is None:
            raise HTTPException(400, "No file name provided")
        file_location = f"{FILES_DIR}/{dict_form['uuid']}_{file.filename}"
        if not is_upload:
            path = f"{FILES_DIR}/{dict_form['uuid_return']}_{file.filename}"
            os.remove(path) if os.path.exists(path) else None
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        file_locations.append(file_location)
    return file_locations


def handle_active_status(service, files, dict_form):
    if dict_form["uuid_return"] == '':
        list_files = save_files(files, dict_form, is_upload=True)
        if service:
            return service.upload_file(list_files, CHUNK_SIZE, CHUNK_OVERLAP, dict_form["uuid"])
        return "Загружены документы",
    else:
        list_files = save_files(files, dict_form, is_upload=False)
        if service:
            return service.update_file(list_files, CHUNK_SIZE, CHUNK_OVERLAP, dict_form["uuid"], dict_form["uuid_return"])
        return "Обновлены документы",


def handle_obsolete_status(service, files, dict_form):
    for file in files:
        path = f"{FILES_DIR}/{dict_form['uuid']}_{file.filename}"
        os.remove(path) if os.path.exists(path) else None
    if service:
        return service.delete_file([dict_form["uuid"]])
    return "Удалены документы",


@files_router.post("/upload_files", tags=["Files"])
def ingest(request: Request, files: List[UploadFile] = File(...)):
    logging.info(f"Files {files}")
    files_name = [f.filename for f in files]
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
            status = "fail"
            logging.error(f"details: {{'file': '{files_name}', 'message': 'Неизвестный статус', 'status': {status}}}")
            return {"file": files_name, "message": "Неизвестный статус", "status": status}
    except Exception as ex:
        status = "fail"
        logging.error(f"Exception is {ex}, "
                      f"details: {{'file': '{files_name}', 'message': 'Файл не был загружен', 'status': {status}}}")
        return {"file": files_name, "message": "Файл не был загружен", "status": status}
    logging.info(f"details: {{'file': '{files_name}', 'message': {message[0]}, 'status': {status}}}")
    return {"file": files_name, "message": message[0], "status": status}
