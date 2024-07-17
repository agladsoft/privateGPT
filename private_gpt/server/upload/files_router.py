import os
import logging
from typing import List
from private_gpt.constants import FILES_DIR
from private_gpt.ui.ui import PrivateGptUi, CHUNK_SIZE, CHUNK_OVERLAP
from fastapi import APIRouter, Request, UploadFile, HTTPException, File


# Not authentication or authorization required to get the health status.
files_router = APIRouter()


def save_files(files: List[UploadFile], directory: str) -> List[str]:
    file_locations = []
    for file in files:
        if file.filename is None:
            raise HTTPException(400, "No file name provided")
        file_location = f"{directory}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        file_locations.append(file_location)
    return file_locations


def handle_active_status(service, list_files, chunk_size, chunk_overlap, uuid, uuid_return):
    if uuid_return == '':
        return service.upload_file(list_files, chunk_size, chunk_overlap, uuid)
    else:
        return service.update_file(list_files, chunk_size, chunk_overlap, uuid, uuid_return)


def handle_obsolete_status(service, uuid):
    return service.delete_file(uuid)


@files_router.post("/upload_files", tags=["Files"])
def ingest(request: Request, files: List[UploadFile] = File(...)):
    logging.info(f"Files {files}")
    service = request.state.injector.get(PrivateGptUi)
    dict_form = request._form._dict
    logging.info(dict_form)

    list_files = save_files(files, FILES_DIR)

    status = "success"
    try:
        if dict_form["status"] == "Действующий":
            message = handle_active_status(service, list_files, CHUNK_SIZE, CHUNK_OVERLAP, dict_form["uuid"], dict_form["uuid_return"])
        elif dict_form["status"] == "Утратил силу":
            message = handle_obsolete_status(service, [dict_form["uuid"]])
        else:
            return {"file": list_files, "message": "Неизвестный статус", "status": "fail"}
    except Exception as ex:
        logging.error(f"Exception is {ex}")
        return {"file": list_files, "message": "Файл не был загружен", "status": "fail"}

    return {"file": list_files, "message": message[0], "status": status}
