import os
import logging
from typing import List
from private_gpt.constants import FILES_DIR
from private_gpt.ui.ui import PrivateGptUi, CHUNK_SIZE, CHUNK_OVERLAP
from fastapi import APIRouter, Request, UploadFile, HTTPException, File
from private_gpt.server.ingest.ingest_service import IngestService, logger

# Not authentication or authorization required to get the health status.
files_router = APIRouter()


@files_router.post("/upload_files", tags=["Files"])
def ingest(request: Request, files: List[UploadFile] = File(...)):
    logging.info(f"Files {files}")
    service = request.state.injector.get(PrivateGptUi)
    dict_form = request._form._dict
    logging.info(dict_form)
    list_files = []
    list_files_obj = []

    for file in files:
        if file.filename is None:
            raise HTTPException(400, "No file name provided")
        file_location = f"{FILES_DIR}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        list_files.append(file_location)

    # for file in list_files:
    status = "success"
    if dict_form["status"] == "Действующий" and dict_form["uuid_return"] == '':  # insert
        try:
            message = service._ingest_service.bulk_ingest(list_files, CHUNK_SIZE, CHUNK_OVERLAP, dict_form["uuid"])
        except Exception as ex:
            logging.error(f"Exception is {ex}")
            message, status = "The file was not uploaded", "fail"
        list_files_obj.append({
            "file": list_files,
            "message": message,
            "status": status
        })
    elif dict_form["status"] == "Действующий" and dict_form["uuid_return"] != '':  # update
        try:
            message = service.update_doc(list_files, CHUNK_SIZE, CHUNK_OVERLAP, dict_form["uuid"], dict_form["uuid_return"])
        except Exception as ex:
            logging.error(f"Exception is {ex}")
            message, status = "The file was not uploaded", "fail"
        list_files_obj.append({
            "file": list_files,
            "message": message,
            "status": status
        })
    elif dict_form["status"] == "Утратил силу":  # delete
        service.delete_doc([os.path.basename(file) for file in list_files])
        list_files_obj.append({
            "file": list_files,
            "message": "Успешно удален",
            "status": status
        })
    else:
        pass
    return list_files_obj
