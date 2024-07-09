from typing import List
from private_gpt.constants import FILES_DIR
from fastapi import APIRouter, Request, UploadFile, HTTPException, File
from private_gpt.server.ingest.ingest_service import IngestService

# Not authentication or authorization required to get the health status.
files_router = APIRouter()


@files_router.post("/upload", tags=["Files"])
def ingest(request: Request, files: List[UploadFile] = File(...)):
    service = request.state.injector.get(IngestService)
    list_files = []
    for file in files:
        if file.filename is None:
            raise HTTPException(400, "No file name provided")

        file_location = f"{FILES_DIR}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        status = "success"
        # try:
        #     message = service.ingest(file_location)
        # except Exception as ex:
        #     logging.error(f"Exception is {ex}")
        #     message, status = "The file was not uploaded", "fail"
        list_files.append({
            "file": file.filename,
            # "message": message,
            "status": status
        })
    return list_files
