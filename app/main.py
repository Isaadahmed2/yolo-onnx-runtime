from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.concurrency import run_in_threadpool
import shutil
import os
import uuid
from .video_processing import process_video_with_yolo
from .logging_config import get_logger

logger = get_logger(__name__)


app = FastAPI()

UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"

# Ensure the directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/process-video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received video upload: {file.filename}")
    if not file.content_type.startswith("video/"):
        logger.warning(f"Unsupported file type uploaded: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Please upload a video.",
        )

    # Create a unique filename to avoid conflicts
    unique_id = uuid.uuid4()
    input_filename = f"{unique_id}_{file.filename}"
    output_filename = f"{unique_id}_processed.mp4"
    input_path = os.path.join(UPLOADS_DIR, input_filename)
    output_path = os.path.join(OUTPUTS_DIR, output_filename)

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    logger.info(f"Processing video {input_filename}...")
    processed_path = await run_in_threadpool(
        process_video_with_yolo, input_path, output_path
    )

    if not processed_path:
        logger.error(f"Failed to process video {input_filename}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process video.",
        )

    logger.info(f"Successfully processed {input_filename}, output saved to {output_path}")
    return {
        "message": "Video processed successfully",
        "input_filename": input_filename,
        "output_filename": output_filename,
        "output_path": processed_path,
    }
