# yolo-onnx-runtime
Implementation of yolo onnx runtime. (computer vision)

---

## YOLO Sports Ball Glow Effect API

This project is a FastAPI backend that takes a short video, detects sports balls using a YOLOv8 ONNX model, and applies a glowing effect to the detected balls.

### Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── logging_config.py
│   ├── main.py
│   ├── utils.py
│   └── video_processing.py
├── logs/
├── outputs/
├── uploads/
├── yolo_model/
│   └── yolov8n.onnx
├── CLAUDE.md
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Setup and Installation

#### Prerequisites

- Docker and Docker Compose

#### Running the Application

1.  **Build and run the Docker container:**

    ```bash
    docker-compose up --build
    ```

    The application will be available at `http://localhost:8000`.

2.  **API Documentation:**

    Interactive API documentation (Swagger UI) is available at `http://localhost:8000/docs`.

### Local Development (Without Docker)

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install `uv`:**

    ```bash
    pip install uv
    ```

3.  **Install dependencies using `uv`:**

    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Run the FastAPI application:**

    ```bash
    uvicorn app.main:app --reload
    ```

### API Usage

#### Process a Video

- **Endpoint:** `/process-video/`
- **Method:** `POST`
- **Body:** `multipart/form-data`
- **Parameter:** `file` (the video file to be processed)

##### Example using `curl`:

```bash
curl -X POST "http://localhost:8000/process-video/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/video.mp4"
```

The processed video will be saved in the `outputs` directory. The API response will contain the path to the output file.

