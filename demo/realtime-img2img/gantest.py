from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request

import markdown2
import logging
import uuid
import time
from types import SimpleNamespace
import asyncio
import os
import mimetypes
import torch
from io import BytesIO

from config import config, Args
from util import pil_to_frame, bytes_to_pil
from connection_manager import ConnectionManager, ServerFullException
from img2img import Pipeline

# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")

THROTTLE = 1.0 / 120

class App:
    def __init__(self, config: Args, pipeline):
        self.args = config
        self.pipeline = pipeline
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        self.init_app()

    def init_app(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Image Upload and Inference</title>
            </head>
            <body>
                <h2>Upload an image and enter a prompt</h2>
                <input type="text" id="promptInput" placeholder="Enter prompt" style="width:300px;" />
                <br/><br/>
                <input type="file" id="imageInput" accept="image/*" />
                <button onclick="uploadImage()">Upload & Infer</button>
                <p id="status"></p>
                <h3>Output Image:</h3>
                <img id="outputImage" src="" alt="Output will appear here" style="max-width: 500px;" />
                <script>
                    async function uploadImage() {
                        const prompt = document.getElementById('promptInput').value;
                        const input = document.getElementById('imageInput');
                        if (input.files.length === 0) {
                            alert('Please select an image first.');
                            return;
                        }
                        const file = input.files[0];
                        const formData = new FormData();
                        formData.append('image', file);
                        formData.append('prompt', prompt);
                        document.getElementById('status').innerText = 'Uploading and processing...';
                        try {
                            const res = await fetch('/api/infer', {
                                method: 'POST',
                                body: formData
                            });
                            if (res.ok) {
                                document.getElementById('status').innerText = 'Inference complete.';
                                const blob = await res.blob();
                                document.getElementById('outputImage').src = URL.createObjectURL(blob);
                            } else {
                                const data = await res.json();
                                document.getElementById('status').innerText = 'Error: ' + (data.error || res.statusText);
                            }
                        } catch (e) {
                            document.getElementById('status').innerText = 'Fetch error: ' + e;
                        }
                    }
                </script>
            </body>
            </html>
            """

        # ... existing websocket and other endpoints remain unchanged ...

        @self.app.post("/api/infer", response_class=Response)
        async def infer(
            image: UploadFile = File(...),
            prompt: str = Form(...)
        ):
            try:
                # Read input image
                image_bytes = await image.read()
                pil_image = bytes_to_pil(image_bytes)

                # Build parameters including prompt
                base = self.pipeline.InputParams()
                pd = base.dict()
                pd['image'] = pil_image
                pd['prompt'] = prompt
                params = SimpleNamespace(**pd)

                logging.info(f"Running inference with prompt: {prompt}")

                # Run pipeline
                result_image = self.pipeline.predict(params)
                if result_image is None:
                    return JSONResponse({"error": "Inference failed"}, status_code=500)

                # Stream output image
                buf = BytesIO()
                result_image.save(buf, format='PNG')
                buf.seek(0)
                return StreamingResponse(buf, media_type='image/png')
            except Exception as e:
                logging.error(f"Inference error: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        # Static mount unchanged
        public_dir = "./StreamDiffusion/demo/realtime-img2img/frontend/public"
        if not os.path.exists(public_dir):
            os.makedirs(public_dir)
        self.app.mount("/static", StaticFiles(directory=public_dir, html=True), name="public")

# GLOBAL app, so uvicorn main:app works
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16
pipeline = Pipeline(config, device, torch_dtype)
app = App(config, pipeline).app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port, reload=config.reload, ssl_certfile=config.ssl_certfile, ssl_keyfile=config.ssl_keyfile)
