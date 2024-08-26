from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import shutil

app = FastAPI()

UPLOAD_DIRECTORY = "./uploaded_images"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


@app.get('/')
def root():
    return {"message": "Hello World"}

@app.post('/upload/')
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_location, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {'opened filename': file.filename}

@app.get('/images/{filename}')
async def get_image(filename: str):
    file_location =os.path.join(UPLOAD_DIRECTORY, filename)
    if os.path.exists(file_location):
        return FileResponse(file_location)
    else: 
        return {"error": "File not found"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)