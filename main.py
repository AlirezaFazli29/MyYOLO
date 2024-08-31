import time
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import io
from PIL import Image
import cv2
import numpy as np
import base64


app = FastAPI()

model = YOLO('yolov8n.pt')

def time_to_str(Time):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(Time) )

@app.get("/")
def root():
    return {"message": "Hello World",
            'api openning time': time_to_str(time.time())}

def process_img(byte_data):
    image = Image.open(io.BytesIO(byte_data))
    if image.mode == 'RGBA': image = image.convert('RGB')
    return image

def run_yolo(image, conf):
    results = model(image, conf=conf)
    bgr_img = results[0].plot()
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    _, encoded_img = cv2.imencode('.jpg', rgb_img)
    return io.BytesIO(encoded_img.tobytes())

def resize(img_arr, cons=300):
    h,v = img_arr.shape[:2]
    if v<h: 
        r = v/h
        return cv2.resize(img_arr,(int(cons*r), cons))
    elif v>h: 
        r = h/v
        return cv2.resize(img_arr, (cons, int(cons*r)))
    else:
        return cv2.resize(img_arr,(cons, cons))
    

@app.post('/upload_n_returnimage/{conf}')
async def upload_n_returnimage(conf: float=0.2 ,file: UploadFile = File(...)):
    byte_data = await file.read()
    img = process_img(byte_data)
    img_arr = np.array(img)
    resized_img = resize(img_arr, 800)
    result = run_yolo(resized_img, conf=conf)
    return StreamingResponse(result, media_type='image/jpeg')

@app.post('/uoload_n_returnbox/{conf}')
async def uoload_n_returnbox(conf: float=0.2 ,file: UploadFile = File(...)):
    byte_data = await file.read()
    img = process_img(byte_data)
    img_arr = np.array(img)
    resized_img = resize(img_arr, 800)
    
    results = model(resized_img, conf=conf)

    bounding_boxes = []
    for r in results:
        for box in r.boxes:
            bounding_boxes.append({
                'x1': int(box.xyxy[0][0]),
                'y1': int(box.xyxy[0][1]),
                'x2': int(box.xyxy[0][2]),
                'y2': int(box.xyxy[0][3]),
                'x' : int(box.xywh[0][0]),
                'y' : int(box.xywh[0][1]),
                'w' : int(box.xywh[0][2]),
                'h' : int(box.xywh[0][3]),
                'confidence': float(box.conf[0]),
                'class': r.names[int(box.cls[0])]
            })

    return {'BBs': bounding_boxes}

# UPLOAD_DIRECTORY = "./uploaded_images"

# if not os.path.exists(UPLOAD_DIRECTORY):
#     os.makedirs(UPLOAD_DIRECTORY)


# @app.get('/')
# def root():
#     return {"message": "Hello World"}

# @app.post('/upload/')
# async def upload_file(file: UploadFile = File(...)):
#     file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
#     with open(file_location, 'wb') as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     return {'opened filename': file.filename}

# @app.get('/images/{filename}')
# async def get_image(filename: str):
#     file_location =os.path.join(UPLOAD_DIRECTORY, filename)
#     if os.path.exists(file_location):
#         return FileResponse(file_location)
#     else: 
        # return {"error": "File not found"}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)