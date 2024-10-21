import time
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import services.ai_services.my_models as Models
import services.image_handler.utils as IMUtils
import services.ai_services.inferences as INF
import services.ai_services.utils as AIUtils
import base64
import io
from PIL import Image


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


device = AIUtils.select_device()
yolo = Models.YOLOv8(Models.YoloType.Custom.Plate_best)
unet = Models.Plate_Unet(Models.UNetType.Corner_best, device=device)
yolo_inference = INF.YOLOInference(yolo)
unet_inference = INF.PlateUNetInference(unet)


@app.get("/")
async def read_root(request: Request):
    """
    Renders the main page with buttons to navigate to different tasks.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/yolo", response_class=HTMLResponse)
async def get_yolo_page(request: Request):
    """
    Renders the YOLO upload page.
    """
    return templates.TemplateResponse("yolo.html", {"request": request})

@app.post("/yolo_process")
async def yolo_process(
    request: Request,
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
):
    image = Image.open(file.file)
    cropped_plates, image_with_boxes = yolo_inference.run_full_pipeline(image, conf=conf_threshold)

    img_byte_arr = io.BytesIO()
    Image.fromarray(image_with_boxes[0]).save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    image_with_boxes_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Convert cropped plates to a list of byte streams
    plates_data = []
    for plate in cropped_plates:
        plate_img = Image.fromarray(IMUtils.convert_to_rgb(plate))
        plate_byte_arr = io.BytesIO()
        plate_img.save(plate_byte_arr, format='PNG')
        plate_byte_arr.seek(0)
        plates_data.append(base64.b64encode(plate_byte_arr.getvalue()).decode('utf-8'))

    return templates.TemplateResponse(
        "yolo_results.html",
        {
            "request": request,
            "image_with_boxes": image_with_boxes_base64,
            "cropped_plates": plates_data
        }
    )

@app.get("/unet", response_class=HTMLResponse)
async def get_unet_page(request: Request):
    """
    Renders the U-Net upload page.
    """
    return templates.TemplateResponse("unet.html", {"request": request})

@app.post("/unet_process")
async def unet_process(
    request: Request,
    file: UploadFile = File(...),
):
    # Open the uploaded image
    image = Image.open(file.file)
    
    # Run U-Net inference
    rectified_image = unet_inference.run_full_pipeline(image)

    # Convert the original image to base64 format
    img_byte_arr_original = io.BytesIO()
    image.save(img_byte_arr_original, format='PNG')
    img_byte_arr_original.seek(0)
    original_image_base64 = base64.b64encode(img_byte_arr_original.getvalue()).decode('utf-8')

    # Convert the rectified image to base64 format
    img_byte_arr = io.BytesIO()
    Image.fromarray(rectified_image[0]).save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    rectified_image_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    

    # Return the results page with both images
    return templates.TemplateResponse(
        "unet_results.html",
        {
            "request": request,
            "original_image": original_image_base64,
            "rectified_image": rectified_image_base64,
        }
    )

@app.get("/full_pipeline", response_class=HTMLResponse)
async def get_full_pipeline_page(request: Request):
    """
    Renders the full pipeline upload page.
    """
    return templates.TemplateResponse("full_pipeline.html", {"request": request})

@app.post("/full_pipeline_process")
async def full_pipeline_process(
    request: Request,
    file: UploadFile = File(...),
    conf_threshold: float = Form(...),
):
    image = Image.open(file.file)
    cropped_plates, image_with_boxes = yolo_inference.run_full_pipeline(image, conf=conf_threshold)

    img_byte_arr = io.BytesIO()
    Image.fromarray(image_with_boxes[0]).save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    image_with_boxes_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Convert cropped plates to a list of byte streams
    plates_data = []
    for plate in cropped_plates:
        plate_img = Image.fromarray(IMUtils.convert_to_rgb(plate))
        plate_byte_arr = io.BytesIO()
        plate_img.save(plate_byte_arr, format='PNG')
        plate_byte_arr.seek(0)
        plates_data.append(base64.b64encode(plate_byte_arr.getvalue()).decode('utf-8'))

    # Feed cropped plates to U-Net
    rectified_plates = unet_inference.run_full_pipeline(cropped_plates)

    # Convert rectified plates to a list of byte streams
    rectified_data = []
    for rectified in rectified_plates:
        rectified_img = Image.fromarray(IMUtils.convert_to_rgb(rectified))
        rectified_byte_arr = io.BytesIO()
        rectified_img.save(rectified_byte_arr, format='PNG')
        rectified_byte_arr.seek(0)
        rectified_data.append(base64.b64encode(rectified_byte_arr.getvalue()).decode('utf-8'))

    return templates.TemplateResponse(
        "full_pipeline_results.html",
        {
            "request": request,
            "image_with_boxes": image_with_boxes_base64,
            "cropped_plates": plates_data,
            "rectified_plates": rectified_data
        }
    )


# def time_to_str(Time):
#     return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(Time) )

# @app.get("/")
# def root():
#     return {"message": "Hello World",
#             'api openning time': time_to_str(time.time())}

# def process_img(byte_data):
#     image = Image.open(io.BytesIO(byte_data))
#     if image.mode == 'RGBA': image = image.convert('RGB')
#     return image

# def run_yolo(image, conf):
#     results = model(image, conf=conf)
#     bgr_img = results[0].plot()
#     rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
#     _, encoded_img = cv2.imencode('.jpg', rgb_img)
#     return io.BytesIO(encoded_img.tobytes())

# def resize(img_arr, cons=300):
#     h,v = img_arr.shape[:2]
#     if v<h: 
#         r = v/h
#         return cv2.resize(img_arr,(int(cons*r), cons))
#     elif v>h: 
#         r = h/v
#         return cv2.resize(img_arr, (cons, int(cons*r)))
#     else:
#         return cv2.resize(img_arr,(cons, cons))
    

# @app.post('/upload_n_returnimage/{conf}')
# async def upload_n_returnimage(conf: float=0.2 ,file: UploadFile = File(...)):
#     byte_data = await file.read()
#     img = process_img(byte_data)
#     img_arr = np.array(img)
#     resized_img = resize(img_arr, 800)
#     result = run_yolo(resized_img, conf=conf)
#     return StreamingResponse(result, media_type='image/jpeg')

# @app.post('/uoload_n_returnbox/{conf}')
# async def uoload_n_returnbox(conf: float=0.2 ,file: UploadFile = File(...)):
#     byte_data = await file.read()
#     img = process_img(byte_data)
#     img_arr = np.array(img)
#     resized_img = resize(img_arr, 800)
    
#     results = model(resized_img, conf=conf)

#     bounding_boxes = []
#     for r in results:
#         for box in r.boxes:
#             bounding_boxes.append({
#                 'x1': int(box.xyxy[0][0]),
#                 'y1': int(box.xyxy[0][1]),
#                 'x2': int(box.xyxy[0][2]),
#                 'y2': int(box.xyxy[0][3]),
#                 'x' : int(box.xywh[0][0]),
#                 'y' : int(box.xywh[0][1]),
#                 'w' : int(box.xywh[0][2]),
#                 'h' : int(box.xywh[0][3]),
#                 'confidence': float(box.conf[0]),
#                 'class': r.names[int(box.cls[0])]
#             })

#     return {'BBs': bounding_boxes}

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