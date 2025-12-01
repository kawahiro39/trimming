"""
FastAPI application for detecting and cropping documents from images or PDF files.
"""
from __future__ import annotations

import base64
import io
from typing import List, Tuple

import cv2
import fitz
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI(title="Document Cropper API")

SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png"}
SUPPORTED_OUTPUT_FORMATS = {"png": "image/png", "jpeg": "image/jpeg"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

def image_to_data_uri(pil_image: Image.Image, mime_type: str = "image/png") -> str:
    """Convert a Pillow image to a data URI string."""
    buffer = io.BytesIO()
    format_name = "PNG" if mime_type == "image/png" else "JPEG"
    pil_image.save(buffer, format=format_name)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_str}"

def read_upload_file(upload_file: UploadFile) -> bytes:
    """Read uploaded file content and validate size."""
    data = upload_file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File is empty.")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds limit (25MB).")
    return data

def process_pdf_to_images(file_bytes: bytes) -> List[Image.Image]:
    """Convert each page of a PDF to a Pillow RGB image using PyMuPDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images: List[Image.Image] = []
    for page in doc:
        # Render page to a pixmap at 200 DPI for clearer edges.
        pix = page.get_pixmap(dpi=200, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def load_image_file(file_bytes: bytes) -> Image.Image:
    """Load a single image file into a Pillow image."""
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc
    return image

def resize_for_processing(image: np.ndarray, max_dim: int = 2000) -> Tuple[np.ndarray, float, float]:
    """Resize image while keeping aspect ratio so that longest side ~= max_dim."""
    height, width = image.shape[:2]
    scale = max(height, width) / float(max_dim)
    if scale <= 1:
        return image, 1.0, 1.0
    new_width = int(width / scale)
    new_height = int(height / scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    ratio_x = width / float(new_width)
    ratio_y = height / float(new_height)
    return resized, ratio_x, ratio_y

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def calculate_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union for two bounding boxes."""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0

def detect_documents(image: np.ndarray) -> List[np.ndarray]:
    """Detect document quadrilaterals in the image using contour detection."""
    resized, ratio_x, ratio_y = resize_for_processing(image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = resized.shape[0] * resized.shape[1]

    candidates: List[Tuple[np.ndarray, float, Tuple[int, int, int, int]]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.02 * image_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        approx_points = approx.reshape(4, 2).astype("float32")
        rect = cv2.boundingRect(approx_points)
        box = (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
        width = rect[2]
        height = rect[3]
        aspect_ratio = width / float(height) if height else 0.0
        a4_ratio = 1 / 1.414  # width / height
        ratio_score = 1 - abs(aspect_ratio - a4_ratio)
        score = area * max(ratio_score, 0.0)
        candidates.append((approx_points, score, box))

    # Sort by score (area + aspect ratio closeness)
    candidates.sort(key=lambda c: c[1], reverse=True)

    selected: List[np.ndarray] = []
    selected_boxes: List[Tuple[int, int, int, int]] = []
    for points, _, box in candidates:
        if any(calculate_iou(box, existing) > 0.3 for existing in selected_boxes):
            continue
        ordered = order_points(points)
        ordered[:, 0] *= ratio_x
        ordered[:, 1] *= ratio_y
        selected.append(ordered)
        selected_boxes.append((
            int(ordered[:, 0].min()),
            int(ordered[:, 1].min()),
            int(ordered[:, 0].max()),
            int(ordered[:, 1].max()),
        ))
    return selected

def crop_and_warp_document(image: np.ndarray, points: np.ndarray) -> Image.Image:
    """Perform perspective transform to obtain a top-down view of the document."""
    rect = order_points(points)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    target_width = max_width
    target_height = max_height

    # Adjust output to approximate A4 ratio if the detected contour is close to it
    if max_width and max_height:
        current_ratio = max_height / float(max_width)
        desired_ratio = 1.414
        if abs(current_ratio - desired_ratio) > 0.05:
            target_width = int(max(max_width, max_height / desired_ratio))
            target_height = int(max(max_height, max_width * desired_ratio))

    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (target_width, target_height))
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

def process_page(image: Image.Image, image_format: str) -> List[dict]:
    """Process a single page image and return cropped document metadata."""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    candidates = detect_documents(cv_image)

    if not candidates:
        h, w = cv_image.shape[:2]
        candidates = [np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")]

    documents: List[dict] = []
    for doc_index, points in enumerate(candidates):
        cropped = crop_and_warp_document(cv_image, points)
        mime_type = SUPPORTED_OUTPUT_FORMATS[image_format]
        data_uri = image_to_data_uri(cropped, mime_type)
        width, height = cropped.size
        documents.append(
            {
                "doc_index": doc_index,
                "mime_type": mime_type,
                "width": width,
                "height": height,
                "data_uri": data_uri,
            }
        )
    return documents

@app.post("/crop-documents")
async def crop_documents(file: UploadFile = File(...), image_format: str = "png") -> JSONResponse:
    """Detect and crop documents from an uploaded PDF or image file."""
    if image_format not in SUPPORTED_OUTPUT_FORMATS:
        raise HTTPException(status_code=400, detail="Unsupported image_format. Use 'png' or 'jpeg'.")

    if file.content_type not in SUPPORTED_IMAGE_MIME_TYPES and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and image are allowed.")

    file_bytes = read_upload_file(file)

    pages: List[Image.Image]
    if file.content_type == "application/pdf":
        pages = process_pdf_to_images(file_bytes)
    else:
        pages = [load_image_file(file_bytes)]

    all_documents: List[dict] = []
    for page_index, page in enumerate(pages):
        page_documents = process_page(page, image_format)
        for doc in page_documents:
            doc["page_index"] = page_index
            all_documents.append(doc)

    return JSONResponse(content={"documents": all_documents})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
