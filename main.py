"""
FastAPI application for detecting and cropping documents from images or PDF files.
"""
from __future__ import annotations

import base64
import io
from typing import List

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


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_document_quads_multi(
    image_bgr: np.ndarray,
    roi_top_ratio: float = 0.25,
    roi_bottom_ratio: float = 0.95,
    min_area_ratio: float = 0.02,
    min_aspect: float = 0.2,
    max_aspect: float = 5.0,
    max_docs: int = 10,
) -> list[np.ndarray]:
    """Detect multiple document quadrilaterals in a single image."""

    h, w = image_bgr.shape[:2]
    y0 = int(h * roi_top_ratio)
    y1 = int(h * roi_bottom_ratio)
    roi = image_bgr[y0:y1, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = float(roi.shape[0] * roi.shape[1])

    candidates: list[tuple[np.ndarray, float]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * roi_area:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype("float32")
        (_, _), (w_box, h_box), _ = rect

        if w_box <= 0 or h_box <= 0:
            continue

        aspect = max(w_box, h_box) / min(w_box, h_box)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        box[:, 1] += y0

        quad = order_points(box)
        candidates.append((quad, area))

    candidates.sort(key=lambda c: c[1], reverse=True)
    quads = [c[0] for c in candidates[:max_docs]]

    return quads


def warp_quad(image_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Apply perspective transform to obtain a top-down view of the quad region."""

    rect = order_points(quad)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image_bgr, matrix, (max_width, max_height))
    return warped


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


def process_pdf_to_bgr_images(file_bytes: bytes) -> List[np.ndarray]:
    """Convert each page of a PDF to a BGR numpy image using PyMuPDF."""

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images: List[np.ndarray] = []
    for page in doc:
        pix = page.get_pixmap(dpi=200, colorspace=fitz.csRGB)
        array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
        bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        images.append(bgr)
    return images


def load_image_file_to_bgr(file_bytes: bytes) -> np.ndarray:
    """Load a single image file into a BGR numpy array."""

    image_array = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return image_bgr


def process_page_documents(page_bgr: np.ndarray, image_format: str, page_index: int) -> List[dict]:
    """Detect, warp, and encode documents for a single page image."""

    quads = detect_document_quads_multi(page_bgr)
    if not quads:
        height, width = page_bgr.shape[:2]
        quads = [
            np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype="float32",
            )
        ]

    documents: List[dict] = []
    mime_type = SUPPORTED_OUTPUT_FORMATS[image_format]
    for doc_index, quad in enumerate(quads):
        warped = warp_quad(page_bgr, quad)
        pil_image = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        data_uri = image_to_data_uri(pil_image, mime_type)
        documents.append(
            {
                "page_index": page_index,
                "doc_index": doc_index,
                "mime_type": mime_type,
                "width": pil_image.width,
                "height": pil_image.height,
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

    if file.content_type == "application/pdf":
        pages = process_pdf_to_bgr_images(file_bytes)
    else:
        pages = [load_image_file_to_bgr(file_bytes)]

    all_documents: List[dict] = []
    for page_index, page_bgr in enumerate(pages):
        all_documents.extend(process_page_documents(page_bgr, image_format, page_index))

    return JSONResponse(content={"documents": all_documents})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
