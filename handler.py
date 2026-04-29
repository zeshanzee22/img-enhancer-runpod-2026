import runpod
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
import cloudinary
import cloudinary.uploader
import requests
import io
import logging
import time
import os
from dotenv import load_dotenv

# Load .env locally; ignored on RunPod (env vars set in dashboard)
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Cloudinary config — reads from env vars
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

# Model
MODEL_NAME = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"

logging.info("Loading Swin2SR model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = Swin2SRForImageSuperResolution.from_pretrained(MODEL_NAME)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

if device == "cuda":
    logging.info(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.info("Running on CPU (no GPU available)")


def handler(job):
    """RunPod handler — called once per job."""
    logging.info("New job received.")
    job_input = job.get("input", {})

    # Validate input
    if "image_url" not in job_input:
        logging.error("Missing image_url in input")
        return {"error": "Missing 'image_url' in input"}

    image_url = job_input["image_url"]
    logging.info(f"Processing image from: {image_url}")

    # Download image
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        logging.info(f"Image downloaded. Size: {image.size}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Download failed: {e}")
        return {"error": f"Failed to download image: {str(e)}"}
    except Exception as e:
        logging.error(f"Image open failed: {e}")
        return {"error": f"Failed to open image: {str(e)}"}

    # Memory-safe resize
    MAX_SIZE = 400 if device == "cpu" else 800
    w, h = image.size
    if max(w, h) > MAX_SIZE:
        scale = MAX_SIZE / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        logging.info(f"Resized to: {image.size}")

    original_size = list(image.size)

    # Enhance with Swin2SR
    try:
        logging.info("Running Swin2SR enhancement...")
        stime = time.perf_counter()

        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        output = (
            outputs.reconstruction
            .squeeze()
            .permute(1, 2, 0)
            .clamp(0, 1)
            .cpu()
            .numpy()
        )
        result = Image.fromarray((output * 255).astype(np.uint8))
        etime = time.perf_counter()
        logging.info(f"Enhancement done in {etime - stime:.2f}s. Enhanced size: {result.size}")

    except Exception as e:
        logging.error(f"Enhancement failed: {e}")
        return {"error": f"Enhancement failed: {str(e)}"}

    # Upload to Cloudinary
    try:
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        upload_result = cloudinary.uploader.upload(
            buffer,
            folder="enhanced",
            resource_type="image"
        )
        enhanced_url = upload_result["secure_url"]
        logging.info(f"Uploaded to Cloudinary: {enhanced_url}")

    except Exception as e:
        logging.error(f"Cloudinary upload failed: {e}")
        return {"error": f"Cloudinary upload failed: {str(e)}"}

    # Return result
    return {
        "enhanced_image_url": enhanced_url,
        "original_size": original_size,
        "enhanced_size": list(result.size)
    }


# IMPORTANT: This exact line is required by RunPod's GitHub scanner
runpod.serverless.start({"handler": handler})