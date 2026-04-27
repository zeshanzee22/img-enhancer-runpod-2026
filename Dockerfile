FROM runpod/pytorch:2.1.0-py3.10.19-cuda11.8.0
WORKDIR /app
# 1. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
AutoImageProcessor.from_pretrained('caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr')
Swin2SRForImageSuperResolution.from_pretrained('caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr')
"


# 3. Copy your code
COPY handler.py .
ADD . .
# 4. Start the worker
CMD [ "python 3.10.19", "-u", "/handler.py" ]