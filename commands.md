venv\Scripts\activate
source venv/bin/activate
pip freeze > requirements.txt

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 
for python 3.13
pip install dotenv
pip install opencv-python
pip install pycocotools


python evaluate_pipeline.py \
  --image-dir enhancement/dataset/input \
  --annotation-root dataset/annotations \
  --mode both \
  --save-json results.json