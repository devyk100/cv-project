venv\Scripts\activate
source venv/bin/activate
pip freeze > requirements.txt

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 
for python 3.13