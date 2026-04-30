# DermaScanX

 Content: High‑quality dermatoscopic skin lesion images covering major skin lesion types (including melanoma, benign keratosis, basal cell carcinoma, nevus, vascular lesions, etc.)
@link : https://huggingface.co/datasets/marmal88/skin_cancer


Harward Skin DatasetLink : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

ISIC DatasetLink : https://challenge2024.isic-archive.com/
                 : https://www.isic-archive.com/
                 
PAD DatasetLink : https://data.mendeley.com/datasets/zr7vgbcyr2/1

Kaggel DatasetLink : https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection

Kaggle DatasetLink(HAM10000) : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Kaggle Dataset Benign:  https://www.kaggle.com/datasets/abbasashwal/isic-2020-dataset

Frontend RepoLink : https://github.com/JITEN-BHARGA/skin-cancer-frontend.git

Huggingface Model Link : https://huggingface.co/spaces/jiten-333/Skin_Cancer

our website Link : https://skin-cancer-frontend-ffho.vercel.app/

# our code running instruction :

first make .env file 
variables :
1. NEXT_PUBLIC_API_URL
2. GROQ_API_KEY

## first open backend folder integrated terminal

download requirement

pip install -r requirements.txt

server run command : python uvicorn main:app --reload 

## open frontend integreted terminal

run 
1. npm install
2. npm run dev
