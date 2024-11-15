import os
from pathlib import Path
from byaldi import RAGMultiModalModel
import torch
from dotenv import load_dotenv

import os
load_dotenv()

actual_key = os.getenv('token')
model_name = os.getenv('model_name')
index_name = os.getenv('index_name')
path= os.getenv('path')
print(actual_key, model_name, index_name)

# os.environ['PATH'] += os.pathsep + r"C:\Users\sachin.kavlekar\Desktop\Repo\Vision_new_multi\Release-24.08.0-0\poppler-24.08.0\Library\bin"


import requests
from huggingface_hub import login
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

login(actual_key)
# Initialize RAGMultiModalModel
model = RAGMultiModalModel.from_pretrained(model_name)

model.index(input_path=Path(r""+str(path)),
    index_name=index_name,
    store_collection_with_index=True, # Stores base64 images along with the vectors
    overwrite=True
)

query = "how much is the total cost?"
results = model.search(query, k=5)

print(f"Search results for '{query}':")
for result in results:
    print(f"Doc ID: {result.doc_id}, Page: {result.page_num}, Score: {result.score}")

print("Test completed successfully!")


model.search(query, k=1)
returned_page = model.search(query, k=1)[0].base64