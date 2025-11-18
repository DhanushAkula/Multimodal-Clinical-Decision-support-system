import sys
sys.path.insert(0, '/home/arutla.a/medpix-project')
from data_loader import MedPixDataLoader
from sentence_transformers import SentenceTransformer
import faiss
import json
import os

data = MedPixDataLoader('/home/arutla.a/medpix-project/MedPix-2.0/MedPix-2-0')
train = data.get_split('train')
print(f"Building RAG from {len(train)} cases")

corpus = []
metadata = []
for case in train:
    descs = data.get_descriptions(case['U_id'])
    if not descs:
        continue
    desc = descs[0]
    text = f"{desc['Description'].get('Age','')} year old {desc['Description'].get('Sex','')} patient.\n{case['Case'].get('History','')}"
    corpus.append(text)
    metadata.append({'U_id': case['U_id']})

print(f"Encoding {len(corpus)} texts...")
model = SentenceTransformer('Linq-AI-Research/Linq-Embed-Mistral')
embeddings = model.encode(corpus, show_progress_bar=True, batch_size=32)

index = faiss.IndexFlatIP(embeddings.shape[1])
faiss.normalize_L2(embeddings)
index.add(embeddings.astype('float32'))

os.makedirs('/scratch/arutla.a/medpix-outputs/rag', exist_ok=True)
faiss.write_index(index, '/scratch/arutla.a/medpix-outputs/rag/rag_full.index')
with open('/scratch/arutla.a/medpix-outputs/rag/rag_metadata.json', 'w') as f:
    json.dump({'corpus': corpus, 'metadata': metadata}, f)
print(f"✓ Saved {index.ntotal} vectors")
