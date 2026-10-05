from pathlib import Path
import sys
from anime_engine import AnimeEngine, Filters, settings
from model_status import model_ready

ROOT = Path(__file__).resolve().parent
cfg = settings()
assert sys.version_info[:2] == (3, 11), "Use Python 3.11"
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
for role in ("embedding", "reranker", "explanation"):
    model = cfg[role + "_model"]
    assert model_ready(ROOT, model), "Incomplete model: " + model
    print("Ready:", model)
engine = AnimeEngine()
engine.load_embeddings()
assert len(engine.search(liked_ids=[1], mode="embedding", filters=Filters())) > 0
assert len(engine.search("pirate", mode="tfidf")) > 0
engine.close()
print("Offline model files, dataset, indexes and basic search verified.")
