"""Retrieval shared by the UI, evaluation runner, and Colab notebook."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent
NULLS = {"", "null", "none", "nan", "na", "n/a", "<na>"}
TASK = "Given an anime preference or plot description, retrieve anime whose synopsis and tags match the request."
TEXT_VERSION = 1


def settings():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def text(value):
    if value is None or pd.isna(value) or str(value).strip().casefold() in NULLS:
        return ""
    return str(value).strip()


def tags(value):
    return [t.strip() for t in text(value).split("|") if t.strip()]


def load_catalog(path):
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    required = {"mal_id", "title", "title_english", "tags", "synopsis", "type", "episodes", "rating", "status", "score", "aired_from"}
    if required - set(df.columns):
        raise ValueError(f"Missing data columns: {sorted(required - set(df.columns))}")
    for col in df.columns:
        df[col] = df[col].map(text)
    df["mal_id"] = pd.to_numeric(df["mal_id"], errors="raise").astype(int)
    df = df.drop_duplicates("mal_id").reset_index(drop=True)
    df["title_english"] = df["title_english"].where(df["title_english"] != "", df["title"])
    for col in ("episodes", "score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    derived_year = pd.to_datetime(df["aired_from"], errors="coerce").dt.year
    supplied_year = pd.to_numeric(df.get("year", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["year"] = derived_year.fillna(supplied_year)
    df["tag_list"] = df["tags"].map(tags)
    df["document"] = df.apply(document, axis=1)
    return df


def document(row):
    return (
        f"Title: {text(row.get('title'))}\n"
        f"English title: {text(row.get('title_english'))}\n"
        f"Tags: {', '.join(tags(row.get('tags')))}\n"
        f"Audience: {text(row.get('demographics'))}\n"
        f"Synopsis: {text(row.get('synopsis'))}"
    )


def catalog_fingerprint(df):
    payload = {"text_version": TEXT_VERSION, "ids": df.mal_id.tolist(), "documents": df.document.tolist()}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@dataclass
class Filters:
    types: list[str] = field(default_factory=list)
    required_tags: list[str] = field(default_factory=list)
    excluded_tags: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    max_episodes: int | None = None
    min_score: float | None = None
    statuses: list[str] = field(default_factory=list)
    studios: list[str] = field(default_factory=list)
    exclude_adult: bool = True


def eligible_mask(df, filters):
    mask = np.ones(len(df), dtype=bool)
    if filters.types:
        mask &= df.type.isin(filters.types).to_numpy()
    if filters.statuses:
        mask &= df.status.isin(filters.statuses).to_numpy()
    if filters.studios:
        selected = set(filters.studios)
        mask &= df.studios.map(lambda s: bool(selected.intersection(tags(s)))).to_numpy()
    if filters.required_tags:
        required = set(filters.required_tags)
        mask &= df.tag_list.map(lambda ts: required.issubset(ts)).to_numpy()
    if filters.excluded_tags:
        excluded = set(filters.excluded_tags)
        mask &= df.tag_list.map(lambda ts: not excluded.intersection(ts)).to_numpy()
    for col, bound, op in (("year", filters.year_min, "min"), ("year", filters.year_max, "max"),
                           ("episodes", filters.max_episodes, "max"), ("score", filters.min_score, "min")):
        if bound is not None:
            values = df[col]
            mask &= ((values >= bound) if op == "min" else (values <= bound)).fillna(False).to_numpy()
    if filters.exclude_adult:
        mask &= ~df.rating.str.startswith(("Rx", "R+")).to_numpy()
        mask &= df.tag_list.map(lambda ts: "Hentai" not in ts).to_numpy()
    return mask


class AnimeEngine:
    def __init__(self, data_path=None, artifact_dir=None):
        cfg = settings()
        self.df = load_catalog(data_path or ROOT / cfg["data"])
        self.artifact_dir = Path(artifact_dir or ROOT / cfg["artifacts"])
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = catalog_fingerprint(self.df)
        self.id_to_index = {int(mid): i for i, mid in enumerate(self.df.mal_id)}
        self.vectorizer = None
        self.tfidf = None
        self.embeddings = None
        self.embedding_meta = None

    def ensure_tfidf(self):
        if self.tfidf is not None:
            return
        path = self.artifact_dir / "tfidf.joblib"
        if path.exists():
            saved = joblib.load(path)
            if saved.get("fingerprint") == self.fingerprint:
                self.vectorizer, self.tfidf = saved["vectorizer"], saved["matrix"]
                return
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=80000,
                                          sublinear_tf=True, strip_accents="unicode", dtype=np.float32)
        self.tfidf = self.vectorizer.fit_transform(self.df.document)
        joblib.dump({"fingerprint": self.fingerprint, "vectorizer": self.vectorizer, "matrix": self.tfidf}, path)

    def load_embeddings(self):
        meta_path = self.artifact_dir / "embeddings.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError("ยังไม่มี embeddings: สร้างใน Colab หรือรัน build_embeddings.py ก่อน")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["fingerprint"] != self.fingerprint or meta["ids"] != self.df.mal_id.tolist():
            raise ValueError("Embedding ไม่ตรงกับ CSV ปัจจุบัน กรุณาสร้างใหม่ด้วยข้อมูลเดียวกัน")
        self.close()
        matrix = np.load(self.artifact_dir / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.df) or matrix.shape[1] != meta["dimension"]:
            matrix._mmap.close()
            raise ValueError("Embedding matrix has an invalid shape")
        self.embeddings, self.embedding_meta = matrix, meta
        return meta

    def close(self):
        if self.embeddings is not None:
            mapping = getattr(self.embeddings, "_mmap", None)
            if mapping is not None:
                mapping.close()
            self.embeddings = None
            self.embedding_meta = None

    def __del__(self):
        if hasattr(self, "embeddings"):
            self.close()

    def search(self, query="", liked_ids=None, filters=None, mode="tfidf", top_k=10,
               candidate_count=50, embedder=None, reranker=None, seed=42):
        liked_ids = list(liked_ids or [])
        query = query.strip()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if mode not in {"tfidf", "embedding", "random"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if not query and not liked_ids and mode != "random":
            raise ValueError("กรุณาใส่คำค้นหรือเลือกอนิเมะที่ชอบ")
        unknown = set(liked_ids) - self.id_to_index.keys()
        if unknown:
            raise ValueError(f"Unknown anime IDs: {sorted(unknown)}")
        mask = eligible_mask(self.df, filters or Filters())
        mask &= ~self.df.mal_id.isin(liked_ids).to_numpy()
        indices = np.flatnonzero(mask)
        if not len(indices):
            return []
        liked_indices = [self.id_to_index[mid] for mid in liked_ids]
        if mode == "random":
            scores = np.random.default_rng(seed).random(len(self.df))
        elif mode == "tfidf":
            self.ensure_tfidf()
            components = []
            if query:
                q = self.vectorizer.transform([query])
                if q.nnz:
                    components.append((self.tfidf @ q.T).toarray().ravel())
                elif not liked_ids:
                    return []
            if liked_indices:
                centroid = normalize(np.asarray(self.tfidf[liked_indices].mean(axis=0)))
                components.append(np.asarray(self.tfidf @ centroid.T).ravel())
            scores = np.mean(components, axis=0)
        else:
            if self.embeddings is None:
                self.load_embeddings()
            components = []
            if query:
                if embedder is None:
                    raise ValueError("Query embedding model is required")
                if embedder.model_name != self.embedding_meta["model"]:
                    raise ValueError("Query model and document embedding model must match")
                if embedder.max_length != self.embedding_meta["max_length"]:
                    raise ValueError("Query model max_length must match the embedding index")
                components.append(embedder.encode_queries([query])[0])
            if liked_indices:
                components.append(np.asarray(self.embeddings[liked_indices]).mean(axis=0))
            vector = np.mean(components, axis=0)
            vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
            scores = np.asarray(self.embeddings @ vector).ravel()
        if mode == "tfidf":
            indices = indices[scores[indices] > 1e-8]
        order = indices[np.argsort(-scores[indices], kind="stable")[:max(top_k, candidate_count)]]
        rerank_scores = {}
        if reranker is not None and len(order):
            rerank_query = query
            if liked_ids:
                liked_description = "\n".join(self.df.iloc[i].document for i in liked_indices)
                rerank_query += "\nFind anime similar to these favorites:\n" + liked_description
            values = reranker.score(rerank_query, self.df.iloc[order].document.tolist())
            if len(values) != len(order):
                raise ValueError("Reranker returned the wrong number of scores")
            rerank_scores = dict(zip(order.tolist(), map(float, values)))
            order = order[np.argsort(-np.asarray(values), kind="stable")]
        results = []
        liked_tags = set(t for i in liked_indices for t in self.df.iloc[i].tag_list)
        for idx in order[:top_k]:
            row = self.df.iloc[int(idx)]
            result = {k: (None if isinstance(v, (float, np.floating)) and np.isnan(v) else v)
                      for k, v in row.to_dict().items() if k not in {"document", "tag_list"}}
            result["mal_id"] = int(result["mal_id"])
            result["similarity"] = float(scores[idx])
            result["rerank_score"] = rerank_scores.get(int(idx))
            result["url"] = f"https://myanimelist.net/anime/{result['mal_id']}"
            overlap = [t for t in row.tag_list if t in liked_tags]
            matched = [t for t in row.tag_list if t.casefold() in query.casefold()]
            if overlap:
                reason = "มีแท็กร่วมกับเรื่องที่คุณชอบ: " + ", ".join(overlap[:5])
            elif matched:
                reason = "มีแท็กตรงกับคำค้น: " + ", ".join(matched[:5])
            else:
                reason = "เลือกจากความคล้ายของเรื่องย่อและแท็กกับคำค้น"
            result["reason"] = reason
            results.append(result)
        return results

    def lookup_titles(self, query, limit=30):
        needle = query.strip().casefold()
        if not needle:
            return self.df.head(limit)
        left, right = self.df.title.str.casefold(), self.df.title_english.str.casefold()
        exact = self.df[(left == needle) | (right == needle)]
        partial = self.df[left.str.contains(needle, regex=False) | right.str.contains(needle, regex=False)]
        return pd.concat([exact, partial]).drop_duplicates("mal_id").head(limit)


def metrics_at_k(predicted_ids, relevance, k=10):
    """Unjudged documents count as 0; denominator for precision is always k."""
    if k < 1:
        raise ValueError("k must be positive")
    grades = [float(relevance.get(str(mid), relevance.get(mid, 0))) for mid in predicted_ids[:k]]
    precision = sum(g > 0 for g in grades) / k
    mrr = next((1 / (i + 1) for i, g in enumerate(grades) if g > 0), 0.0)
    dcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(grades))
    ideal = sorted(map(float, relevance.values()), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    relevant = sum(float(g) > 0 for g in relevance.values())
    recall = sum(g > 0 for g in grades) / relevant if relevant else 0.0
    return {f"precision@{k}": precision, f"mrr@{k}": mrr,
            f"ndcg@{k}": float(dcg / idcg) if idcg else 0.0, f"recall@{k}": recall}
