import os, re
from typing import List, Dict, Set, Optional
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="Resume-Vacancy Matcher API (v2 + bulk)")

# ---- Config ----
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" yoki "cuda"
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "2"))  # CPU-da haddan tashqari ko'p threadlarni cheklaymiz
torch.set_num_threads(TORCH_THREADS)

model: SentenceTransformer = None

@app.on_event("startup")
def startup():
    global model
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# ---------- Utilities ----------
def normalize_text_simple(t: str) -> str:
    t = re.sub(r"\s+", " ", t.strip().lower())
    t = re.sub(r"[^a-zа-яё0-9\+\#\./\- ]+", " ", t)
    return t

def split_into_chunks(text: str, max_chars: int = 600) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[\.\!\?\:]\s)", text)  # approx sentence split
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) <= max_chars:
            cur += p
        else:
            if cur: chunks.append(cur.strip())
            cur = p
    if cur: chunks.append(cur.strip())
    return chunks

# Aliases map -> normalized key (namuna ro'yxat, xohlaganingizcha kengaytiring)
SKILL_ALIASES: Dict[str, Set[str]] = {
    "python": {"python", "py"},
    "django": {"django"},
    "fastapi": {"fastapi"},
    "flask": {"flask"},
    "postgresql": {"postgresql", "postgres", "psql", "postgre", "постгрес"},
    "mysql": {"mysql"},
    "redis": {"redis"},
    "docker": {"docker", "докер"},
    "kubernetes": {"kubernetes", "k8s"},
    "aws": {"aws", "amazon web services", "ec2", "s3", "lambda"},
    "gcp": {"gcp", "google cloud"},
    "azure": {"azure"},
    "ci/cd": {"ci/cd", "cicd", "github actions", "gitlab ci", "jenkins", "ci", "cd"},
    "rest": {"rest", "restful"},
    "graphql": {"graphql"},
    "microservices": {"microservice", "microservices"},
    "rabbitmq": {"rabbitmq"},
    "kafka": {"kafka"},
    "celery": {"celery"},
    "pytorch": {"pytorch", "torch"},
    "tensorflow": {"tensorflow", "tf"},
    "nlp": {"nlp", "natural language processing"},
    "transformers": {"transformers", "hugging face", "sentence-transformers"},
    "linux": {"linux", "unix"},
    # Inter AI stack uchun yana qo'shishingiz mumkin: "laravel", "vue", "inertia", "telegram", "hh api", ...
}

def extract_skills(text: str) -> Set[str]:
    nt = normalize_text_simple(text)
    found = set()
    for key, aliases in SKILL_ALIASES.items():
        for a in aliases:
            if re.search(rf"\b{re.escape(a)}\b", nt):
                found.add(key); break
    return found

def encode_batch(texts: List[str], mode: str):
    """E5 oilasi bo'lsa prefix qo'shamiz; normalize_embeddings=True barqarorlik beradi."""
    if "e5" in MODEL_NAME.lower():
        if mode == "query":
            texts = [f"query: {t}" for t in texts]
        else:
            texts = [f"passage: {t}" for t in texts]
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=32, device=DEVICE)

def soft_alignment_score(resume_chunks: List[str], vacancy_chunks: List[str]) -> float:
    if not resume_chunks or not vacancy_chunks:
        return 0.0
    R = encode_batch(resume_chunks, mode="query")
    V = encode_batch(vacancy_chunks, mode="passage")
    sim = util.cos_sim(R, V)  # [R,V]
    # resume->vacancy va aksincha top-k o'rtacha
    r2v = sim.max(dim=1).values
    v2r = sim.max(dim=0).values
    kr = max(1, int(0.6 * len(r2v)))
    kv = max(1, int(0.6 * len(v2r)))
    top_r = torch.topk(r2v, kr).values.mean().item()
    top_v = torch.topk(v2r, kv).values.mean().item()
    return (top_r + top_v) / 2.0  # [-1,1]

def scale_cos_to_01(x: float) -> float:
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

# ---------- Single match (sizda bor) ----------
class MatchRequest(BaseModel):
    resume: str
    vacancy: str

class MatchResponse(BaseModel):
    score: float               # 0..100
    embedding_score: float     # 0..100
    skills_jaccard: float      # 0..100
    skill_matches: List[str]
    skill_missing: List[str]
    keyword_coverage: float    # 0..100
    chunks: Dict[str, int]
    model: str

@app.get("/")
def root():
    return {"ok": True, "message": "Resume-Vacancy Matcher API", "endpoints": ["/match2 (POST)", "/bulk-match (POST)", "/docs"]}

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}

@app.post("/match2", response_model=MatchResponse)
def match_v2(data: MatchRequest):
    with torch.no_grad():
        r_chunks = split_into_chunks(data.resume, 600)
        v_chunks = split_into_chunks(data.vacancy, 600)
        emb_sim = soft_alignment_score(r_chunks, v_chunks)         # [-1,1]
        emb01  = scale_cos_to_01(emb_sim)                          # [0,1]

        r_sk = extract_skills(data.resume)
        v_sk = extract_skills(data.vacancy)
        inter = sorted(r_sk & v_sk)
        missing = sorted(v_sk - r_sk)
        union = r_sk | v_sk
        jaccard = (len(inter) / len(union)) if union else 0.0
        coverage = (len(inter) / len(v_sk)) if v_sk else 0.0

        final01 = 0.70 * emb01 + 0.20 * jaccard + 0.10 * coverage

        return {
            "score": round(final01 * 100, 2),
            "embedding_score": round(emb01 * 100, 2),
            "skills_jaccard": round(jaccard * 100, 2),
            "skill_matches": inter,
            "skill_missing": missing,
            "keyword_coverage": round(coverage * 100, 2),
            "chunks": {"resume": len(r_chunks), "vacancy": len(v_chunks)},
            "model": MODEL_NAME,
        }

# ---------- BULK MATCH ----------
class BulkMatchRequest(BaseModel):
    resumes: List[str] = Field(..., description="List of resume texts (N)")
    vacancies: List[str] = Field(..., description="List of vacancy texts (M)")
    top_k: int = Field(5, ge=1, description="Top K vacancies for each resume")
    min_score: float = Field(0.0, ge=0.0, le=100.0, description="Filter final score threshold (0..100)")
    # performance tuning
    chunk_size: int = Field(600, ge=200, le=2000, description="Max chars per chunk")
    weight_embed: float = Field(0.70, ge=0.0, le=1.0)
    weight_jaccard: float = Field(0.20, ge=0.0, le=1.0)
    weight_cov: float = Field(0.10, ge=0.0, le=1.0)

class BulkTopItem(BaseModel):
    vacancy_index: int
    score: float
    embedding_score: float
    skills_jaccard: float
    keyword_coverage: float
    skill_matches: List[str]
    skill_missing: List[str]

class BulkMatchResponse(BaseModel):
    model: str
    device: str
    resumes: int
    vacancies: int
    sim_matrix_shape: List[int]
    top_k: int
    results: List[List[BulkTopItem]]  # per resume, list of top items

def _pairwise_scores(
    r_text: str, v_text: str, chunk_size: int,
    w_embed: float, w_jacc: float, w_cov: float
) -> Dict[str, float | List[str]]:
    """Yagona juftlik uchun komponent skorlari."""
    r_chunks = split_into_chunks(r_text, chunk_size)
    v_chunks = split_into_chunks(v_text, chunk_size)
    emb_sim = soft_alignment_score(r_chunks, v_chunks)         # [-1,1]
    emb01  = scale_cos_to_01(emb_sim)                          # [0,1]

    r_sk = extract_skills(r_text)
    v_sk = extract_skills(v_text)
    inter = sorted(r_sk & v_sk)
    missing = sorted(v_sk - r_sk)
    union = r_sk | v_sk
    jaccard = (len(inter) / len(union)) if union else 0.0
    coverage = (len(inter) / len(v_sk)) if v_sk else 0.0

    final01 = w_embed * emb01 + w_jacc * jaccard + w_cov * coverage

    return {
        "score": round(final01 * 100, 2),
        "embedding_score": round(emb01 * 100, 2),
        "skills_jaccard": round(jaccard * 100, 2),
        "keyword_coverage": round(coverage * 100, 2),
        "skill_matches": inter,
        "skill_missing": missing,
    }

@app.post("/bulk-match", response_model=BulkMatchResponse)
def bulk_match(req: BulkMatchRequest):
    """
    N ta resume va M ta vakansiyani bir urinishda tahlil qiladi:
    - embeddinglar batch tarzida olinadi (tez)
    - N x M cos-sim matritsa hisoblanadi
    - Har bir resume uchun top_k vacancylar qaytariladi
    - So'ngra har bir top juftlik uchun komponentlar (skills/coverage) alohida hisoblanadi
    """
    resumes = req.resumes
    vacancies = req.vacancies
    N, M = len(resumes), len(vacancies)
    assert N > 0 and M > 0, "Empty input"

    # 1) Tezlik uchun: matnlarni chunklamasdan global embedding (qo'pol, lekin tez).
    #    Keyin top_k juftliklar uchun batafsil komponent skorlari hisoblanadi.
    with torch.no_grad():
        R = encode_batch(resumes, mode="query")      # [N, D]
        V = encode_batch(vacancies, mode="passage")  # [M, D]
        S = util.cos_sim(R, V)                       # [N, M], [-1,1] interval

    # 2) Har bir resume uchun top_k indekslarni tanlaymiz
    top_k = min(req.top_k, M)
    results: List[List[BulkTopItem]] = []

    for i in range(N):
        sims = S[i]  # [M]
        top_vals, top_idx = torch.topk(sims, top_k)
        items: List[BulkTopItem] = []
        for val, j in zip(top_vals.tolist(), top_idx.tolist()):
            # Bu juftlik uchun to'liq komponent skorlari (chunk + skills) ni hisoblaymiz
            comp = _pairwise_scores(
                resumes[i], vacancies[j], req.chunk_size,
                req.weight_embed, req.weight_jaccard, req.weight_cov
            )
            if comp["score"] >= req.min_score:
                items.append(BulkTopItem(
                    vacancy_index=j,
                    score=comp["score"],
                    embedding_score=comp["embedding_score"],
                    skills_jaccard=comp["skills_jaccard"],
                    keyword_coverage=comp["keyword_coverage"],
                    skill_matches=comp["skill_matches"],
                    skill_missing=comp["skill_missing"],
                ))
        # score bo'yicha saralaymiz
        items.sort(key=lambda x: x.score, reverse=True)
        results.append(items)

    return {
        "model": MODEL_NAME,
        "device": DEVICE,
        "resumes": N,
        "vacancies": M,
        "sim_matrix_shape": [N, M],
        "top_k": top_k,
        "results": results,
    }
