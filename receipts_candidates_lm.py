import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import hnswlib
from sentence_transformers import SentenceTransformer


MODEL_HF_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def set_offline_mode():
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode", total=(len(texts) + batch_size - 1) // batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        out.append(emb.astype(np.float32, copy=False))
    if len(out) == 0:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return np.vstack(out)


def download_model(model_dir: Path):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_HF_NAME, device="cpu")
    model.save(str(model_dir))
    print(f"Saved model to: {model_dir}")


def build_user_embeddings_from_receipts(
    receipts: pd.DataFrame,
    model: SentenceTransformer,
    reference_date: str | None,
    window_days: int | None,
    tau_days: float | None,
    batch_size: int,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    df = receipts.copy()

    need_cols = {"user_id", "date_operated", "category_name", "items_count", "items_cost"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"receipts is missing columns: {sorted(missing)}")

    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df = df[df["user_id"].notna()].copy()
    df["user_id"] = df["user_id"].astype(np.int64)

    df["category_name"] = df["category_name"].astype("string").fillna("").str.strip().str.lower()
    df = df[df["category_name"] != ""].copy()

    df["items_cost"] = pd.to_numeric(df["items_cost"], errors="coerce").fillna(0.0).astype(np.float32)
    df["items_cost"] = np.clip(df["items_cost"].to_numpy(), 0.0, None).astype(np.float32)

    df["items_count"] = pd.to_numeric(df["items_count"], errors="coerce").fillna(0.0).astype(np.float32)
    df["items_count"] = np.clip(df["items_count"].to_numpy(), 0.0, None).astype(np.float32)

    df["date_operated"] = pd.to_datetime(df["date_operated"], errors="coerce")
    df = df[df["date_operated"].notna()].copy()

    ref_ts = None
    if reference_date is not None:
        ref_ts = pd.Timestamp(reference_date)
        df = df[df["date_operated"] < ref_ts].copy()
        if window_days is not None:
            df = df[df["date_operated"] >= (ref_ts - pd.Timedelta(days=int(window_days)))].copy()

    if df.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    user_ids, user_codes = pd.factorize(df["user_id"], sort=True)
    cat_ids, cat_codes = pd.factorize(df["category_name"], sort=True)

    cat_texts = [f"покупка: {c}" for c in cat_ids.tolist()]
    cat_emb = encode_texts(model, cat_texts, batch_size=batch_size)

    w = np.log1p(df["items_cost"].to_numpy(dtype=np.float32))

    if ref_ts is not None and tau_days is not None and tau_days > 0:
        days_ago = (ref_ts - df["date_operated"]).dt.days.to_numpy(dtype=np.float32)
        days_ago = np.clip(days_ago, 0.0, None)
        w = w * np.exp(-days_ago / float(tau_days)).astype(np.float32)

    w = np.clip(w, 0.0, None).astype(np.float32)

    n_users = int(user_ids.shape[0])
    dim = int(cat_emb.shape[1])

    sums = np.zeros((n_users, dim), dtype=np.float32)
    wsum = np.zeros((n_users,), dtype=np.float32)

    uc = user_codes.astype(np.int64, copy=False)
    cc = cat_codes.astype(np.int64, copy=False)

    for s in tqdm(range(0, df.shape[0], chunk_rows), desc="Aggregate users"):
        e = min(s + chunk_rows, df.shape[0])
        u = uc[s:e]
        c = cc[s:e]
        ww = w[s:e]
        vec = cat_emb[c]
        vec = vec * ww.reshape(-1, 1)
        np.add.at(sums, u, vec)
        np.add.at(wsum, u, ww)

    mask = wsum > 0
    sums[mask] = sums[mask] / wsum[mask].reshape(-1, 1)

    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12).astype(np.float32)
    sums = (sums / norms).astype(np.float32)

    return user_ids.astype(np.int64), sums


def build_hnsw_index(user_emb: np.ndarray, out_path: Path, m: int = 48, ef_construction: int = 200, ef_search: int = 200):
    if user_emb.dtype != np.float32:
        user_emb = user_emb.astype(np.float32)

    dim = int(user_emb.shape[1])
    n = int(user_emb.shape[0])

    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=n, ef_construction=int(ef_construction), M=int(m))

    labels = np.arange(n, dtype=np.int32)
    for s in tqdm(range(0, n, 20000), desc="Add to HNSW"):
        e = min(s + 20000, n)
        idx.add_items(user_emb[s:e], labels[s:e])

    idx.set_ef(int(ef_search))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx.save_index(str(out_path))
    print(f"Saved HNSW index to: {out_path}")


def query_candidates(
    offers: pd.DataFrame,
    model: SentenceTransformer,
    user_ids: np.ndarray,
    index_path: Path,
    offer_text_col: str,
    top_k: int,
    batch_size: int,
) -> pd.DataFrame:
    if offer_text_col not in offers.columns:
        raise ValueError(f"offers has no column: {offer_text_col}")

    if "offer_id" not in offers.columns:
        raise ValueError("offers has no column: offer_id")

    set_offline_mode()

    idx = hnswlib.Index(space="cosine", dim=model.get_sentence_embedding_dimension())
    idx.load_index(str(index_path))
    idx.set_ef(max(200, int(top_k)))

    texts = offers[offer_text_col].astype("string").fillna("").tolist()
    offer_ids = offers["offer_id"].to_numpy()

    emb = encode_texts(model, [f"оффер: {t}" for t in texts], batch_size=batch_size)

    rows = []
    for i in tqdm(range(emb.shape[0]), desc="Query offers"):
        labels, distances = idx.knn_query(emb[i], k=int(top_k))
        labels = labels.reshape(-1)
        distances = distances.reshape(-1)
        sim = (1.0 - distances).astype(np.float32)
        uids = user_ids[labels.astype(np.int64)]
        oid = offer_ids[i]
        for uid, sc in zip(uids.tolist(), sim.tolist()):
            rows.append((oid, int(uid), float(sc)))

    return pd.DataFrame(rows, columns=["offer_id", "user_id", "score_receipts_semantic"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_dl = sub.add_parser("download-model")
    ap_dl.add_argument("--model_dir", type=str, required=True)

    ap_build = sub.add_parser("build-index")
    ap_build.add_argument("--receipts_path", type=str, required=True)
    ap_build.add_argument("--model_dir", type=str, required=True)
    ap_build.add_argument("--out_dir", type=str, required=True)
    ap_build.add_argument("--reference_date", type=str, default=None)
    ap_build.add_argument("--window_days", type=int, default=180)
    ap_build.add_argument("--tau_days", type=float, default=90.0)
    ap_build.add_argument("--batch_size", type=int, default=64)
    ap_build.add_argument("--chunk_rows", type=int, default=200000)
    ap_build.add_argument("--hnsw_m", type=int, default=48)
    ap_build.add_argument("--ef_construction", type=int, default=200)
    ap_build.add_argument("--ef_search", type=int, default=200)

    ap_q = sub.add_parser("query")
    ap_q.add_argument("--offers_path", type=str, required=True)
    ap_q.add_argument("--model_dir", type=str, required=True)
    ap_q.add_argument("--index_dir", type=str, required=True)
    ap_q.add_argument("--offer_text_col", type=str, default="offer_text")
    ap_q.add_argument("--top_k", type=int, default=20000)
    ap_q.add_argument("--batch_size", type=int, default=64)
    ap_q.add_argument("--out_path", type=str, required=True)

    args = ap.parse_args()

    if args.cmd == "download-model":
        download_model(Path(args.model_dir))
        return

    if args.cmd == "build-index":
        set_offline_mode()
        model = SentenceTransformer(str(Path(args.model_dir)), device="cpu")
        receipts = read_table(Path(args.receipts_path))
        user_ids, user_emb = build_user_embeddings_from_receipts(
            receipts=receipts,
            model=model,
            reference_date=args.reference_date,
            window_days=args.window_days,
            tau_days=args.tau_days,
            batch_size=args.batch_size,
            chunk_rows=args.chunk_rows,
        )
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "user_ids.npy", user_ids)
        np.save(out_dir / "user_emb.npy", user_emb)

        build_hnsw_index(
            user_emb=user_emb,
            out_path=out_dir / "hnsw_cosine.bin",
            m=args.hnsw_m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )
        return

    if args.cmd == "query":
        set_offline_mode()
        model = SentenceTransformer(str(Path(args.model_dir)), device="cpu")
        offers = read_table(Path(args.offers_path))
        index_dir = Path(args.index_dir)
        user_ids = np.load(index_dir / "user_ids.npy")
        res = query_candidates(
            offers=offers,
            model=model,
            user_ids=user_ids,
            index_path=index_dir / "hnsw_cosine.bin",
            offer_text_col=args.offer_text_col,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() in {".parquet", ".pq"}:
            res.to_parquet(out_path, index=False)
        else:
            res.to_csv(out_path, index=False)
        print(f"Saved candidates to: {out_path}")
        return


if __name__ == "__main__":
    main()