import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
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
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def save_table(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return (x / n).astype(np.float32, copy=False)


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encode", total=(len(texts) + batch_size - 1) // batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        out.append(emb.astype(np.float32, copy=False))
    if not out:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return np.vstack(out)


def download_model(model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_HF_NAME, device="cpu")
    model.save(str(model_dir))
    print(f"Saved model to: {model_dir}")


def build_category_cache(receipts_path: Path, model_dir: Path, out_dir: Path, batch_size: int):
    set_offline_mode()
    model = SentenceTransformer(str(model_dir), device="cpu")

    receipts = read_table(receipts_path)
    if "category_name" not in receipts.columns:
        raise ValueError("receipts has no column: category_name")

    cats = (
        receipts["category_name"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )
    cats = cats[cats != ""]
    cats = cats.drop_duplicates().sort_values().tolist()

    texts = [f"категория покупки: {c}" for c in cats]
    emb = encode_texts(model, texts, batch_size=batch_size)
    emb = normalize_rows(emb)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cat_emb.npy", emb)
    pd.DataFrame({"category_name": cats}).to_parquet(out_dir / "cat_names.parquet", index=False)
    print(f"Saved category cache to: {out_dir}")


def load_category_cache(cache_dir: Path) -> tuple[pd.Series, np.ndarray]:
    names_path = cache_dir / "cat_names.parquet"
    emb_path = cache_dir / "cat_emb.npy"
    if not names_path.exists() or not emb_path.exists():
        raise FileNotFoundError("Category cache not found. Run build-cat-cache first.")
    names = pd.read_parquet(names_path)["category_name"].astype("string")
    emb = np.load(emb_path)
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    return names, emb


def prepare_receipts_aggregates(
    receipts: pd.DataFrame,
    reference_date: str,
    window_10: int,
    window_45: int,
    user_col: str,
    date_col: str,
    cat_col: str,
    count_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    df = receipts.copy()

    for col in [user_col, date_col, cat_col]:
        if col not in df.columns:
            raise ValueError(f"receipts has no column: {col}")

    df[user_col] = pd.to_numeric(df[user_col], errors="coerce")
    df = df[df[user_col].notna()].copy()
    df[user_col] = df[user_col].astype(np.int64)

    df[cat_col] = df[cat_col].astype("string").fillna("").str.strip().str.lower()
    df = df[df[cat_col] != ""].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy()

    ref = pd.Timestamp(reference_date)
    df = df[df[date_col] < ref].copy()

    df45 = df[df[date_col] >= (ref - pd.Timedelta(days=window_45))].copy()
    df10 = df[df[date_col] >= (ref - pd.Timedelta(days=window_10))].copy()

    if count_col is not None and count_col in df.columns:
        v45 = pd.to_numeric(df45[count_col], errors="coerce").fillna(0.0).astype(np.float32)
        v10 = pd.to_numeric(df10[count_col], errors="coerce").fillna(0.0).astype(np.float32)
    else:
        v45 = pd.Series(1.0, index=df45.index, dtype=np.float32)
        v10 = pd.Series(1.0, index=df10.index, dtype=np.float32)

    df45["_cnt"] = np.clip(v45.to_numpy(dtype=np.float32), 0.0, None)
    df10["_cnt"] = np.clip(v10.to_numpy(dtype=np.float32), 0.0, None)

    agg45 = df45.groupby([user_col, cat_col], sort=False)["_cnt"].sum().rename("cnt_45").reset_index()
    agg10 = df10.groupby([user_col, cat_col], sort=False)["_cnt"].sum().rename("cnt_10").reset_index()

    agg = agg45.merge(agg10, on=[user_col, cat_col], how="left")
    agg["cnt_10"] = agg["cnt_10"].fillna(0.0).astype(np.float32)
    agg["cnt_45"] = agg["cnt_45"].fillna(0.0).astype(np.float32)

    active_days_45 = (
        df45.assign(_d=df45[date_col].dt.date)
        .groupby(user_col)["_d"]
        .nunique()
        .astype(np.int32)
    )

    df_c = df45.groupby(cat_col)[user_col].nunique().astype(np.int32)
    n_users = int(df45[user_col].nunique())

    return agg, df_c, active_days_45, pd.Series(df45[cat_col].unique(), dtype="string"), n_users


def pick_top_categories_for_offer(
    model: SentenceTransformer,
    offer_text: str,
    cat_names: pd.Series,
    cat_emb: np.ndarray,
    top_k: int,
    batch_size: int,
) -> pd.DataFrame:
    offer_emb = encode_texts(model, [f"оффер: {offer_text}"], batch_size=batch_size)
    offer_emb = normalize_rows(offer_emb)
    sim = (cat_emb @ offer_emb[0]).astype(np.float32)
    if top_k >= sim.shape[0]:
        idx = np.argsort(-sim)
    else:
        idx = np.argpartition(-sim, top_k - 1)[:top_k]
        idx = idx[np.argsort(-sim[idx], kind="mergesort")]
    return pd.DataFrame({"category_name": cat_names.iloc[idx].to_numpy(), "sim": sim[idx]})


def score_users_for_offer(
    agg_user_cat: pd.DataFrame,
    df_c: pd.Series,
    active_days_45: pd.Series,
    top_cats: pd.DataFrame,
    n_users: int,
    user_col: str,
    cat_col: str,
    alpha: float,
    beta: float,
    p: float,
    top_users: int,
    banned_users_path: Path | None,
) -> pd.DataFrame:
    w = top_cats.copy()
    w["sim"] = np.clip(w["sim"].astype(np.float32), 0.0, 1.0)
    w["q"] = np.power(w["sim"], float(p)).astype(np.float32)

    dfc = df_c.reindex(w["category_name"]).fillna(0).astype(np.int32).to_numpy()
    idf = np.log1p(float(n_users) / (1.0 + dfc.astype(np.float32))).astype(np.float32)

    w["idf"] = idf
    w["weight"] = (w["q"] * w["idf"]).astype(np.float32)

    sub = agg_user_cat.merge(w[["category_name", "weight"]], left_on=cat_col, right_on="category_name", how="inner")
    if sub.shape[0] == 0:
        return pd.DataFrame({user_col: np.array([], dtype=np.int64), "score_receipts": np.array([], dtype=np.float32)})

    sub["tf10"] = np.log1p(sub["cnt_10"].astype(np.float32)).astype(np.float32)
    sub["tf45"] = np.log1p(sub["cnt_45"].astype(np.float32)).astype(np.float32)
    sub["term"] = (sub["weight"].astype(np.float32) * (float(alpha) * sub["tf10"] + float(beta) * sub["tf45"])).astype(np.float32)

    score_s = sub.groupby(user_col, sort=False)["term"].sum().astype(np.float32)
    uids = score_s.index.to_numpy(dtype=np.int64)
    vals = score_s.to_numpy(dtype=np.float32)

    ad = active_days_45.reindex(score_s.index).fillna(0).to_numpy(dtype=np.float32)
    vals = (vals * (1.0 / np.sqrt(1.0 + ad))).astype(np.float32)

    res = pd.DataFrame({user_col: uids, "score_receipts": vals})

    if banned_users_path is not None and banned_users_path.exists():
        banned = np.load(banned_users_path).astype(np.int64)
        res = res[~np.isin(res[user_col].to_numpy(dtype=np.int64), banned)].copy()

    res = res.sort_values("score_receipts", ascending=False)
    if top_users is not None and top_users > 0:
        res = res.head(int(top_users)).copy()

    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_dl = sub.add_parser("download-model")
    ap_dl.add_argument("--model_dir", type=str, required=True)

    ap_cc = sub.add_parser("build-cat-cache")
    ap_cc.add_argument("--receipts_path", type=str, required=True)
    ap_cc.add_argument("--model_dir", type=str, required=True)
    ap_cc.add_argument("--cache_dir", type=str, required=True)
    ap_cc.add_argument("--batch_size", type=int, default=64)

    ap_run = sub.add_parser("make-candidates")
    ap_run.add_argument("--receipts_path", type=str, required=True)
    ap_run.add_argument("--offers_path", type=str, required=True)
    ap_run.add_argument("--offer_id", type=int, required=True)
    ap_run.add_argument("--offer_text_col", type=str, default="offer_text")
    ap_run.add_argument("--model_dir", type=str, required=True)
    ap_run.add_argument("--cache_dir", type=str, required=True)
    ap_run.add_argument("--reference_date", type=str, required=True)
    ap_run.add_argument("--topk_categories", type=int, default=30)
    ap_run.add_argument("--alpha", type=float, default=1.0)
    ap_run.add_argument("--beta", type=float, default=0.5)
    ap_run.add_argument("--p", type=float, default=2.0)
    ap_run.add_argument("--top_users", type=int, default=20000)
    ap_run.add_argument("--batch_size", type=int, default=64)
    ap_run.add_argument("--user_col", type=str, default="user_id")
    ap_run.add_argument("--date_col", type=str, default="date_operated")
    ap_run.add_argument("--cat_col", type=str, default="category_name")
    ap_run.add_argument("--count_col", type=str, default="items_count")
    ap_run.add_argument("--banned_users_npy", type=str, default=None)
    ap_run.add_argument("--out_path", type=str, required=True)

    args = ap.parse_args()

    if args.cmd == "download-model":
        download_model(Path(args.model_dir))
        return

    if args.cmd == "build-cat-cache":
        build_category_cache(
            receipts_path=Path(args.receipts_path),
            model_dir=Path(args.model_dir),
            out_dir=Path(args.cache_dir),
            batch_size=int(args.batch_size),
        )
        return

    if args.cmd == "make-candidates":
        set_offline_mode()
        model = SentenceTransformer(str(Path(args.model_dir)), device="cpu")

        cat_names, cat_emb = load_category_cache(Path(args.cache_dir))

        offers = read_table(Path(args.offers_path))
        if "offer_id" not in offers.columns:
            raise ValueError("offers has no column: offer_id")
        if args.offer_text_col not in offers.columns:
            raise ValueError(f"offers has no column: {args.offer_text_col}")

        row = offers.loc[offers["offer_id"] == int(args.offer_id)]
        if row.shape[0] == 0:
            raise ValueError(f"offer_id={args.offer_id} not found in offers")
        offer_text = row.iloc[0][args.offer_text_col]
        offer_text = "" if pd.isna(offer_text) else str(offer_text)

        receipts = read_table(Path(args.receipts_path))
        agg, df_c, active_days_45, _, n_users = prepare_receipts_aggregates(
            receipts=receipts,
            reference_date=args.reference_date,
            window_10=10,
            window_45=45,
            user_col=args.user_col,
            date_col=args.date_col,
            cat_col=args.cat_col,
            count_col=args.count_col if args.count_col != "None" else None,
        )

        top_cats = pick_top_categories_for_offer(
            model=model,
            offer_text=offer_text,
            cat_names=cat_names,
            cat_emb=cat_emb,
            top_k=int(args.topk_categories),
            batch_size=int(args.batch_size),
        )

        banned_path = Path(args.banned_users_npy) if args.banned_users_npy else None

        res = score_users_for_offer(
            agg_user_cat=agg,
            df_c=df_c,
            active_days_45=active_days_45,
            top_cats=top_cats,
            n_users=n_users,
            user_col=args.user_col,
            cat_col=args.cat_col,
            alpha=float(args.alpha),
            beta=float(args.beta),
            p=float(args.p),
            top_users=int(args.top_users),
            banned_users_path=banned_path,
        )

        res.insert(0, "offer_id", int(args.offer_id))
        save_table(res, Path(args.out_path))
        print(f"Saved candidates to: {args.out_path}")
        return


if __name__ == "__main__":
    main()