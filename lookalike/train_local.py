from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from ml.pipeline import LookalikePipeline
from ml.data_prep import DataBundle


def load_bundle(data_dir: Path) -> DataBundle:
    return DataBundle(
        people=pd.read_csv(data_dir / "prod_clients.csv"),
        segments=pd.read_csv(data_dir / "prizm_segments.csv"),
        transaction=pd.read_csv(data_dir / "prod_financial_transaction.csv"),
        offer=pd.read_csv(data_dir / "t_offer.csv"),
        merchant=pd.read_csv(data_dir / "t_merchant.csv"),
        financial_account=pd.read_csv(data_dir / "financial_account.csv"),
        offer_seens=pd.read_csv(data_dir / "offer_seens.csv"),
        offer_activation=pd.read_csv(data_dir / "offer_activation.csv"),
        offer_reward=pd.read_csv(data_dir / "offer_reward.csv"),
        receipts=pd.read_csv(data_dir / "receipts.csv"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default='data/v1')
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/lookalike_v2")
    args = parser.parse_args()
    bundle = load_bundle(Path(args.data_dir))
    pipeline = LookalikePipeline(params_path=args.params, artifacts_dir=args.artifacts_dir)
    metrics = pipeline.train(bundle)
    print(metrics)


if __name__ == "__main__":
    main()
