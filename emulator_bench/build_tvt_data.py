import argparse

from build_tvt_features import build_features


def main():
    parser = argparse.ArgumentParser(
        description="Build ProSmith TVT feature artifacts from a CSV."
    )
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_pkl", required=True, type=str)
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)
    parser.add_argument("--target_col", required=True, type=str)
    parser.add_argument("--prot_batch_size", default=8, type=int)
    parser.add_argument("--mol_batch_size", default=16, type=int)
    parser.add_argument("--cache_dir", default="emulator_bench/.cache_embeddings", type=str)
    parser.add_argument("--no_cache_read", action="store_true")
    parser.add_argument("--no_cache_write", action="store_true")

    args = parser.parse_args()

    build_features(
        input_csv=args.input_csv,
        output_pkl=args.output_pkl,
        target_col=args.target_col,
        sequence_col=args.sequence_col,
        smiles_col=args.smiles_col,
        prot_batch_size=args.prot_batch_size,
        mol_batch_size=args.mol_batch_size,
        cache_dir=args.cache_dir,
        cache_read=not args.no_cache_read,
        cache_write=not args.no_cache_write,
    )


if __name__ == "__main__":
    main()
