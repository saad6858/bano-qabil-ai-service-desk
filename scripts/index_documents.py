#!/usr/bin/env python3
"""CLI wrapper for guide-compliant document indexing."""

from __future__ import annotations

import argparse
from pathlib import Path

from bq_service_desk.config import get_settings
from bq_service_desk.rag.ingestion import index_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()

    count = index_documents(
        source_dir=Path(args.source_dir),
        index_name=args.index,
        namespace=args.namespace,
        settings=get_settings(),
    )
    print(f"Indexed {count} chunks.")


if __name__ == "__main__":
    main()
