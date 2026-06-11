#!/usr/bin/env python3
"""
AI Hub 「상황에 따른 판례」 JSON → processed_data CSV 전처리 스크립트.

원본 JSON은 AI Hub 라이선스로 인해 저장소에 포함되지 않습니다.
다운로드 후 --json-dir 에 라벨링 JSON 폴더 경로를 지정하세요.

사용 예:
  python preprocess/make_processed_data.py --json-dir /path/to/라벨링데이터
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# 저장소 루트 기준 출력 경로
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "processed_data"

EXPECTED_CLASSES = [
    "가사",
    "개인정보/ICT",
    "근로자",
    "금융조세",
    "기업",
    "민사",
    "특허/저작권",
    "행정",
    "형사A(생활형)",
    "형사B(일반형)",
]


def clean_text(text: str) -> str:
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_json_file(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as f:
            d = json.load(f)
        return {
            "jdgmn": d["jdgmn"],
            "summ_pass": d["Summary"][0]["summ_pass"],
            "class_name": d["Class_info"]["class_name"],
        }
    except (KeyError, IndexError, json.JSONDecodeError, TypeError):
        return None


def load_json_records(json_dir: Path) -> pd.DataFrame:
    records = []
    skipped = 0
    for root, _, files in os.walk(json_dir):
        for name in files:
            if not name.endswith(".json"):
                continue
            row = parse_json_file(Path(root) / name)
            if row is None:
                skipped += 1
                continue
            records.append(row)
    if not records:
        raise FileNotFoundError(f"JSON 레코드를 찾지 못했습니다: {json_dir}")
    print(f"JSON 파싱: {len(records)}건 성공, {skipped}건 스킵(필드 누락·파싱 오류)")
    return pd.DataFrame(records)


def drop_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """결측 제거 및 클래스별 제거율 통계 반환."""
    before = len(df)
    valid = (
        df["jdgmn"].notna()
        & df["summ_pass"].notna()
        & df["class_name"].notna()
        & (df["jdgmn"].astype(str).str.strip() != "")
        & (df["summ_pass"].astype(str).str.strip() != "")
        & (df["class_name"].astype(str).str.strip() != "")
    )
    removed = df[~valid]
    kept = df[valid].reset_index(drop=True)
    after = len(kept)
    print(f"결측 제거: {before} → {after} ({before - after}건 제거, {100*(before-after)/before:.1f}%)")

    before_counts = df["class_name"].value_counts()
    removed_counts = removed["class_name"].value_counts()
    stats = []
    for cls in sorted(before_counts.index):
        b = int(before_counts.get(cls, 0))
        r = int(removed_counts.get(cls, 0))
        rate = 100.0 * r / b if b else 0.0
        stats.append({"class_name": cls, "before": b, "removed": r, "kept": b - r, "removal_rate_pct": rate})
    removal_stats = pd.DataFrame(stats)
    print("\n=== 클래스별 결측 제거율 ===")
    for row in stats:
        print(
            f"  {row['class_name']}: {row['before']} → {row['kept']} "
            f"(제거 {row['removed']}, {row['removal_rate_pct']:.1f}%)"
        )
    return kept, removal_stats


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["jdgmn"] = df["jdgmn"].astype(str).map(clean_text)
    df["summ_pass"] = df["summ_pass"].astype(str).map(clean_text)
    return df


def build_label_mapping(df: pd.DataFrame) -> pd.DataFrame:
    classes = sorted(df["class_name"].unique())
    unknown = set(classes) - set(EXPECTED_CLASSES)
    if unknown:
        print(f"경고: 예상 외 클래스 {unknown}")
    mapping = pd.DataFrame(
        {"class_name": EXPECTED_CLASSES, "label": list(range(len(EXPECTED_CLASSES)))}
    )
    missing = set(EXPECTED_CLASSES) - set(classes)
    if missing:
        print(f"경고: 데이터에 없는 클래스 {missing}")
    return mapping


def print_class_distribution(df: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    counts = df["class_name"].value_counts().sort_index()
    for name, cnt in counts.items():
        pct = 100.0 * cnt / len(df)
        print(f"  {name}: {cnt} ({pct:.2f}%)")


def stratified_split(
    df: pd.DataFrame, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=random_state,
    )
    return train_df, val_df, test_df


def main() -> int:
    parser = argparse.ArgumentParser(description="판례 JSON → processed_data CSV")
    parser.add_argument(
        "--json-dir",
        type=Path,
        required=True,
        help="AI Hub 라벨링 JSON 폴더 경로",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"출력 폴더 (기본: {DEFAULT_OUT})",
    )
    parser.add_argument("--seed", type=int, default=42, help="분할 random_state")
    args = parser.parse_args()

    if not args.json_dir.is_dir():
        print(f"오류: json-dir 없음 — {args.json_dir}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_json_records(args.json_dir)
    print_class_distribution(df, "JSON 추출 직후 클래스 분포")

    df, removal_stats = drop_missing(df)
    df = apply_cleaning(df)

    label_mapping = build_label_mapping(df)
    class_to_label = dict(zip(label_mapping["class_name"], label_mapping["label"]))
    df["label"] = df["class_name"].map(class_to_label)
    if df["label"].isna().any():
        bad = df[df["label"].isna()]["class_name"].unique()
        raise ValueError(f"label 매핑 실패 클래스: {bad}")

    print_class_distribution(df, "결측 제거·정제 후 클래스 분포")

    train_df, val_df, test_df = stratified_split(df, random_state=args.seed)
    print(
        f"\n분할 결과 (stratify, seed={args.seed}): "
        f"train {len(train_df)} / val {len(val_df)} / test {len(test_df)}"
    )

    removal_stats.to_csv(args.out_dir / "removal_stats_by_class.csv", index=False)
    label_mapping.to_csv(args.out_dir / "label_mapping.csv", index=False)
    train_df.to_csv(args.out_dir / "train.csv", index=False)
    val_df.to_csv(args.out_dir / "val.csv", index=False)
    test_df.to_csv(args.out_dir / "test.csv", index=False)

    print(f"\n저장 완료: {args.out_dir}")
    print("  label_mapping.csv, removal_stats_by_class.csv, train.csv, val.csv, test.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
