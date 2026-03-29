"""
Verify padding correctness for dataset CSVs.

Logic:
- Different cancer datasets (BLCA, BRCA, GBMLGG, LUAD, UCEC) have different genomic columns.
- For joint training, columns must be unified: missing columns are added and filled with 0.
- The padded CSV should have the UNION of all columns from all datasets.
- For each padded CSV, original non-zero values must be preserved, and new columns must be all zeros.

This script compares:
  dataset_csv_padding/<name>_padded.csv  vs  dataset_csv_padding/<name>.csv (or .csv.zip)

Usage:
  python utils/verify_padding.py
"""

import os
import sys

import numpy as np
import pandas as pd


def load_csv(filepath):
    """Load CSV file, handling .zip compression."""
    if os.path.exists(filepath + '.zip') and not os.path.exists(filepath):
        return pd.read_csv(filepath + '.zip', compression='zip', header=0, index_col=0, sep=',', low_memory=False)
    elif os.path.exists(filepath):
        return pd.read_csv(filepath, header=0, index_col=0, sep=',', low_memory=False)
    else:
        print(f"  [WARNING] File not found: {filepath}")
        return None


def verify_padding_dir(base_dir):
    """Verify padding correctness for CSVs in a directory."""
    print(f"\n{'=' * 60}")
    print(f"Verifying: {base_dir}")
    print(f"{'=' * 60}")

    padded_files = sorted([f for f in os.listdir(base_dir) if f.endswith('_padded.csv')])
    if not padded_files:
        print("  No padded files found!")
        return False

    cancer_names = []
    for padded_file in padded_files:
        cancer_names.append(padded_file.replace('_padded.csv', '.csv'))

    print(f"  Found {len(padded_files)} padded files: {padded_files}")
    print(f"  Corresponding originals: {cancer_names}")

    originals = {}
    for name in cancer_names:
        dataframe = load_csv(os.path.join(base_dir, name))
        if dataframe is not None:
            originals[name] = dataframe
            print(f"  {name}: {dataframe.shape[0]} rows x {dataframe.shape[1]} cols")

    gbmlgg_file = 'tcga_gbmlgg_all_clean.csv'
    if gbmlgg_file not in originals:
        dataframe = load_csv(os.path.join(base_dir, gbmlgg_file))
        if dataframe is not None:
            originals[gbmlgg_file] = dataframe
            print(f"  {gbmlgg_file} (no padded version): {dataframe.shape[0]} rows x {dataframe.shape[1]} cols")

    all_columns_set = set()
    for dataframe in originals.values():
        all_columns_set.update(dataframe.columns)
    print(f"\n  Union of all columns: {len(all_columns_set)} columns")

    sample_df = list(originals.values())[0]
    metadata_cols = [c for c in sample_df.columns if not any(c.endswith(suffix) for suffix in ['_mut', '_cnv', '_rnaseq'])]
    print(f"  Metadata columns (non-genomic): {metadata_cols}")

    all_ok = True
    for padded_file in padded_files:
        name = padded_file.replace('_padded.csv', '.csv')
        padded_df = pd.read_csv(os.path.join(base_dir, padded_file), header=0, index_col=0, sep=',', low_memory=False)
        orig_df = originals.get(name)

        if orig_df is None:
            print(f"\n  [SKIP] No original for {padded_file}")
            continue

        print(f"\n  --- Verifying {padded_file} ---")
        print(f"  Original: {orig_df.shape[0]} rows x {orig_df.shape[1]} cols")
        print(f"  Padded:   {padded_df.shape[0]} rows x {padded_df.shape[1]} cols")

        if orig_df.shape[0] != padded_df.shape[0]:
            print(f"  [FAIL] Row count mismatch: {orig_df.shape[0]} vs {padded_df.shape[0]}")
            all_ok = False
        else:
            print(f"  [OK] Row count matches: {orig_df.shape[0]}")

        if padded_df.shape[1] < orig_df.shape[1]:
            print("  [FAIL] Padded has fewer columns than original!")
            all_ok = False
        else:
            new_cols = set(padded_df.columns) - set(orig_df.columns)
            print(f"  [OK] Padded has {len(new_cols)} additional columns")

        common_cols = sorted(set(orig_df.columns) & set(padded_df.columns))
        if len(common_cols) == 0:
            print("  [FAIL] No common columns found!")
            all_ok = False
            continue

        orig_subset = orig_df[common_cols].reset_index(drop=True)
        pad_subset = padded_df[common_cols].reset_index(drop=True)

        values_match = True
        mismatch_cols = []
        for column in common_cols:
            original_series = orig_subset[column]
            padded_series = pad_subset[column]
            match = (original_series == padded_series) | (original_series.isna() & padded_series.isna())
            if not match.all():
                values_match = False
                mismatch_count = (~match).sum()
                mismatch_cols.append((column, mismatch_count))

        if values_match:
            print(f"  [OK] All {len(common_cols)} common columns values preserved exactly")
        else:
            print(f"  [FAIL] Value mismatches in {len(mismatch_cols)} columns:")
            for column, mismatch_count in mismatch_cols[:5]:
                print(f"    {column}: {mismatch_count} mismatched values")
            all_ok = False

        new_cols_list = sorted(set(padded_df.columns) - set(orig_df.columns))
        genomic_new_cols = [column for column in new_cols_list if not column.startswith('Unnamed')]
        index_cols = [column for column in new_cols_list if column.startswith('Unnamed')]

        if index_cols:
            print(f"  [INFO] Ignoring {len(index_cols)} 'Unnamed' index columns (pandas CSV artifact): {index_cols}")

        if genomic_new_cols:
            padded_new = padded_df[genomic_new_cols]
            non_zero_cols = []
            for column in genomic_new_cols:
                values = padded_new[column]
                non_zero = values[(values != 0) & values.notna()]
                if len(non_zero) > 0:
                    non_zero_cols.append((column, len(non_zero)))

            if not non_zero_cols:
                print(f"  [OK] All {len(genomic_new_cols)} new genomic columns are zero-filled")
            else:
                print(f"  [FAIL] {len(non_zero_cols)} new genomic columns have non-zero values:")
                for column, count in non_zero_cols[:5]:
                    print(f"    {column}: {count} non-zero values")
                all_ok = False
        else:
            print("  [INFO] No new genomic columns (all columns already present)")

    padded_dfs = {}
    for padded_file in padded_files:
        padded_dfs[padded_file] = pd.read_csv(os.path.join(base_dir, padded_file), header=0, index_col=0, sep=',', low_memory=False)

    if len(padded_dfs) > 1:
        col_sets = [set(dataframe.columns) for dataframe in padded_dfs.values()]
        if all(col_set == col_sets[0] for col_set in col_sets):
            print(f"\n  [OK] All padded files have the same column set ({len(col_sets[0])} columns)")
        else:
            print("\n  [WARNING] Padded files have DIFFERENT column sets:")
            for padded_file, dataframe in padded_dfs.items():
                print(f"    {padded_file}: {dataframe.shape[1]} cols")
            all_ok = False

    return all_ok


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dirs_to_check = []
    for directory_name in ['dataset_csv_padding']:
        full_path = os.path.join(base, directory_name)
        if os.path.exists(full_path):
            dirs_to_check.append(full_path)
        else:
            print(f"Directory not found: {full_path}")

    if not dirs_to_check:
        print("No padding directories found!")
        sys.exit(1)

    all_ok = True
    for directory_path in dirs_to_check:
        ok = verify_padding_dir(directory_path)
        all_ok = all_ok and ok

    print(f"\n{'=' * 60}")
    if all_ok:
        print("OVERALL: All padding verifications PASSED!")
    else:
        print("OVERALL: Some checks FAILED or had WARNINGS. Please review above.")
    print(f"{'=' * 60}")