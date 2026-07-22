#!/usr/bin/env python
"""
Collect all summary CSV files from individual jobs and combine into one master CSV.
Run this after all HPC jobs complete.
"""

import os
import glob
import pandas as pd

def collect_results(base_output_dir='./output', output_file='master_results.csv'):
    """Collect all version summary files into one master CSV."""
    
    summary_files = glob.glob(os.path.join(base_output_dir, '*', '*_summary.csv'))
    
    if not summary_files:
        print(f"No summary files found in {base_output_dir}")
        return None
    
    print(f"Found {len(summary_files)} summary files")
    
    dfs = []
    for f in sorted(summary_files):
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  ✓ {os.path.basename(f)}")
        except Exception as e:
            print(f"  ✗ Error reading {f}: {e}")
    
    if not dfs:
        print("No valid results found!")
        return None
    
    master_df = pd.concat(dfs, ignore_index=True)
    master_df.to_csv(output_file, index=False)
    
    print(f"\nSUCCESS: Combined {len(dfs)} results to {output_file}")
    print(f"Total runs: {len(master_df)}")
    print(f"Columns: {list(master_df.columns)}")
    
    return master_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='./output')
    parser.add_argument('--output-file', default='master_results.csv')
    args = parser.parse_args()
    
    collect_results(args.output_dir, args.output_file)