#!/usr/bin/env python
"""
Generate all combinations of model parameters for sweeping.

Run this script to:
1. Print all ModelParameters definitions
2. Save them to model_versions_config.py
3. Create a versions.txt file for SLURM array jobs
"""

import itertools
import os
import csv

# ---- Define your parameter grids ----
# Grid 1: full Cartesian product over 7 parameters
param_grid1 = {
    'conjunction_lateral_weight': [-0.25, -0.45, -0.65],
    'conjunction_to_feature_gain': [0.04, 0.08, 0.16],
    'feature_lateral_weight': [-0.14, -0.28, -0.56],
    'feature_to_conjunction_gain': [0.02, 0.04, 0.08],
    'maximum_fast_weight': [0.2, 0.5, 1.0],
    'maximum_slow_weight': [0.2, 0.5, 1.0],
    'gating_self_weight': [1.0, 1.6, 2.0]
}

# Grid 2: pairs for (max_fast, max_slow) and (gating_max_fast, gating_max_slow)
# Define two sets of three pairs
pair_set_A = [(0.2, 1.0), (0.5, 0.5), (1.0, 0.2)]   # for maximum_fast/slow_weight
pair_set_B = [(0.2, 1.0), (0.5, 0.5), (1.0, 0.2)]   # for gating_maximum_fast/slow_weight

# Other parameters to combine with each of the 9 cap combinations
param_grid2_other = {
    'conjunction_lateral_weight': [-0.25, -0.65],
    'conjunction_to_feature_gain': [0.04, 0.16],
    'feature_lateral_weight': [-0.14, -0.56],
    'feature_to_conjunction_gain': [0.02, 0.08],
    'gating_self_weight': [1.0, 1.6, 2.0]
}

# ---- Fixed parameters (same for all versions of BOTH grids) ----
base_fixed_params = {
    'num_cues_per_rule': 2,
    'number_of_conjunction_units': 4,
    'conjunction_self_weight': 1.0,
    'feature_self_weight': 0.73,
    'baseline_activity': 0.175,
    'fast_learning_rate': 0.02,
    'slow_learning_rate': 0.0002,
    'fast_weight_blend': 1.0,
    'slow_weight_blend': 1.0,
    # gating parameters that are not varied
    'number_of_gating_units': 2,
    'cue_to_gating_gain': 1.0,
    'gating_to_feature_gain': 0.7,
    'gating_to_relevant_feature_gain': 0.6,
    'gating_fast_learning_rate': 0.02,
    'gating_slow_learning_rate': 0.0002,
    'gating_fast_weight_blend': 1.0,
    'gating_slow_weight_blend': 1.0,
    'gating_trace_decay': 0.98,
    'gating_lateral_weight': -0.45,
    # These defaults will be overridden by the grids as needed
    'maximum_fast_weight': 1.0,
    'maximum_slow_weight': 0.2,
    'gating_maximum_fast_weight': 1.0,
    'gating_maximum_slow_weight': 0.2,
    'gating_self_weight': 1.6,
}

# ---- Helper to generate versions from Grid 1 (full product) ----
def generate_from_full_grid(grid, prefix):
    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    print(f"Grid {prefix}: {len(combos)} combinations from full product")
    print(f"  Parameters varied: {keys}")

    versions = []
    for idx, combo in enumerate(combos):
        params_dict = dict(base_fixed_params)
        for k, v in zip(keys, combo):
            params_dict[k] = v
        version_name = f"{prefix}_v{idx:04d}"
        for k, v in zip(keys, combo):
            if isinstance(v, float):
                if v < 0:
                    v_str = f"neg{abs(v):.2f}".replace('.', '')
                else:
                    v_str = f"{v:.2f}".replace('.', '')
            else:
                v_str = str(v)
            version_name += f"_{k}{v_str}"
        versions.append((version_name, params_dict))
    return versions

# ---- Helper to generate versions from Grid 2 (pair product) ----
def generate_from_paired_grid(other_grid, pair_set_A, pair_set_B, prefix):
    # Other parameters: keys and values
    other_keys = list(other_grid.keys())
    other_values = list(other_grid.values())
    other_combos = list(itertools.product(*other_values))
    # Product of the two sets of pairs
    cap_combos = list(itertools.product(pair_set_A, pair_set_B))
    total = len(cap_combos) * len(other_combos)
    print(f"Grid {prefix}: {total} combinations (caps: {len(cap_combos)} × others: {len(other_combos)})")
    print(f"  Other parameters: {other_keys}")
    print(f"  Pair set A (max_fast, max_slow): {pair_set_A}")
    print(f"  Pair set B (gating_max_fast, gating_max_slow): {pair_set_B}")

    versions = []
    idx = 0
    for (max_fast, max_slow), (gmax_fast, gmax_slow) in cap_combos:
        for other_combo in other_combos:
            params_dict = dict(base_fixed_params)
            # Set the four cap parameters
            params_dict['maximum_fast_weight'] = max_fast
            params_dict['maximum_slow_weight'] = max_slow
            params_dict['gating_maximum_fast_weight'] = gmax_fast
            params_dict['gating_maximum_slow_weight'] = gmax_slow
            # Set the other parameters
            for k, v in zip(other_keys, other_combo):
                params_dict[k] = v
            # Build version name
            version_name = f"{prefix}_v{idx:04d}"
            version_name += f"_mfast{max_fast:.1f}_mslow{max_slow:.1f}"
            version_name += f"_gfast{gmax_fast:.1f}_gslow{gmax_slow:.1f}"
            for k, v in zip(other_keys, other_combo):
                if isinstance(v, float):
                    if v < 0:
                        v_str = f"neg{abs(v):.2f}".replace('.', '')
                    else:
                        v_str = f"{v:.2f}".replace('.', '')
                else:
                    v_str = str(v)
                version_name += f"_{k}{v_str}"
            versions.append((version_name, params_dict))
            idx += 1
    return versions

# ---- Generate all versions ----
all_versions = []

# Grid 1
versions_g1 = generate_from_full_grid(param_grid1, 'g1')
all_versions.extend(versions_g1)

# Grid 2
versions_g2 = generate_from_paired_grid(param_grid2_other, pair_set_A, pair_set_B, 'g2')
all_versions.extend(versions_g2)

print(f"\nTotal combinations: {len(all_versions)}")
print()

# ---- Print first few definitions to console ----
print("=" * 80)
print("MODELPARAMETERS DEFINITIONS (first 5)")
print("=" * 80)
for version_name, params in all_versions[:5]:
    print(f"    '{version_name}': ModelParameters(")
    for key, value in params.items():
        print(f"        {key}={value},")
    print("    ),")
    print()
if len(all_versions) > 5:
    print(f"    ... and {len(all_versions) - 5} more versions")
    print()

# ---- Write to model_versions_config.py ----
output_file = 'model_versions_config.py'
with open(output_file, 'w') as f:
    f.write('"""Auto-generated model versions for parameter sweep (gated model).')
    f.write(f'\nGenerated {len(all_versions)} combinations from two grids.\n"""\n\n')
    f.write('from gated_attractor import ModelParameters\n\n')
    f.write('model_versions = {\n')
    for version_name, params in all_versions:
        f.write(f"    '{version_name}': ModelParameters(\n")
        for key, value in params.items():
            f.write(f"        {key}={value},\n")
        f.write("    ),\n")
    f.write('}\n')

print(f"Written {len(all_versions)} versions to {output_file}")

# ---- Write versions.txt for SLURM ----
with open('versions.txt', 'w') as f:
    for version_name, _ in all_versions:
        f.write(version_name + '\n')

print(f"Written {len(all_versions)} version names to versions.txt")

# ---- Summary statistics ----
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total parameter combinations: {len(all_versions)}")
print("Grid 1 (full product):")
for key, values in param_grid1.items():
    print(f"  {key}: {values}")
print()
print("Grid 2 (pair products):")
print("  Pair set A (max_fast, max_slow):", pair_set_A)
print("  Pair set B (gating_max_fast, gating_max_slow):", pair_set_B)
print("  Other params:")
for key, values in param_grid2_other.items():
    print(f"    {key}: {values}")
print()
print("Fixed parameters (base):")
for key, value in base_fixed_params.items():
    print(f"  {key}: {value}")
print()
print(f"Output saved to: {output_file}")
print(f"Version list saved to: versions.txt")
print()
print("To use in SLURM:")
print(f"  #SBATCH --array=0-{len(all_versions)-1}")

# ---- Save as CSV for reference ----
csv_file = 'parameter_grid.csv'
all_keys = set()
for _, params in all_versions:
    all_keys.update(params.keys())
all_keys = sorted(all_keys)

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['version_name'] + list(all_keys)
    writer.writerow(header)
    for version_name, params in all_versions:
        row = [version_name] + [params.get(k, '') for k in all_keys]
        writer.writerow(row)

print(f"Parameter grid saved to: {csv_file}")