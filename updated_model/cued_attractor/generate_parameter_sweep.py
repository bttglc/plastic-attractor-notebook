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

# ---- Define your parameter grid ----
param_grid = {
    'conjunction_lateral_weight': [-0.25, -0.45, -0.65],
    'conjunction_to_feature_gain': [0.04, 0.08, 0.16],
    'feature_lateral_weight': [-0.14, -0.28, -0.56],
    'feature_to_conjunction_gain': [0.02, 0.04, 0.08],
    'maximum_fast_weight': [0.2, 0.5, 1.0],
    'maximum_slow_weight': [0.2, 0.5, 1.0],
}


# ---- Fixed parameters (same for all versions) ----
fixed_params = {
    'num_cues_per_rule': 2,
    'number_of_conjunction_units': 4,
    'conjunction_self_weight': 1.0,
    'feature_self_weight': 0.73,
    'baseline_activity': 0.175,
    'fast_learning_rate': 0.02,
    'slow_learning_rate': 0.0002,
    'fast_weight_blend': 1.0,
    'slow_weight_blend': 1.0,
}

# ---- Generate all combinations ----
keys = list(param_grid.keys())
values = list(param_grid.values())
combinations = list(itertools.product(*values))

print(f"Total combinations: {len(combinations)}")
print(f"Number of parameters varied: {len(keys)}")
print()

# ---- Generate ModelParameters definitions ----
all_versions = []

for idx, combo in enumerate(combinations):
    # Create a dictionary of all parameters (fixed + varied)
    params_dict = dict(fixed_params)
    for k, v in zip(keys, combo):
        params_dict[k] = v
    
    # Create a version name
    version_name = f"v{idx:04d}"
    for k, v in zip(keys, combo):
        # Convert value to a compact string for the filename
        if isinstance(v, float):
            if v < 0:
                v_str = f"neg{abs(v):.2f}".replace('.', '')
            else:
                v_str = f"{v:.2f}".replace('.', '')
        else:
            v_str = str(v)
        version_name += f"_{k}{v_str}"
    
    all_versions.append((version_name, params_dict))

# ---- Print all definitions to console ----
print("=" * 80)
print("MODELPARAMETERS DEFINITIONS")
print("=" * 80)

for version_name, params in all_versions[:5]:  # Show first 5 as example
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
    f.write('"""Auto-generated model versions for parameter sweep.')
    f.write(f'\nGenerated {len(all_versions)} combinations.\n"""\n\n')
    
    f.write('from cued_attractor import ModelParameters\n\n')
    
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
print(f"Parameters varied: {len(param_grid)}")
print()
print("Parameter ranges:")
for key, values in param_grid.items():
    print(f"  {key}: {values}")
print()
print("Fixed parameters:")
for key, value in fixed_params.items():
    print(f"  {key}: {value}")
print()
print(f"Output saved to: {output_file}")
print(f"Version list saved to: versions.txt")
print()
print("To use in SLURM:")
print(f"  #SBATCH --array=0-{len(all_versions)-1}")

# ---- Optional: Save as CSV for reference ----
import csv
csv_file = 'parameter_grid.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header
    header = ['version_name'] + list(fixed_params.keys()) + list(param_grid.keys())
    writer.writerow(header)
    # Data
    for version_name, params in all_versions:
        row = [version_name] + [params[k] for k in fixed_params.keys()] + [params[k] for k in param_grid.keys()]
        writer.writerow(row)

print(f"Parameter grid saved to: {csv_file}")