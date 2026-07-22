"""Combine the four models' controls_summary.json files into one comparison.

Run each model's own controls script first (they're independent, don't
share a package, and take a while at full scale):

    python "attractor version1/attractor_control_analysis.py"
    python cued_attractor/run_controls.py
    python gated_attractor/run_controls.py
    python "original model/run_baseline_and_control3.py"

Each writes a controls_summary.json in a common schema (mean/SEM/t/p per
condition, null where a control doesn't apply to that model -- see any of
those scripts' own JSON-export section for the exact fields). This script
just reads the four files back and plots them side by side; it does not run
any simulations itself, so it's fast regardless of how large n_seeds was.

Run it from the updated_model folder, after the four scripts above:

    python compare_all_models.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = os.getcwd()

SUMMARY_PATHS = {
    'Original\nmodel': os.path.join(_this_dir, 'original model', 'output', 'controls_summary.json'),
    'Updated\nmodel v1': os.path.join(_this_dir, 'attractor version1', 'output', 'controls_summary.json'),
    'Cued\nattractor': os.path.join(_this_dir, 'cued_attractor', 'output', 'controls', 'controls_summary.json'),
    'Gated\nattractor': os.path.join(_this_dir, 'gated_attractor', 'output', 'controls', 'controls_summary.json'),
}

output_dir = os.path.join(_this_dir, 'output')


def load_summaries():
    summaries = {}
    missing = []
    for label, path in SUMMARY_PATHS.items():
        if os.path.exists(path):
            with open(path) as f:
                summaries[label] = json.load(f)
        else:
            missing.append((label, path))
    if missing:
        print('Missing summaries (run that model\'s controls script first):')
        for label, path in missing:
            print(f'  {label.replace(chr(10), " ")}: {path}')
    return summaries


def _stars(p):
    if p is None:
        return ''
    return ' *' if p < .05 else ''


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    summaries = load_summaries()
    if not summaries:
        raise SystemExit('No controls_summary.json files found -- run the per-model scripts first.')

    labels = list(summaries.keys())
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # ---------------- Panel 1: baseline accuracy, every model ----------------
    means = [summaries[m]['baseline_accuracy_mean'] for m in labels]
    sems = [summaries[m]['baseline_accuracy_sem'] for m in labels]
    ax[0, 0].bar(labels, means, yerr=sems, color='steelblue', alpha=.8)
    ax[0, 0].set_ylim(0, 1.1)
    ax[0, 0].set_title('Baseline real-block accuracy')
    ax[0, 0].axhline(0.5, color='grey', linestyle=':', label='Chance (2AFC)')
    ax[0, 0].legend()

    # ---------------- Panel 2: Control 1, models with a cue mechanism only --
    c1_labels = [m for m in labels if summaries[m]['control1_accuracy_mean'] is not None]
    x = np.arange(len(c1_labels))
    width = 0.35
    baseline_means = [summaries[m]['baseline_accuracy_mean'] for m in c1_labels]
    baseline_sems = [summaries[m]['baseline_accuracy_sem'] for m in c1_labels]
    c1_means = [summaries[m]['control1_accuracy_mean'] for m in c1_labels]
    c1_sems = [summaries[m]['control1_accuracy_sem'] for m in c1_labels]
    ax[0, 1].bar(x - width / 2, baseline_means, width, yerr=baseline_sems, label='Baseline', color='blue', alpha=.7)
    ax[0, 1].bar(x + width / 2, c1_means, width, yerr=c1_sems, label='Shuffled cues', color='red', alpha=.7)
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(c1_labels)
    ax[0, 1].set_ylim(0, 1.1); ax[0, 1].legend()
    p_labels = [
        f'p={summaries[m]["control1_p"]:.3f}{_stars(summaries[m]["control1_p"])}' for m in c1_labels
    ]
    ax[0, 1].set_title('Control 1: baseline vs shuffled cues\n' + ', '.join(p_labels))
    if not c1_labels:
        ax[0, 1].text(0.5, 0.5, 'No cue mechanism\nin any available model', ha='center', va='center', transform=ax[0, 1].transAxes)

    # ---------------- Panel 3: Control 2, models with a cue mechanism only --
    c2_labels = [m for m in labels if summaries[m]['control2_trained_mean'] is not None]
    x = np.arange(len(c2_labels))
    trained_means = [summaries[m]['control2_trained_mean'] for m in c2_labels]
    trained_sems = [summaries[m]['control2_trained_sem'] for m in c2_labels]
    novel_means = [summaries[m]['control2_novel_mean'] for m in c2_labels]
    novel_sems = [summaries[m]['control2_novel_sem'] for m in c2_labels]
    ax[1, 0].bar(x - width / 2, trained_means, width, yerr=trained_sems, label='Trained cues', color='seagreen', alpha=.8)
    ax[1, 0].bar(x + width / 2, novel_means, width, yerr=novel_sems, label='Novel cues', color='lightgreen', alpha=.8)
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(c2_labels)
    ax[1, 0].set_ylim(0, 1.1); ax[1, 0].legend()
    p_labels = [
        f'p={summaries[m]["control2_p"]:.3f}{_stars(summaries[m]["control2_p"])}' for m in c2_labels
    ]
    ax[1, 0].set_title('Control 2: trained vs novel cue\n' + ', '.join(p_labels))
    if not c2_labels:
        ax[1, 0].text(0.5, 0.5, 'No cue mechanism\nin any available model', ha='center', va='center', transform=ax[1, 0].transAxes)

    # ---------------- Panel 4: Control 3, every model (translates everywhere) --
    c3_labels = [m for m in labels if summaries[m]['control3_other_mean'] is not None]
    x = np.arange(len(c3_labels))
    other_means = [summaries[m]['control3_other_mean'] for m in c3_labels]
    other_sems = [summaries[m]['control3_other_sem'] for m in c3_labels]
    omitted_means = [summaries[m]['control3_omitted_mean'] for m in c3_labels]
    omitted_sems = [summaries[m]['control3_omitted_sem'] for m in c3_labels]
    ax[1, 1].bar(x - width / 2, other_means, width, yerr=other_sems, label='Other combos', color='navajowhite', alpha=.9)
    ax[1, 1].bar(x + width / 2, omitted_means, width, yerr=omitted_sems, label='Omitted combo', color='darkorange', alpha=.9)
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(c3_labels)
    ax[1, 1].set_ylim(0, 1.1); ax[1, 1].legend()
    p_labels = [
        f'p={summaries[m]["control3_p"]:.3f}{_stars(summaries[m]["control3_p"])}' for m in c3_labels
    ]
    ax[1, 1].set_title('Control 3: other vs omitted combo (all models)\n' + ', '.join(p_labels))

    fig.suptitle('Controls 1-3 across all four model architectures', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'all_models_comparison.png'), dpi=300)
    plt.show()

    # ---------------- Printed table ----------------
    print('\n=== CROSS-MODEL SUMMARY ===\n')
    col_widths = (20, 4, 12, 20, 26, 26)
    headers = ('Model', 'n', 'Baseline', 'Shuffled (C1)', 'Trained/Novel (C2)', 'Other/Omitted (C3)')
    header = ''.join(f'{h:<{w}}' for h, w in zip(headers, col_widths))
    print(header)
    print('-' * len(header))
    for m in labels:
        s = summaries[m]
        name = m.replace('\n', ' ')
        baseline = f'{s["baseline_accuracy_mean"]:.3f}'
        c1 = (
            f'{s["control1_accuracy_mean"]:.3f} (p={s["control1_p"]:.3f}{_stars(s["control1_p"])})'
            if s['control1_accuracy_mean'] is not None else 'n/a'
        )
        c2 = (
            f'{s["control2_trained_mean"]:.3f}/{s["control2_novel_mean"]:.3f} '
            f'(p={s["control2_p"]:.3f}{_stars(s["control2_p"])})'
            if s['control2_trained_mean'] is not None else 'n/a'
        )
        c3 = (
            f'{s["control3_other_mean"]:.3f}/{s["control3_omitted_mean"]:.3f} '
            f'(p={s["control3_p"]:.3f}{_stars(s["control3_p"])})'
            if s['control3_other_mean'] is not None else 'n/a'
        )
        row = (name, str(s['n_seeds']), baseline, c1, c2, c3)
        print(''.join(f'{cell:<{w}}' for cell, w in zip(row, col_widths)))
    print('\n* p < .05. Controls 1/2 are n/a for the original model: no cue mechanism to shuffle or restrict.')

    # ---------------- Equivalence (TOST) table -- only meaningful for C2/C3 ----------------
    # A non-significant t-test above only means "failed to reject a
    # difference," not "these are equivalent." tost_equivalent (written by
    # each model's own script) is the field that actually licenses a
    # "performs just as well as" claim -- missing/None if that model's
    # summary predates this field, or if the control doesn't apply.
    has_tost = any(s.get('control2_tost_p') is not None or s.get('control3_tost_p') is not None for s in summaries.values())
    if has_tost:
        print('\n=== EQUIVALENCE (TOST), Controls 2/3 only ===\n')
        eq_widths = (20, 36, 36)
        eq_headers = ('Model', 'Novel vs trained cue (C2)', 'Omitted vs other combo (C3)')
        eq_header = ''.join(f'{h:<{w}}' for h, w in zip(eq_headers, eq_widths))
        print(eq_header)
        print('-' * len(eq_header))
        for m in labels:
            s = summaries[m]
            name = m.replace('\n', ' ')

            def _tost_cell(prefix):
                d = s.get(f'{prefix}_cohens_d')
                tp = s.get(f'{prefix}_tost_p')
                eq = s.get(f'{prefix}_equivalent')
                if d is None or tp is None:
                    return 'n/a'
                verdict = 'EQUIVALENT' if eq else 'not equiv.'
                return f'd={d:.2f}, TOST p={tp:.3f} ({verdict})'

            row = (name, _tost_cell('control2'), _tost_cell('control3'))
            print(''.join(f'{cell:<{w}}' for cell, w in zip(row, eq_widths)))
        print("\nbound = ±equivalence_bound accuracy points (see each model's script; default 0.05).")
        print('EQUIVALENT means the difference is significantly smaller than that bound -- a real')
        print('equivalence claim, not just a large p-value on the standard t-test.')

    print(f'\nFigure saved to {os.path.join(output_dir, "all_models_comparison.png")}')
