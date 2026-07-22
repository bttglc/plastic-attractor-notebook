# Plastic Attractor Models — Task-Switching Controls

Four related attractor-network models of cue-driven, rule-based task-switching (no-cue baseline → cued → gated), tested for whether their behaviour reflects genuine rule use or memorisation of specific cue-stimulus associations, plus decoding analyses of what the gated model represents internally.

Full write-up, methods, and results: https://docs.google.com/document/d/1lYC_vfjHhKIMcs9OkZpM2ylg9BObDuNgVbS4lQHIZi0/edit?pli=1&tab=t.3gb47qriqvod

## What's here

- `original_model/`, `attractor_version1/`, `cued_attractor/`, `gated_attractor/` — the four model versions, each with its own controls and results in `output/`
- `compare_all_models.py` — cross-model comparison
- `congruency_matched_control3.py` — the congruency-controlled version of Control 3
- `decode_gated_attractor.py`, `decode_gated_attractor_stats.py`, `cross_decode_gated_attractor.py` — decoding and cross-cue decoding analyses on `gated_attractor`
- `run_all_controls_colab.ipynb` — run all four models' controls on Google Colab
- `decoding_colab.ipynb` — run the decoding analyses on Google Colab
