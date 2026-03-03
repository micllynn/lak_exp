# CLAUDE.md

## Project overview

This is a Python package for loading and analyzing complex rodent behaviour experiments
with simultaneous neural recordings (two-photon imaging or high-density neurophysiology).
It focuses on parsing complex behavioural readouts and aligning with neural recordings
for complex plotting and analysis.

## Data types
- Behaviour: standardized Rigbox behavioural outputs (.Block and .Timeline files)
- Imaging (Bruker/Suite2P pipeline): .tif stacks up to 20GB (memmap required) and Suite2P
.npy outputs.
- Neurophysiology (Neuropixels/Kilosort pipeline): Standard Kilosort .npy objects.

## Key modules
| Module | Purpose |
|---|---|
| `load_exp_twop.py` | `TwoPRec`, `TwoPRec_DualColour` — main two-photon loaders |
| `load_exp.py` | `ExpObj_ReportOpto`, `ExpObj_ValuePFC` — neurophysiology loaders |
| `beh.py` | `BlockParser`, `TimelineParser`, `StimParserNew` — behavior parsing |
| `exp_defs.py` | `ExpSubtypes` — `beh_type` → subtypes dict factory |
| `dset_twop.py` / `dset.py` | Dataset metadata (CSV-backed) |
| `align_imgbeh.py` / `align_ephysbeh.py` | Clock alignment |
| `signal_correction.py` | Dual-colour artefact correction (linear, PCA, ICA, NMF) |
| `batch_run.py` | `batch_run()` — multi-experiment loading/processing |
| `cmd_line_tools/beh_check.py` | `beh-check` CLI — daily behavior summary |

## TwoPRec (primary class)

```python
rec = TwoPRec(
    enclosing_folder='/data/MBL001/2025-01-10/',
    folder_beh='1',
    folder_img='TwoP/2025-01-10_t-001',
    fname_img='..._Ch2.tif',
    beh_type='visual_pavlov',  # or 'auditory_pavlov', or None (manual)
    rec_type='trig_rew',       # or 'paqio'
)
```

Key attributes: `.rec` (tiff memmap), `.rec_t`, `.beh.tr_inds`, `.beh.tr_conds`, `.neur` (suite2p)

Key methods: `.add_frame()`, `.add_sectors()`, `.add_neurs()`, `.plt_frame()`, `.plt_sectors()`

## beh_type presets

`beh_type` drives `parse_by` and `subtypes` passed to `StimParserNew`.
Subtypes are defined in `ExpSubtypes` (`exp_defs.py`); add new presets there.

| `beh_type` | `parse_by` | subtypes |
|---|---|---|
| `'visual_pavlov'` (default) | `'stimulusOrientation'` | `'0'`, `'0.5'`, `'1'`, `'0.5_rew'`, `'0.5_norew'`, `'0.5_prelick'`, `'0.5_noprelick'` |
| `'auditory_pavlov'` | `'stimulusType'` | none |
| `None` | must pass `parse_by` explicitly | pass `subtypes` manually |

## Code conventions

- Classes: `PascalCase`; methods: `snake_case`; private: `_prefix`
- Data containers: `SimpleNamespace` (`.beh`, `.ops`, `.neur`)
- NumPy-style docstrings; no type annotations
- 4-space indentation
