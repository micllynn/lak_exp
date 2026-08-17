"""
batch_run.py  --  Batch processing for 2-photon experiments.

This module groups three layers of batch utilities:

1.  ``batch_run`` -- generic primitive. Takes a pre-built dict of
    ``{exp_id: kwargs}``, instantiates a (configurable) recording class,
    and runs an arbitrary sequence of ``steps`` (method names, callables)
    on each. Supports both single-colour ``TwoPRec`` (rec_type='paqio')
    and any other rec_class.

2.  ``bulk_run_twop_correction`` -- dual-colour wrapper. Reads an
    expref-style filelist .csv, infers folder paths for each row, and
    delegates to ``batch_run`` to load each recording with
    ``TwoPRec_DualColour`` and run ``plt_qc``.

3.  ``plt_grab_ctrl_dualcolour`` -- cross-recording summary plot. Loads the
    `_qc_response_mean*.npy` outputs left in each recording's
    ``data_mbl/`` folder by ``plt_qc`` and draws a 6-panel cohort
    figure (whole-frame averages, per-sector heatmaps, paired
    Pearson / Spearman dot plots, and integral histogram / scatter).

4.  ``plt_summary`` -- same figure plus two knobs: pass
    ``correct_signal_kwargs`` to load one specific correction run out of
    a sweep rather than whichever file is newest; ``plt_cells=True`` to
    add a pooled per-cell heatmap row from the
    `*_celltraces_trialavg.npy` outputs. Axis/colour scaling is
    data-driven throughout; ``sector_vlim`` / ``cell_vlim`` optionally
    pin the sector- / cell-heatmap colour range to an explicit value
    (e.g. to compare two cohorts on the same scale).

batch_run usage
---------------
from lak_exp.batch_run import batch_run
from lak_exp.load_exp_twop import TwoPRec

EXP_DICTS = {
    'MBL001_2025-01-10': dict(
        enclosing_folder='/path/to/MBL001/2025-01-10/',
        folder_beh='1',
        folder_img='TwoP/2025-01-10_t-001',
        fname_img='2025-01-10_t-001_Cycle00001_Ch2.tif',
        beh_type='visual_pavlov',
    ),
}

# steps can be:
#   - a method name string              -> exp.method()
#   - a (name, kwargs) tuple            -> exp.method(**kwargs)
#   - a (name, args, kwargs) tuple      -> exp.method(*args, **kwargs)
#   - a callable                        -> fn(exp)

results = batch_run(
    EXP_DICTS,
    steps=['add_neurs', ('add_sectors', {'use_zscore': True})],
    rec_class=TwoPRec,
    rec_type='paqio',
)

Dual-colour usage
-----------------
from lak_exp.batch_run import bulk_run_twop_correction, plt_grab_ctrl_dualcolour

# 1) Run QC + signal correction on each recording in the cohort
bulk_run_twop_correction(
    csv_path='/path/to/cohort_filelist.csv',
    data_folder='/Volumes/T7/5HTCtx',
)

# 2) Build the cross-recording summary plot from the saved npy outputs
plt_grab_ctrl_dualcolour(
    csv_path='/path/to/cohort_filelist.csv',
    data_folder='/Volumes/T7/5HTCtx',
    sort_by='red_corr',
    sector_corr_metric='pearson',
)
"""

import os
import re
import time
import warnings
import traceback
from types import SimpleNamespace

import numpy as np
import scipy.stats as sp_stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd

from .load_exp_twop import TwoPRec, TwoPRec_DualColour
from .twop_qc import corrsig_suffix_from_kwargs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_step(exp, step):
    """Dispatch a single step on *exp*.

    Accepted step formats
    ---------------------
    str                         ->  exp.<str>()
    (str,)                      ->  exp.<str>()
    (str, dict)                 ->  exp.<str>(**dict)
    (str, list/tuple, dict)     ->  exp.<str>(*list, **dict)
    callable                    ->  step(exp)
    """
    if callable(step):
        return step(exp)

    if isinstance(step, str):
        method_name, args, kwargs = step, [], {}
    elif isinstance(step, (list, tuple)):
        if len(step) == 1:
            method_name, args, kwargs = step[0], [], {}
        elif len(step) == 2:
            method_name, second = step
            if isinstance(second, dict):
                args, kwargs = [], second
            else:
                args, kwargs = list(second), {}
        elif len(step) == 3:
            method_name, args, kwargs = step[0], list(step[1]), step[2]
        else:
            raise ValueError(f'Step tuple has too many elements: {step!r}')
    else:
        raise TypeError(f'Unrecognised step format: {step!r}')

    method = getattr(exp, method_name)
    return method(*args, **kwargs)


# ---------------------------------------------------------------------------
# Main batch function
# ---------------------------------------------------------------------------

def batch_run(exp_dicts, steps=None, rec_class=TwoPRec, **rec_kwargs):
    """Batch-load experiments and run a sequence of steps on each.

    Parameters
    ----------
    exp_dicts : dict
        Mapping of ``{exp_id: params_dict}`` where each ``params_dict`` may
        contain any subset of:
            enclosing_folder, folder_beh, folder_img, fname_img, beh_type,
            parse_by, subtypes, trial_end, rec_type, n_px_remove_sides
        and any other keyword accepted by ``rec_class.__init__``.
        Per-experiment values override the shared ``**rec_kwargs``.
    steps : list or None
        Sequence of steps to run on each successfully loaded experiment.
        Each step can be:
            - a method name string   ->  ``exp.method()``
            - ``(name, kwargs)``     ->  ``exp.method(**kwargs)``
            - ``(name, args, kw)``   ->  ``exp.method(*args, **kw)``
            - a callable             ->  ``fn(exp)``
        If ``None``, only loading is performed.
    rec_class : class
        Class to instantiate for each experiment (default ``TwoPRec``).
        Use ``TwoPRec_DualColour`` for dual-colour recordings.
    **rec_kwargs
        Shared keyword arguments forwarded to ``rec_class.__init__`` for every
        experiment.  Per-experiment values in ``exp_dicts`` take precedence.

    Returns
    -------
    dict
        ``{exp_id: {'exp': <loaded object or None>, 'error': <str or None>}}``
    """
    if steps is None:
        steps = []

    results = {}

    for exp_id, params in exp_dicts.items():
        print(f'\n{"="*60}')
        print(f'  Experiment: {exp_id}')
        print(f'{"="*60}')

        result = {'exp': None, 'error': None}

        # --- load ---
        try:
            init_kwargs = {**rec_kwargs, **params}
            exp = rec_class(**init_kwargs)
            result['exp'] = exp
        except Exception:
            msg = traceback.format_exc()
            print(f'[LOAD ERROR] {exp_id}:\n{msg}')
            result['error'] = f'load: {msg}'
            results[exp_id] = result
            continue

        # --- steps ---
        for step in steps:
            step_label = step if isinstance(step, str) else repr(step)
            try:
                _call_step(exp, step)
            except Exception:
                msg = traceback.format_exc()
                print(f'[STEP ERROR] {exp_id} | step={step_label}:\n{msg}')
                result['error'] = f'step {step_label!r}: {msg}'
                break  # skip remaining steps for this experiment

        results[exp_id] = result

    # summary
    n_ok = sum(1 for r in results.values() if r['error'] is None)
    n_fail = len(results) - n_ok
    print(f'\n{"="*60}')
    print(f'  Done.  {n_ok}/{len(results)} experiments succeeded'
          + (f', {n_fail} failed.' if n_fail else '.'))
    print(f'{"="*60}\n')

    return results


# ===========================================================================
# Dual-colour 2p batch helpers
# ===========================================================================
# Everything below this point is specific to dual-colour TwoPRec_DualColour
# cohorts driven by an expref-style filelist .csv (see CLAUDE.md).
# ===========================================================================


# Channel colour scheme used by plt_grab_ctrl_dualcolour
_CHANNEL_COLOURS = {
    'red': '#d62728',
    'grn': '#2ca02c',
    'red_corr': '#7f3fbf',
}
_CHANNEL_LABELS = {
    'red': 'red',
    'grn': 'green',
    'red_corr': r'red$_\mathrm{corrected}$',
}
# Per-cell donut-ring traces; xkcd 'bright orange', matching the mean
# trace in twop_qc._plt_qc_response_mean_cells. Hard-coded rather than
# looked up through seaborn so this module keeps its import list.
_CELL_COLOUR = '#ff5b00'


# Pretty-printing helpers
# -------------------------------------------------------

def _fmt_time(secs):
    """Return a human-readable H:MM:SS string for a duration in seconds."""
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f'{h}:{m:02d}:{s:02d}'
    return f'{m:02d}:{s:02d}'


def _print_banner(title, width=72, char='='):
    """Print a centered banner line."""
    print('\n' + char * width)
    print(title.center(width))
    print(char * width, flush=True)


def _print_status(msg, symbol='>'):
    """Print an indented status line."""
    print(f'  {symbol} {msg}', flush=True)


# Expref parsing
# -------------------------------------------------------

def _parse_expref(expref):
    """Split an expref like 'MBL036_2026-04-21_1' into (animal, date, beh).
    """
    parts = expref.split('_')
    if len(parts) < 3:
        raise ValueError(
            f'expref {expref!r} does not match '
            f'expected ANIMAL_DATE_BEH format')
    animal, date, folder_beh = parts[0], parts[1], parts[2]
    return animal, date, folder_beh


def _resolve_img_folder(enclosing_folder, date, dual2p_folder):
    """Infer the imaging subfolder from the dual2p_folder tag.

    The csv column ``dual2p_folder`` typically holds a tag like 't001'
    (or '001' or 't-001'); the on-disk folder is named
    'TwoP/{date}_t-{NNN}'. This first tries that canonical pattern, and
    falls back to scanning TwoP/ for a folder whose name ends in the
    given tag.
    """
    # Pull the numeric suffix from things like 't001', 't-001', '001'
    m = re.search(r'(\d+)$', str(dual2p_folder))
    twop_dir = os.path.join(enclosing_folder, 'TwoP')

    if m is not None:
        nnn = m.group(1).zfill(3)
        candidate = f'{date}_t-{nnn}'
        candidate_full = os.path.join(twop_dir, candidate)
        if os.path.isdir(candidate_full):
            return os.path.join('TwoP', candidate)

    # Fallback: scan TwoP/ for a folder ending in the dual2p_folder tag
    if os.path.isdir(twop_dir):
        for entry in os.listdir(twop_dir):
            if entry.startswith('.'):
                continue
            if entry.endswith(str(dual2p_folder)) \
                    and os.path.isdir(os.path.join(twop_dir, entry)):
                return os.path.join('TwoP', entry)

    raise FileNotFoundError(
        f'Could not locate imaging folder for dual2p_folder='
        f'{dual2p_folder!r} under {twop_dir}')


def _build_exp_dicts_from_csv(csv_path,
                              data_folder,
                              fname_img_red,
                              fname_img_grn,
                              beh_type,
                              rec_type,
                              extra_load_kwargs,
                              verbose=True):
    """Read a filelist .csv and build a batch_run-compatible exp_dicts.

    Skips (with a printed error) any row whose expref cannot be parsed
    or whose imaging folder cannot be located on disk. Honours a per-row
    'rec_type' csv column when present.
    """
    filelist = pd.read_csv(csv_path)
    exp_dicts = {}
    skipped = []

    for i, row in enumerate(filelist.to_dict('records')):
        try:
            expref = str(row['expref'])
        except Exception:
            expref = f'row_{i}'

        try:
            animal, date, folder_beh = _parse_expref(expref)
            enclosing_folder = os.path.join(data_folder, animal, date)
            folder_img = _resolve_img_folder(
                enclosing_folder, date, row['dual2p_folder'])
        except Exception as e:
            if verbose:
                _print_status(f'SKIP {expref}: {e}', symbol='x')
            skipped.append((expref, str(e)))
            continue

        # Per-row rec_type override from the csv, if present
        _rec_type = rec_type
        _row_rec_type = row.get('rec_type', None)
        if isinstance(_row_rec_type, str) and _row_rec_type:
            _rec_type = _row_rec_type

        params = dict(
            enclosing_folder=enclosing_folder,
            folder_beh=folder_beh,
            folder_img=folder_img,
            fname_img_red=fname_img_red,
            fname_img_grn=fname_img_grn,
            beh_type=beh_type,
            rec_type=_rec_type,
        )
        params.update(extra_load_kwargs)
        exp_dicts[expref] = params

        if verbose:
            _print_status(
                f'{expref}  ->  {enclosing_folder} | beh={folder_beh} | '
                f'img={folder_img}')

    return exp_dicts, skipped


def _build_exp_dicts_from_csv_singlech(csv_path,
                                       data_folder,
                                       fname_img,
                                       beh_type,
                                       rec_type,
                                       extra_load_kwargs,
                                       verbose=True):
    """Single-channel variant of _build_exp_dicts_from_csv.

    Same expref / folder resolution as the dual-colour helper, but
    builds params for ``TwoPRec`` (single ``fname_img``) instead of
    ``TwoPRec_DualColour`` (separate red/green filenames).
    """
    filelist = pd.read_csv(csv_path)
    exp_dicts = {}
    skipped = []

    for i, row in enumerate(filelist.to_dict('records')):
        try:
            expref = str(row['expref'])
        except Exception:
            expref = f'row_{i}'

        try:
            animal, date, folder_beh = _parse_expref(expref)
            enclosing_folder = os.path.join(data_folder, animal, date)
            folder_img = _resolve_img_folder(
                enclosing_folder, date, row['dual2p_folder'])
        except Exception as e:
            if verbose:
                _print_status(f'SKIP {expref}: {e}', symbol='x')
            skipped.append((expref, str(e)))
            continue

        _rec_type = rec_type
        _row_rec_type = row.get('rec_type', None)
        if isinstance(_row_rec_type, str) and _row_rec_type:
            _rec_type = _row_rec_type

        params = dict(
            enclosing_folder=enclosing_folder,
            folder_beh=folder_beh,
            folder_img=folder_img,
            fname_img=fname_img,
            beh_type=beh_type,
            rec_type=_rec_type,
        )
        params.update(extra_load_kwargs)
        exp_dicts[expref] = params

        if verbose:
            _print_status(
                f'{expref}  ->  {enclosing_folder} | beh={folder_beh} | '
                f'img={folder_img}')

    return exp_dicts, skipped


# Main dual-colour bulk function
# -------------------------------------------------------

def bulk_run_twop_correction(csv_path,
                             data_folder='/Volumes/T7/5HTCtx',
                             fname_img_red='compiled_Ch1.tif',
                             fname_img_grn='compiled_Ch2.tif',
                             beh_type='visual_pavlov',
                             rec_type='trig_rew',
                             load_kwargs=None,
                             plt_qc_kwargs=None,
                             plt_show=False):
    """Bulk-load dual-colour 2p recordings and run plt_qc on each.

    Thin wrapper around ``batch_run``: parses each expref in the csv,
    infers the enclosing / behaviour / imaging folders, builds a
    ``batch_run``-style ``exp_dicts``, then dispatches to
    ``batch_run(steps=[('plt_qc', plt_qc_kwargs)], rec_class=
    TwoPRec_DualColour, ...)``. All per-recording error handling
    (load failures, plt_qc failures) is delegated to ``batch_run``.

    Parameters
    ----------
    csv_path : str
        Path to a filelist .csv with columns 'expref' and
        'dual2p_folder' (and optionally 'rec_type').
    data_folder : str
        Root data folder. Each expref ANIMAL_DATE_BEH is expected at
        ``{data_folder}/ANIMAL/DATE/`` with a 'TwoP' subdirectory and
        a behaviour subfolder named BEH.
        Default '/Volumes/T7/5HTCtx'.
    fname_img_red : str
        Red-channel tif filename inside the imaging folder.
        Default 'compiled_Ch1.tif'.
    fname_img_grn : str
        Green-channel tif filename inside the imaging folder.
        Default 'compiled_Ch2.tif'.
    beh_type : str or None
        Forwarded to TwoPRec_DualColour. Default 'visual_pavlov'.
    rec_type : str
        Forwarded to TwoPRec_DualColour when the csv does not provide
        a per-row 'rec_type'. Default 'trig_rew'.
    load_kwargs : dict or None
        Extra keyword arguments forwarded to
        ``TwoPRec_DualColour.__init__`` for every recording (after the
        inferred folder paths). Default None.
    plt_qc_kwargs : dict or None
        Keyword arguments forwarded to ``exp.plt_qc``. ``plt_show`` is
        injected from the top-level argument unless already present.
        Default None (uses plt_qc defaults).
    plt_show : bool
        Forwarded to ``exp.plt_qc(plt_show=...)``. Default False so
        that figure creation does not block the script.

    Returns
    -------
    dict
        ``{expref: {'exp': <TwoPRec_DualColour or None>,
                    'error': <str or None>}}`` -- the dict returned by
        ``batch_run``. Recordings whose folders could not be resolved
        from the csv are not included (they appear in stdout but not
        in the returned dict).
    """
    if load_kwargs is None:
        load_kwargs = {}
    if plt_qc_kwargs is None:
        plt_qc_kwargs = {}
    # User-supplied plt_qc_kwargs takes precedence over the top-level
    # plt_show argument.
    plt_qc_kwargs = {'plt_show': plt_show, **plt_qc_kwargs}

    _print_banner('BULK 2P CORRECTION')
    print(f'  csv          : {csv_path}')
    print(f'  data_folder  : {data_folder}', flush=True)

    t_start = time.time()
    exp_dicts, skipped = _build_exp_dicts_from_csv(
        csv_path,
        data_folder=data_folder,
        fname_img_red=fname_img_red,
        fname_img_grn=fname_img_grn,
        beh_type=beh_type,
        rec_type=rec_type,
        extra_load_kwargs=load_kwargs,
        verbose=True)

    print(f'  resolved     : {len(exp_dicts)} recordings'
          + (f' ({len(skipped)} skipped)' if skipped else ''),
          flush=True)

    if not exp_dicts:
        _print_banner('BULK 2P CORRECTION COMPLETE')
        print('  no recordings to run.', flush=True)
        return {}

    results = batch_run(
        exp_dicts,
        steps=[('plt_qc', plt_qc_kwargs)],
        rec_class=TwoPRec_DualColour)

    _print_banner('BULK 2P CORRECTION COMPLETE')
    print(f'  total elapsed : {_fmt_time(time.time() - t_start)}',
          flush=True)
    return results


# Single-channel hemisphere-control bulk function
# -------------------------------------------------------

def bulk_run_twop_hemispherectrl(csv_path,
                                 data_folder='/Volumes/T7/5HTCtx',
                                 fname_img_red='compiled_Ch1.tif',
                                 beh_type='visual_pavlov',
                                 rec_type='trig_rew',
                                 load_kwargs=None,
                                 plt_qc_kwargs=None,
                                 plt_show=False):
    """Bulk-load single-channel (red) 2p recordings and run plt_qc with
    left/right hemisphere extraction on each.

    Mirrors ``bulk_run_twop_correction`` but for single-channel
    recordings (Ch1 / red only). Each recording is loaded as a
    ``TwoPRec`` (not ``TwoPRec_DualColour``); ``plt_qc`` is run with
    ``split_lr=True`` so whole-frame F is extracted separately for the
    top ('right') and bottom ('left') halves of each frame, providing
    a within-recording hemisphere control. Signal correction is
    disabled (single-channel, no isosbestic to regress out).

    Parameters
    ----------
    csv_path : str
        Path to a filelist .csv with columns 'expref' and
        'dual2p_folder' (and optionally 'rec_type'). Same format as
        ``bulk_run_twop_correction``.
    data_folder : str
        Root data folder. Each expref ANIMAL_DATE_BEH is expected at
        ``{data_folder}/ANIMAL/DATE/`` with a 'TwoP' subdirectory and
        a behaviour subfolder named BEH.
        Default '/Volumes/T7/5HTCtx'.
    fname_img_red : str
        Red-channel (Ch1) tif filename inside the imaging folder.
        Default 'compiled_Ch1.tif'.
    beh_type : str or None
        Forwarded to TwoPRec. Default 'visual_pavlov'.
    rec_type : str
        Forwarded to TwoPRec when the csv does not provide a per-row
        'rec_type'. Default 'trig_rew'.
    load_kwargs : dict or None
        Extra keyword arguments forwarded to ``TwoPRec.__init__`` for
        every recording (after the inferred folder paths). Default None.
    plt_qc_kwargs : dict or None
        Keyword arguments forwarded to ``exp.plt_qc``. ``split_lr=True``
        and ``correct_signal=False`` are injected unless already
        present; ``plt_show`` is injected from the top-level argument
        unless already present. Default None.
    plt_show : bool
        Forwarded to ``exp.plt_qc(plt_show=...)``. Default False.

    Returns
    -------
    dict
        ``{expref: {'exp': <TwoPRec or None>,
                    'error': <str or None>}}`` -- the dict returned by
        ``batch_run``.
    """
    if load_kwargs is None:
        load_kwargs = {}
    if plt_qc_kwargs is None:
        plt_qc_kwargs = {}
    # Inject hemisphere-control defaults; caller can override.
    plt_qc_kwargs = {
        'split_lr': True,
        'correct_signal': False,
        'plt_show': plt_show,
        **plt_qc_kwargs}

    _print_banner('BULK 2P HEMISPHERE CONTROL')
    print(f'  csv          : {csv_path}')
    print(f'  data_folder  : {data_folder}', flush=True)

    t_start = time.time()
    exp_dicts, skipped = _build_exp_dicts_from_csv_singlech(
        csv_path,
        data_folder=data_folder,
        fname_img=fname_img_red,
        beh_type=beh_type,
        rec_type=rec_type,
        extra_load_kwargs=load_kwargs,
        verbose=True)

    print(f'  resolved     : {len(exp_dicts)} recordings'
          + (f' ({len(skipped)} skipped)' if skipped else ''),
          flush=True)

    if not exp_dicts:
        _print_banner('BULK 2P HEMISPHERE CONTROL COMPLETE')
        print('  no recordings to run.', flush=True)
        return {}

    results = batch_run(
        exp_dicts,
        steps=[('plt_qc', plt_qc_kwargs)],
        rec_class=TwoPRec)

    _print_banner('BULK 2P HEMISPHERE CONTROL COMPLETE')
    print(f'  total elapsed : {_fmt_time(time.time() - t_start)}',
          flush=True)
    return results


# ---------------------------------------------------------------------------
# Cross-recording summary plot (plt_grab_ctrl_dualcolour)
# ---------------------------------------------------------------------------

# Filename pattern for _plt_qc_response_mean outputs:
#   {animal}_{date}_{beh}_qc_response_mean{_ch=...}{_corrsig=...}{_dffsig}.npy
_NPY_PATTERN_RE = re.compile(
    r'^(?P<animal>[^_]+)_(?P<date>[^_]+)_(?P<beh>[^_]+)'
    r'_qc_response_mean(?P<suffix>.*)\.npy$')


def _find_latest_response_mean_npy(figs_dir, beh_folder=None,
                                   corrsig_suffix=None):
    """Return the most-recently-modified _qc_response_mean*.npy in figs_dir.

    Parameters
    ----------
    figs_dir : str
        Path to a recording's data_mbl/ folder (where the
        _qc_response_mean*.npy summaries are written).
    beh_folder : str or None
        If given, only consider files whose embedded behaviour folder
        matches (so multi-beh enclosing folders pick the right file).
    corrsig_suffix : str or None
        If given, only consider files whose filename suffix contains
        this fragment -- i.e. the '_corrsig=..._method=..._detrend=...'
        string built by ``twop_qc.corrsig_suffix_from_kwargs``. Lets a
        caller target one run out of a correction sweep rather than
        whichever happened to be written last. None (default) keeps
        every candidate.

    Returns
    -------
    fpath : str or None
    suffix : str or None
        The suffix fragment of the chosen filename (the part after
        '_qc_response_mean' and before '.npy'). Used to verify that
        kwargs are consistent across recordings.
    all_suffixes : list of str
        Every suffix seen before the corrsig_suffix filter was applied,
        so a caller can report what WAS available when the filter
        matched nothing.
    """
    if not os.path.isdir(figs_dir):
        return None, None, []

    candidates = []
    all_suffixes = []
    for entry in os.listdir(figs_dir):
        m = _NPY_PATTERN_RE.match(entry)
        if m is None:
            continue
        if beh_folder is not None and m.group('beh') != str(beh_folder):
            continue
        _sfx = m.group('suffix')
        all_suffixes.append(_sfx)
        if corrsig_suffix is not None and corrsig_suffix not in _sfx:
            continue
        fpath = os.path.join(figs_dir, entry)
        candidates.append((os.path.getmtime(fpath), fpath, _sfx))

    if not candidates:
        return None, None, all_suffixes

    candidates.sort(reverse=True)
    _, fpath, suffix = candidates[0]
    return fpath, suffix, all_suffixes


# Per-cell trial-averaged donut-ring traces, written by
# TwoPRec._save_cell_trialavg_traces_to_disk as
# '{fname_img_stem}_corr[_<tag>]_celltraces_trialavg.npy' in data_mbl/.
# Only correct_signal(method='pixel_spatial_subtr', segmentation_type=
# 'cells') emits one, and that path never adds a kwarg tag -- so unlike
# the response_mean files these carry no _corrsig= fragment and cannot
# be filtered by correct_signal_kwargs.
_CELLTRACES_RE = re.compile(r'_corr(?:_.*)?_celltraces_trialavg\.npy$')


def _find_celltraces_npy(data_dir, prefix=None):
    """Return the most-recently-modified *_celltraces_trialavg.npy path.

    Files written by current code are prefixed
    ``{animal}_{date}_{beh}_`` (``TwoPRec._rec_file_prefix``) because
    data_mbl/ is shared by every behaviour session of a date. Files
    written before that change carry no prefix and so cannot be
    attributed to a session — for a date with several sessions there is
    only one such file, holding whichever correction ran last.

    A prefixed match therefore always wins; an unprefixed file is used
    only as a fallback, and flagged as legacy so the caller can warn.

    Parameters
    ----------
    data_dir : str
        A recording's data_mbl/ folder.
    prefix : str or None
        The recording's ``{animal}_{date}_{beh}_`` prefix. None keeps
        every candidate and always reports legacy=False.

    Returns
    -------
    fpath : str or None
    legacy : bool
        True when the returned file predates per-session naming, so it
        may belong to a different behaviour session of the same date.
    """
    if not os.path.isdir(data_dir):
        return None, False

    tagged, legacy = [], []
    for entry in os.listdir(data_dir):
        if _CELLTRACES_RE.search(entry) is None:
            continue
        fpath = os.path.join(data_dir, entry)
        _item = (os.path.getmtime(fpath), fpath)
        if prefix is not None and not entry.startswith(prefix):
            legacy.append(_item)
        else:
            tagged.append(_item)

    for _pool, _is_legacy in ((tagged, False), (legacy, True)):
        if _pool:
            _pool.sort(reverse=True)
            return _pool[0][1], _is_legacy
    return None, False


def _finite_minmax(*arrays):
    """Min/max over the finite values of one or more arrays.

    Parameters
    ----------
    *arrays : np.ndarray
        Any number of arrays (any shape); NaN/inf entries are ignored.

    Returns
    -------
    lo, hi : float
        (-1.0, 1.0) when no finite values are present anywhere, so
        callers get a valid (non-degenerate) range rather than NaN.
    """
    _vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel()
                            for a in arrays]) if arrays else np.array([])
    _vals = _vals[np.isfinite(_vals)]
    if _vals.size == 0:
        return -1.0, 1.0
    lo, hi = float(np.min(_vals)), float(np.max(_vals))
    if hi <= lo:
        lo, hi = lo - 1.0, hi + 1.0
    return lo, hi


def _symmetric_vlim(*arrays, vlim=None, clip_pct=5.0):
    """Symmetric (vmin, vmax) for a heatmap: explicit override or data-driven.

    Parameters
    ----------
    *arrays : np.ndarray
        Data the heatmap will display; used to derive the range when
        `vlim` is None.
    vlim : float or None
        If given, used directly as the symmetric half-range (vmin =
        -vlim, vmax = +vlim). If None (default), the half-range is
        derived from `arrays` via `clip_pct` below.
    clip_pct : float
        Only used when `vlim` is None. The half-range is set to the
        ``(100 - clip_pct)``-th percentile of |values| across `arrays`,
        so roughly `clip_pct`% of values fall outside [vmin, vmax] and
        get clamped by imshow -- trading a few saturated pixels for a
        colour scale that isn't dominated by rare extreme outliers (the
        true max can badly compress everything else). Default 5.0.

    Returns
    -------
    vmin, vmax : float
    """
    if vlim is not None:
        _v = float(vlim)
        return -_v, _v
    _vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel()
                            for a in arrays]) if arrays else np.array([])
    _vals = _vals[np.isfinite(_vals)]
    _vmag = (float(np.percentile(np.abs(_vals), 100.0 - clip_pct))
             if _vals.size else 1.0)
    if not (np.isfinite(_vmag) and _vmag > 0):
        _vmag = 1.0
    return -_vmag, _vmag


def _resolve_corr_key(summary):
    """Return the corrected-channel key from a response_mean summary, or None.
    """
    key = summary.get('corr_key')
    if isinstance(key, str) and key in summary.get('whole_frame', {}):
        return key
    # Fallback: search whole_frame for a *_corr key
    for k in summary.get('whole_frame', {}):
        if k.endswith('_corr'):
            return k
    return None


def _resolve_stat_key(stats_group, base):
    """Return the '<base>_vs_<control>' key present in a stats sub-dict.

    plt_qc names its correlation stats after the two channels actually
    compared, so a green-functional recording writes 'grn_vs_red' where a
    red-functional one writes 'red_vs_grn'. Tries both control channels
    and returns None when neither is present.

    Parameters
    ----------
    stats_group : dict
        e.g. summary['stats']['whole_frame_pearson'].
    base : str
        Channel (or corrected-channel) key on the left of '_vs_'.

    Returns
    -------
    key : str or None
    """
    for _ctrl in ('grn', 'red'):
        _k = f'{base}_vs_{_ctrl}'
        if _k in stats_group:
            return _k
    return None


def _interp_to_common(t_src, y_src, t_common):
    """Linear interpolation of y_src onto t_common with NaN fill outside."""
    t_src = np.asarray(t_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    if t_src.size == 0 or y_src.size == 0:
        return np.full_like(t_common, np.nan, dtype=np.float64)
    if y_src.ndim == 1:
        if y_src.size != t_src.size:
            return np.full_like(t_common, np.nan, dtype=np.float64)
    else:
        if y_src.shape[-1] != t_src.size:
            return np.full(y_src.shape[:-1] + t_common.shape, np.nan,
                           dtype=np.float64)
    f = interp1d(t_src, y_src, kind='linear', axis=-1,
                 bounds_error=False, fill_value=np.nan)
    return f(t_common)


def _paired_dot_plot(ax, x_left, x_right, labels=('A', 'B'),
                     colour_left='#444', colour_right='#444',
                     jitter=0.0):
    """Two-column paired dot plot with connecting lines and a paired t-test.

    Parameters
    ----------
    ax : matplotlib axes
    x_left, x_right : array-like
        Paired observations (same length).
    labels : tuple of str
        Column labels.
    colour_left, colour_right : str
        Marker edge colours.
    jitter : float
        Horizontal jitter to add (rng-based).

    Returns
    -------
    t_stat, p_val : float, float
        Paired-T test result (NaN, NaN if fewer than 2 finite pairs).
    """
    x_left = np.asarray(x_left, dtype=np.float64)
    x_right = np.asarray(x_right, dtype=np.float64)
    n = min(x_left.size, x_right.size)
    x_left, x_right = x_left[:n], x_right[:n]
    finite = np.isfinite(x_left) & np.isfinite(x_right)

    rng = np.random.default_rng(0)
    jx_l = rng.uniform(-jitter, jitter, size=n) if jitter else np.zeros(n)
    jx_r = rng.uniform(-jitter, jitter, size=n) if jitter else np.zeros(n)

    for i in range(n):
        if not finite[i]:
            continue
        ax.plot([0 + jx_l[i], 1 + jx_r[i]],
                [x_left[i], x_right[i]],
                color='#888', lw=0.6, alpha=0.8, zorder=1)
    ax.scatter(np.full(n, 0) + jx_l, x_left,
               facecolors='none', edgecolors=colour_left,
               s=36, lw=1.2, zorder=2)
    ax.scatter(np.full(n, 1) + jx_r, x_right,
               facecolors='none', edgecolors=colour_right,
               s=36, lw=1.2, zorder=2)

    if finite.sum() >= 2:
        t_stat, p_val = sp_stats.ttest_rel(
            x_left[finite], x_right[finite])
        t_stat, p_val = float(t_stat), float(p_val)
    else:
        t_stat, p_val = np.nan, np.nan

    ax.set_xticks([0, 1])
    _wrapped = [str(_l).replace(' vs ', '\nvs\n') for _l in labels]
    ax.set_xticklabels(_wrapped, fontsize=6)
    ax.set_xlim(-0.5, 1.5)
    for _side in ('top', 'right'):
        ax.spines[_side].set_visible(False)
    ax.set_title(f'paired t = {t_stat:.2f}\n'
                 f'p = {p_val:.2g}  (n = {int(finite.sum())})',
                 fontsize=6)
    return t_stat, p_val


_L23_TAGS = {'t001', 't-001', '001', '1'}
_L1_TAGS = {'t002', 't-002', '002', '2',
            't003', 't-003', '003', '3'}


def _iter_csv_recordings(csv_path, data_folder, plt_layer='all',
                         verbose=True):
    """Yield (expref, data_dir) for every csv row passing the layer filter.

    Shared front-end of the per-recording collectors: reads the csv,
    applies the ``plt_layer`` ``dual2p_folder`` filter, parses each
    expref and resolves its ``{data_folder}/{animal}/{date}/data_mbl/``
    path. Rows that cannot be parsed are reported and skipped.

    Parameters
    ----------
    csv_path : str
        Path to a filelist csv with at least an 'expref' column.
    data_folder : str
        Root data folder.
    plt_layer : str
        'all', '2/3' (t001 only) or '1' (t002 / t003 only).
    verbose : bool

    Yields
    ------
    expref : str
    folder_beh : str
    data_dir : str
    """
    filelist = pd.read_csv(csv_path)

    for i, row in enumerate(filelist.to_dict('records')):
        try:
            expref = str(row['expref'])
        except Exception:
            expref = f'row_{i}'

        # plt_layer: filter by canonical dual2p_folder tag.
        #   'all'  -> include every row
        #   '2/3'  -> only t001 (layer 2/3)
        #   '1'    -> only t002 / t003 (layer 1)
        if plt_layer != 'all':
            _dual_tag = str(row.get('dual2p_folder', '')).strip().lower()
            if plt_layer == '2/3':
                _allowed = _L23_TAGS
            elif plt_layer == '1':
                _allowed = _L1_TAGS
            else:
                raise ValueError(
                    f"plt_layer must be 'all', '2/3', or '1'; "
                    f'got {plt_layer!r}')
            if _dual_tag and _dual_tag not in _allowed:
                if verbose:
                    print(f'  - {expref}: skipped (plt_layer='
                          f'{plt_layer!r}, dual2p_folder='
                          f'{_dual_tag!r})')
                continue

        try:
            animal, date, folder_beh = _parse_expref(expref)
        except Exception as e:
            if verbose:
                print(f'  x {expref}: could not parse expref ({e})')
            continue

        yield expref, folder_beh, os.path.join(
            data_folder, animal, date, 'data_mbl')


def _collect_celltraces_records(csv_path, data_folder, verbose=True,
                                plt_layer='all', keep_exprefs=None):
    """Read the csv and load the *_celltraces_trialavg.npy per recording.

    Companion to ``_collect_response_mean_records`` for the per-cell
    donut-ring traces written by
    ``TwoPRec._save_cell_trialavg_traces_to_disk``.

    Two asymmetries with the response_mean files have to be handled
    here, both stemming from the fact that the cell-trace filename is
    derived from the imaging stem alone
    (``compiled_Ch1_corr_celltraces_trialavg.npy``) and carries no
    behaviour-folder tag, unlike ``{animal}_{date}_{beh}_qc_response
    _mean...npy``:

    - A date with several behaviour sessions has ONE cell-trace file in
      its data_mbl/, shared by every one of those csv rows (the last
      correction run for that date wins). Loading it once per row would
      count those cells two or more times in the pooled matrix, so
      records are de-duplicated on the resolved path.
    - A recording can have cell traces but no matching response_mean
      file, which would let the cells row describe a wider cohort than
      the rest of the figure. ``keep_exprefs`` restricts loading to the
      recordings that made it into the response_mean records.

    Parameters
    ----------
    keep_exprefs : set of str or None
        If given, only these exprefs are considered. None loads every
        row passing the plt_layer filter.

    Returns
    -------
    records : list of SimpleNamespace
        One entry per unique cell-trace file, with .expref (the first
        row that resolved to it), .fpath, .cells (the loaded dict).
    """
    records = []
    seen = {}
    n_legacy = 0

    for expref, _folder_beh, data_dir in _iter_csv_recordings(
            csv_path, data_folder, plt_layer=plt_layer, verbose=verbose):
        if keep_exprefs is not None and expref not in keep_exprefs:
            if verbose:
                print(f'  - {expref}: skipped (no matching '
                      f'_qc_response_mean*.npy)')
            continue
        try:
            fpath, is_legacy = _find_celltraces_npy(
                data_dir, prefix=f'{expref}_')
        except Exception as e:
            if verbose:
                print(f'  x {expref}: error scanning {data_dir} ({e})')
            continue
        if fpath is None:
            if verbose:
                print(f'  x {expref}: no *_celltraces_trialavg.npy in '
                      f'{data_dir}')
            continue
        if fpath in seen:
            if verbose:
                print(f'  - {expref}: shares '
                      f'{os.path.basename(fpath)} with {seen[fpath]} '
                      f'(legacy per-date file, no session tag); '
                      f'counted once')
            continue
        if is_legacy:
            n_legacy += 1

        try:
            cells = np.load(fpath, allow_pickle=True).item()
        except Exception as e:
            if verbose:
                print(f'  x {expref}: failed to load {fpath} ({e})')
            continue

        seen[fpath] = expref
        records.append(SimpleNamespace(
            expref=expref, fpath=fpath, cells=cells,
            legacy=bool(is_legacy)))
        if verbose:
            _n = np.asarray(
                cells.get('cell_traces_trialavg', [])).shape[0] \
                if cells.get('cell_traces_trialavg') is not None else 0
            print(f'  + {expref}: {os.path.basename(fpath)} '
                  f'({_n} cells)'
                  + ('  [legacy per-date file]' if is_legacy else ''))

    if verbose and n_legacy:
        print(f'  ! {n_legacy} of {len(records)} cell-trace files predate '
              f'per-session naming, so they cannot be attributed to a '
              f'behaviour session (a date with two sessions has one such '
              f'file, from whichever correction ran last). Re-run '
              f'correct_signal / plt_qc(save_tif=True) on those '
              f'recordings to write per-session files.', flush=True)

    return records


def _collect_response_mean_records(csv_path, data_folder, verbose=True,
                                   plt_layer='all', corrsig_suffix=None):
    """Read the csv and load the latest response_mean .npy per row.

    Parameters
    ----------
    corrsig_suffix : str or None
        Forwarded to ``_find_latest_response_mean_npy``; restricts each
        row to the saved file matching one correction parameterisation.

    Returns
    -------
    records : list of SimpleNamespace
        One entry per successfully-loaded recording, with .expref,
        .fpath, .suffix, .summary.
    suffix_counts : dict
        {suffix: count} of all observed filename suffixes; used to
        check kwarg consistency across recordings.
    """
    records = []
    suffix_counts = {}

    for expref, folder_beh, data_dir in _iter_csv_recordings(
            csv_path, data_folder, plt_layer=plt_layer, verbose=verbose):
        try:
            fpath, suffix, all_sfx = _find_latest_response_mean_npy(
                data_dir, beh_folder=folder_beh,
                corrsig_suffix=corrsig_suffix)
        except Exception as e:
            if verbose:
                print(f'  x {expref}: error scanning {data_dir} ({e})')
            continue
        if fpath is None:
            if verbose:
                if all_sfx and corrsig_suffix is not None:
                    # Files exist but none carry the requested correction
                    # tag -- show what IS there so a typo in
                    # correct_signal_kwargs is diagnosable.
                    print(f'  x {expref}: no _qc_response_mean*.npy '
                          f'matching {corrsig_suffix!r}; available:')
                    for _s in sorted(set(all_sfx)):
                        print(f'        {_s!r}')
                else:
                    print(f'  x {expref}: no _qc_response_mean*.npy in '
                          f'{data_dir}')
            continue

        try:
            summary = np.load(fpath, allow_pickle=True).item()
        except Exception as e:
            if verbose:
                print(f'  x {expref}: failed to load {fpath} ({e})')
            continue

        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        records.append(SimpleNamespace(
            expref=expref, fpath=fpath, suffix=suffix, summary=summary))
        if verbose:
            print(f'  + {expref}: {os.path.basename(fpath)}')

    return records, suffix_counts


def _plt_summary_core(csv_path,
            data_folder='/Volumes/T7/5HTCtx',
            sort_by='red_corr',
            sector_corr_metric='pearson',
            t_pre=2.0,
            t_post=6.0,
            dt=0.05,
            t_rew=2.0,
            sector_vlim=None,
            norm_resp_hist=True,
            plt_layer='all',
            figsize=(6.86, 5),
            save_path=None,
            plt_show=True,
            correct_signal=True,
            correct_signal_kwargs=None,
            detrend_sigs=False,
            plt_cells=False,
            drop_flat_cells=False,
            dff_min=None,
            dff_max=None,
            cell_vlim=None,
            fname_stem='fig_dualcolour_grab',
            title='plt_grab_ctrl_dualcolour'):
    """Across-recording summary plot of QC response_mean outputs.

    Shared implementation behind ``plt_grab_ctrl_dualcolour`` (the
    original 6-panel figure) and ``plt_summary`` (which adds correction
    targeting, a per-cell row and explicit dF/F limits). See those two
    for the user-facing documentation; the extra parameters here are:

    correct_signal, correct_signal_kwargs, detrend_sigs
        Rebuild the '_corrsig=...' filename fragment via
        ``twop_qc.corrsig_suffix_from_kwargs`` and use it to pick which
        saved `_qc_response_mean*.npy` to load per recording. All None /
        default when called from ``plt_grab_ctrl_dualcolour``, which
        then falls back to newest-mtime selection.
    plt_cells : bool
        Add the pooled per-cell heatmap row (panel 7).
    drop_flat_cells : bool
        Exclude cells whose saved trace has zero variance.
    dff_min, dff_max : float or None
        Unused by this function; accepted only so existing callers that
        still pass them don't error. Panel 1's y-axes and panels 2/7's
        heatmap colour scales are now data-driven (panel 1) or governed
        by ``sector_vlim`` / ``cell_vlim`` (panels 2 / 7) -- see those
        two parameters and ``plt_summary``'s docstring.
    cell_vlim : float or None
        Symmetric colour-scale limit for the panel-7 per-cell heatmap
        (vmin=-cell_vlim, vmax=+cell_vlim), mirroring ``sector_vlim``.
        None (default) derives it from the actual pooled cell data as
        the 95th percentile of |dF/F| (clipping ~5% of values rather
        than stretching to the true max, which a few outlier cells can
        dominate); pass a float to pin the same numeric range across
        separately-plotted cohorts for comparison.
    fname_stem, title : str
        Saved-filename stem and figure suptitle prefix.
    """
    if sector_corr_metric not in ('pearson', 'spearman'):
        raise ValueError(
            f'sector_corr_metric must be "pearson" or "spearman", got '
            f'{sector_corr_metric!r}')

    # Rebuild the filename fragment plt_qc stamped onto its saves for
    # this correction parameterisation, so one run of a sweep can be
    # targeted rather than whichever file was written last. None means
    # "take the newest", the historical behaviour.
    _cs_method = (correct_signal_kwargs or {}).get('method', 'full_regress')
    if correct_signal_kwargs is None:
        _corrsig_suffix = None
    else:
        _corrsig_suffix = corrsig_suffix_from_kwargs(
            correct_signal_kwargs=correct_signal_kwargs,
            correct_signal=correct_signal,
            detrend_sigs=detrend_sigs)

    _print_banner('PLT_ALL')
    print(f'  csv         : {csv_path}')
    print(f'  data_folder : {data_folder}')
    print(f'  sort_by     : {sort_by}')
    print(f'  sec metric  : {sector_corr_metric}')
    print(f'  plt_layer   : {plt_layer!r}')
    if _corrsig_suffix is not None:
        print(f'  corrsig     : {_corrsig_suffix!r}')
    print('', end='', flush=True)

    records, suffix_counts = _collect_response_mean_records(
        csv_path, data_folder, verbose=True, plt_layer=plt_layer,
        corrsig_suffix=_corrsig_suffix)
    if not records:
        raise RuntimeError(
            'No _qc_response_mean*.npy files were loaded; cannot plot.'
            + ('' if _corrsig_suffix is None else
               f' (filtered on corrsig suffix {_corrsig_suffix!r} -- '
               f'check correct_signal_kwargs against the available '
               f'suffixes listed above)'))

    # Warn (but proceed) if more than one filename suffix is present.
    if len(suffix_counts) > 1:
        print('  ! WARNING: response_mean .npy files have inconsistent '
              'kwarg suffixes:')
        for sfx, n in sorted(suffix_counts.items(), key=lambda kv: -kv[1]):
            print(f'      ({n:3d}) {sfx!r}')
        # Keep only the most common suffix
        _dom_sfx = max(suffix_counts, key=lambda k: suffix_counts[k])
        print(f'  ! using the dominant suffix: {_dom_sfx!r}')
        records = [r for r in records if r.suffix == _dom_sfx]
    else:
        print(f'  suffix      : {next(iter(suffix_counts))!r}')

    print(f'  n loaded    : {len(records)}', flush=True)

    # Common time grid
    t_common = np.arange(-t_pre, t_post + dt / 2, dt)
    n_t = t_common.size

    channels = ('red', 'grn', 'red_corr')
    if sort_by not in channels:
        raise ValueError(
            f'sort_by must be one of {channels}, got {sort_by!r}')

    # ----------------------------------------------------------
    # Pull per-recording traces onto t_common (whole-frame + sectors)
    # ----------------------------------------------------------
    wf = {ch: [] for ch in channels}
    sec_per_rec = {ch: [] for ch in channels}
    # Per-recording scalars for panels 3 and 4
    pear_rg_wf = []
    pear_cg_wf = []
    sec_corr_rg = []
    sec_corr_cg = []

    _stat_key = ('sector_pearson' if sector_corr_metric == 'pearson'
                 else 'sector_spearman')
    _stat_val_key = 'r' if sector_corr_metric == 'pearson' else 'rho'

    for rec in records:
        summary = rec.summary
        t_rel = np.asarray(summary.get('t_rel', []), dtype=np.float64)
        t_rel_sec = np.asarray(
            summary.get('t_rel_sec', t_rel), dtype=np.float64)
        wf_d = summary.get('whole_frame', {})
        sec_d = summary.get('sectors', {})
        corr_key = _resolve_corr_key(summary)

        for ch in channels:
            src_key = corr_key if (ch == 'red_corr' and corr_key) else ch
            entry = wf_d.get(src_key)
            if entry is None or 'mean' not in entry or entry['mean'] is None:
                wf[ch].append(np.full(n_t, np.nan))
            else:
                wf[ch].append(_interp_to_common(
                    t_rel, np.asarray(entry['mean']), t_common))

            sec_arr = sec_d.get(src_key)
            if sec_arr is None:
                sec_per_rec[ch].append(np.full((0, n_t), np.nan))
            else:
                sec_arr = np.asarray(sec_arr, dtype=np.float64)
                if sec_arr.ndim != 2 or sec_arr.shape[1] != t_rel_sec.size:
                    sec_per_rec[ch].append(np.full((0, n_t), np.nan))
                else:
                    sec_per_rec[ch].append(
                        _interp_to_common(t_rel_sec, sec_arr, t_common))

        # Panel 3 scalars: whole-frame Pearson from saved stats. The
        # signal-channel key follows the recording's functional channel
        # ('red_vs_grn' or 'grn_vs_red'), so resolve rather than assume.
        stats = summary.get('stats', {})
        sig_ch = summary.get('signal_ch', 'red')
        wf_p = stats.get('whole_frame_pearson', {})
        rg = wf_p.get(_resolve_stat_key(wf_p, sig_ch), {})
        pear_rg_wf.append(float(rg.get('r', np.nan)))
        if corr_key is not None:
            cg = wf_p.get(_resolve_stat_key(wf_p, corr_key), {})
            pear_cg_wf.append(float(cg.get('r', np.nan)))
        else:
            pear_cg_wf.append(np.nan)

        # Panel 4 scalars: per-sector pearson/spearman from saved stats
        sec_stats = stats.get(_stat_key, {})
        rg_sec = sec_stats.get(_resolve_stat_key(sec_stats, sig_ch), {})
        sec_corr_rg.append(float(rg_sec.get(_stat_val_key, np.nan)))
        if corr_key is not None:
            cg_sec = sec_stats.get(
                _resolve_stat_key(sec_stats, corr_key), {})
            sec_corr_cg.append(float(cg_sec.get(_stat_val_key, np.nan)))
        else:
            sec_corr_cg.append(np.nan)

    wf = {ch: np.asarray(v, dtype=np.float64) for ch, v in wf.items()}

    # Resolve y-axis label (z-score vs dF/F) from the saved summary
    # 'mode' field; default to a generic label if not present.
    _modes = [r.summary.get('mode') for r in records
              if r.summary.get('mode') is not None]
    _mode_dom = max(set(_modes), key=_modes.count) if _modes else None
    if _mode_dom == 'dff_sig':
        _wf_ylabel = 'dF/F'
    elif _mode_dom == 'iti_zscore':
        _wf_ylabel = 'z-score'
    else:
        _wf_ylabel = 'response'

    # Reward marker: prefer per-recording median rew_t_rel from the
    # saved summaries; fall back to the t_rew kwarg.
    _rew_ts = [float(r.summary.get('rew_t_rel'))
               for r in records
               if r.summary.get('rew_t_rel') is not None
               and np.isfinite(float(r.summary.get('rew_t_rel')))]
    _t_rew_plot = float(np.median(_rew_ts)) if _rew_ts else float(t_rew)

    # Stack all sectors (within each channel) across recordings into a
    # single (n_total_sectors, n_t) matrix. Concatenation preserves
    # row alignment across channels (sector i in red corresponds to
    # sector i in grn / red_corr for the same recording).
    all_sec = {}
    for ch in channels:
        per_rec = sec_per_rec[ch]
        if per_rec:
            all_sec[ch] = np.concatenate(per_rec, axis=0)
        else:
            all_sec[ch] = np.zeros((0, n_t), dtype=np.float64)

    # Cross-channel row counts must match; if a channel is missing
    # entirely we drop it down to NaN-rows of the right size.
    n_rows_target = max(a.shape[0] for a in all_sec.values())
    for ch in channels:
        if all_sec[ch].shape[0] != n_rows_target:
            _pad = np.full((n_rows_target - all_sec[ch].shape[0], n_t),
                           np.nan, dtype=np.float64)
            all_sec[ch] = np.vstack([all_sec[ch], _pad])

    # Sort by post-stim peak of the requested channel
    _post = t_common >= 0
    _src_for_sort = all_sec[sort_by]
    with np.errstate(invalid='ignore'):
        if np.any(_post):
            _peak = np.nanmax(_src_for_sort[:, _post], axis=1)
        else:
            _peak = np.nanmax(_src_for_sort, axis=1)
    _order = np.argsort(np.where(np.isfinite(_peak),
                                 _peak, -np.inf))[::-1]
    for ch in channels:
        all_sec[ch] = all_sec[ch][_order]

    # ----------------------------------------------------------
    # Per-cell donut-ring traces (optional extra row).
    # Pooled across recordings the same way the sectors are, but from a
    # different artefact: *_celltraces_trialavg.npy, written only by
    # correct_signal(method='pixel_spatial_subtr',
    # segmentation_type='cells'). Those files carry no _corrsig= tag, so
    # they cannot be filtered by correct_signal_kwargs -- a request for
    # any other method means no cells exist to plot.
    # ----------------------------------------------------------
    cells = np.zeros((0, n_t), dtype=np.float64)
    _cell_records = []
    _n_flat = 0
    if plt_cells:
        if correct_signal_kwargs is not None \
                and _cs_method != 'pixel_spatial_subtr':
            print(f'  ! plt_cells: correct_signal_kwargs asks for '
                  f'method={_cs_method!r}, but per-cell traces are only '
                  f'written by "pixel_spatial_subtr"; skipping the '
                  f'cells row.', flush=True)
        else:
            print('  loading per-cell traces:', flush=True)
            # Restricted to the recordings that produced a
            # response_mean record, so every panel of the figure
            # describes the same cohort.
            _cell_records = _collect_celltraces_records(
                csv_path, data_folder, verbose=True, plt_layer=plt_layer,
                keep_exprefs={r.expref for r in records})
            _cells_per_rec = []
            for _crec in _cell_records:
                _d = _crec.cells
                _ct = _d.get('cell_traces_trialavg')
                _tw = _d.get('t_win')
                if _ct is None or _tw is None:
                    continue
                _ct = np.asarray(_ct, dtype=np.float64)
                _tw = np.asarray(_tw, dtype=np.float64)
                if _ct.ndim != 2 or _ct.shape[1] != _tw.size:
                    continue
                # The saved traces are raw dF/F fractions and are NOT
                # baseline-subtracted, unlike the per-recording figure
                # (_plt_qc_response_mean_cells re-derives them through
                # _compute_event_avg, which subtracts the pre-event
                # baseline, then scales to %). Match that convention
                # here so the cohort row is comparable to the
                # single-recording PDFs.
                _pre = _tw < 0
                if np.any(_pre):
                    with warnings.catch_warnings(), \
                            np.errstate(invalid='ignore'):
                        warnings.simplefilter(
                            'ignore', category=RuntimeWarning)
                        _base = np.nanmean(_ct[:, _pre], axis=1)
                    _ct = _ct - _base[:, None]
                # Cells whose saved trace is perfectly flat carry no
                # measurement -- correct_pixel_spatial_subtr emits an
                # all-zero row when a soma's donut ring ends up with no
                # usable pixels. They are not "no response"; averaging
                # them in pulls the pooled mean toward zero and flattens
                # the heatmap. Counted always, dropped only on request.
                with warnings.catch_warnings(), \
                        np.errstate(invalid='ignore'):
                    warnings.simplefilter(
                        'ignore', category=RuntimeWarning)
                    _flat = np.nanstd(_ct, axis=1) == 0
                _n_flat += int(_flat.sum())
                if drop_flat_cells and _flat.any():
                    _ct = _ct[~_flat]
                if _ct.shape[0] == 0:
                    continue
                _cells_per_rec.append(
                    _interp_to_common(_tw, _ct * 100.0, t_common))
            if _cells_per_rec:
                cells = np.concatenate(_cells_per_rec, axis=0)
            print(f'  n cells     : {cells.shape[0]} '
                  f'(from {len(_cells_per_rec)} recordings)', flush=True)
            if _n_flat:
                _n_src = cells.shape[0] + (_n_flat if drop_flat_cells else 0)
                _pct = 100.0 * _n_flat / max(_n_src, 1)
                if drop_flat_cells:
                    print(f'  ! dropped {_n_flat} flat (all-zero) cells '
                          f'of {_n_src} ({_pct:.0f}%) -- failed ring '
                          f'extraction, not a null response.', flush=True)
                else:
                    print(f'  ! WARNING: {_n_flat} of {_n_src} cells '
                          f'({_pct:.0f}%) have a perfectly flat saved '
                          f'trace (failed ring extraction). They are '
                          f'averaged in as zeros, biasing the pooled '
                          f'mean toward 0. Pass drop_flat_cells=True to '
                          f'exclude them.', flush=True)

    # Sort cells descending by mean stim -> reward response, matching
    # _plt_qc_response_mean_cells.
    if cells.shape[0]:
        _c_mask = (t_common >= 0.0) & (t_common <= _t_rew_plot) \
            if _t_rew_plot > 0 else (t_common >= 0.0)
        with warnings.catch_warnings(), np.errstate(invalid='ignore'):
            warnings.simplefilter('ignore', category=RuntimeWarning)
            _c_int = (np.nanmean(cells[:, _c_mask], axis=1)
                      if np.any(_c_mask) else np.nanmean(cells, axis=1))
        cells = cells[np.argsort(-np.nan_to_num(_c_int, nan=-np.inf))]

    _has_cells = bool(plt_cells and cells.shape[0])

    # ----------------------------------------------------------
    # Build the figure
    # ----------------------------------------------------------
    # constrained_layout (rather than tight_layout) so the colorbars
    # added in panel 2 don't clash with the layout engine.
    # hspace/wspace omitted — constrained_layout manages spacing.
    # publication_ml stylesheet sets uniform font sizes; wrap figure
    # construction in a style context so it doesn't leak globally.
    _style_ctx = plt.style.context('publication_ml')
    _style_ctx.__enter__()  # closed below (search: _style_ctx.__exit__)
    fig = plt.figure(figsize=figsize, layout='constrained')
    if _has_cells:
        gs = gridspec.GridSpec(
            4, 6, figure=fig,
            height_ratios=[1.0, 2.0, 1.2, 1.6])
    else:
        gs = gridspec.GridSpec(
            3, 6, figure=fig,
            height_ratios=[1.0, 2.0, 1.2])

    # Panel 1: whole-frame trial-averaged traces (3 columns).
    # y-lim is data-driven, not dff_min/dff_max: red and grn share one
    # range (spanning both channels' actual data, so they stay visually
    # comparable), while red_corr -- on a different scale after
    # correction -- gets its own range from just its own data.
    axes_wf = [fig.add_subplot(gs[0, 2 * i:2 * i + 2])
               for i in range(3)]
    _rg_lo, _rg_hi = _finite_minmax(wf['red'], wf['grn'])
    _rc_lo, _rc_hi = _finite_minmax(wf['red_corr'])
    _wf_ylim = {'red': (_rg_lo, _rg_hi), 'grn': (_rg_lo, _rg_hi),
               'red_corr': (_rc_lo, _rc_hi)}
    for ax, ch in zip(axes_wf, channels):
        col = _CHANNEL_COLOURS[ch]
        traces = wf[ch]
        for row in traces:
            if np.any(np.isfinite(row)):
                ax.plot(t_common, row, color=col, lw=0.5, alpha=0.4)
        with warnings.catch_warnings(), np.errstate(invalid='ignore'):
            # All-NaN columns trigger RuntimeWarning from np.nanmean
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean = np.nanmean(traces, axis=0)
            n_per_t = np.sum(np.isfinite(traces), axis=0)
            sd = np.nanstd(traces, axis=0)
            sem = sd / np.sqrt(np.maximum(n_per_t, 1))
        ax.fill_between(t_common, mean - sem, mean + sem,
                        color=col, alpha=0.25, lw=0)
        ax.plot(t_common, mean, color=col, lw=2, alpha=1.0)
        ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)
        # Reward marker (median rew_t_rel across recordings, or t_rew)
        if (t_common[0] <= _t_rew_plot <= t_common[-1]):
            ax.axvline(_t_rew_plot, color='#1f77b4',
                       lw=0.5, ls='--', alpha=0.7)
        ax.set_title(f'{_CHANNEL_LABELS[ch]}  (n={traces.shape[0]})',
                     color=col)
        ax.set_xlabel('time from stim (s)')
        ax.set_ylabel(_wf_ylabel)
        ax.set_xlim(t_common[0], t_common[-1])
        ax.set_ylim(*_wf_ylim[ch])
        for _side in ('top', 'right'):
            ax.spines[_side].set_visible(False)

    # Panel 2: per-sector heatmaps (3 columns, shared sorting).
    # Symmetric colour scale (vmin=-vlim, vmax=+vlim): sector_vlim=None
    # (default) derives vlim from the actual pooled sector data across
    # all three channels; pass an explicit sector_vlim to pin the same
    # numeric range across separate cohort figures for comparison.
    axes_sec = [fig.add_subplot(gs[1, 2 * i:2 * i + 2])
                for i in range(3)]
    _vmin, _vmax = _symmetric_vlim(
        all_sec['red'], all_sec['grn'], all_sec['red_corr'],
        vlim=sector_vlim)

    _last_im = None
    for ax, ch in zip(axes_sec, channels):
        mat = all_sec[ch]
        if mat.size == 0:
            ax.text(0.5, 0.5, 'no sectors', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        im = ax.imshow(
            mat, aspect='auto', origin='upper',
            extent=[t_common[0], t_common[-1], mat.shape[0], 0],
            cmap='RdBu_r', vmin=_vmin, vmax=_vmax,
            interpolation='nearest')
        _last_im = im
        ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.6)
        # Reward marker (same as panel 1)
        if (t_common[0] <= _t_rew_plot <= t_common[-1]):
            ax.axvline(_t_rew_plot, color='#1f77b4',
                       lw=0.5, ls='--', alpha=0.7)
        ax.set_title(f'{_CHANNEL_LABELS[ch]} sectors',
                     color=_CHANNEL_COLOURS[ch])
        ax.set_xlabel('time from stim (s)')
        ax.set_ylabel(f'sector (sorted by {sort_by} peak)')
    if _last_im is not None:
        fig.colorbar(_last_im, ax=axes_sec[-1],
                     fraction=0.04, pad=0.02)

    # Panel 3: whole-frame Pearson dot plot (red vs grn) / (red_corr vs grn)
    # Narrow column (1/6 of width); enforce a 3:1 (h:w) box aspect.
    ax_wf_dot = fig.add_subplot(gs[2, 0])
    t_wf, p_wf = _paired_dot_plot(
        ax_wf_dot, pear_rg_wf, pear_cg_wf,
        labels=('red vs grn',
                r'red$_\mathrm{corrected}$ vs grn'),
        colour_left=_CHANNEL_COLOURS['red'],
        colour_right=_CHANNEL_COLOURS['red_corr'])
    ax_wf_dot.set_ylabel('whole-frame Pearson r')
    ax_wf_dot.axhline(0, color='#bbb', lw=0.5, ls='--', zorder=0)
    ax_wf_dot.set_box_aspect(3.0)
    ax_wf_dot.spines['bottom'].set_visible(False)
    ax_wf_dot.tick_params(axis='x', bottom=False)

    # Panel 4: sector Pearson / Spearman dot plot (narrow column)
    ax_sec_dot = fig.add_subplot(gs[2, 1])
    t_sec, p_sec = _paired_dot_plot(
        ax_sec_dot, sec_corr_rg, sec_corr_cg,
        labels=('red vs grn',
                r'red$_\mathrm{corrected}$ vs grn'),
        colour_left=_CHANNEL_COLOURS['red'],
        colour_right=_CHANNEL_COLOURS['red_corr'])
    _ylab = ('sector Pearson r' if sector_corr_metric == 'pearson'
             else 'sector Spearman rho')
    ax_sec_dot.set_ylabel(_ylab)
    ax_sec_dot.axhline(0, color='#bbb', lw=0.5, ls='--', zorder=0)
    ax_sec_dot.set_box_aspect(3.0)
    ax_sec_dot.spines['bottom'].set_visible(False)
    ax_sec_dot.tick_params(axis='x', bottom=False)

    # ----------------------------------------------------------
    # Per-sector integrals (stim -> reward) for panels 5 and 6.
    # all_sec[ch] is (n_total_sectors, n_t) on t_common, row-aligned
    # across channels (sector i is the same sector for red/grn/red_corr).
    # ----------------------------------------------------------
    _int_t0 = 0.0
    _int_t1 = float(_t_rew_plot) if _t_rew_plot > 0 else float(t_post)
    _mask_int = (t_common >= _int_t0) & (t_common <= _int_t1)
    _integrals = {}
    if _mask_int.sum() >= 2:
        _t_int = t_common[_mask_int]
        for ch in channels:
            _mat = all_sec[ch][:, _mask_int]
            with warnings.catch_warnings(), np.errstate(invalid='ignore'):
                warnings.simplefilter('ignore', category=RuntimeWarning)
                _integrals[ch] = np.trapz(_mat, _t_int, axis=1)
    else:
        for ch in channels:
            _integrals[ch] = np.full(all_sec[ch].shape[0], np.nan)

    # Panel 5: histogram of per-sector stim->rew integrals (red, red_corr)
    ax_hist = fig.add_subplot(gs[2, 2:4])
    _hist_pairs = (('red', _CHANNEL_COLOURS['red']),
                   ('red_corr', _CHANNEL_COLOURS['red_corr']))
    _hist_data = {}
    for ch, _col in _hist_pairs:
        _vals = _integrals[ch]
        _vals = _vals[np.isfinite(_vals)]
        if norm_resp_hist and _vals.size >= 2:
            _sd = float(np.std(_vals))
            if _sd > 0:
                _vals = (_vals - float(np.mean(_vals))) / _sd
        _hist_data[ch] = _vals

    # Shared bins across the two distributions for fair overlay
    _all_h = np.concatenate(
        [v for v in _hist_data.values() if v.size]) \
        if any(v.size for v in _hist_data.values()) else np.array([])
    if _all_h.size:
        _bins = np.linspace(float(np.nanmin(_all_h)),
                            float(np.nanmax(_all_h)), 101)
    else:
        _bins = 100

    for ch, _col in _hist_pairs:
        _vals = _hist_data[ch]
        if _vals.size == 0:
            continue
        ax_hist.hist(_vals, bins=_bins, histtype='stepfilled',
                     color=_col, edgecolor=_col, alpha=0.4,
                     lw=1.0, density=True,
                     label=_CHANNEL_LABELS[ch])
    _hist_xlab = ('activity (stim) (z)'
                  if norm_resp_hist else 'activity (stim)')
    ax_hist.set_xlabel(_hist_xlab)
    ax_hist.set_ylabel('probability density')
    ax_hist.legend(fontsize=7, frameon=False)
    for _side in ('top', 'right'):
        ax_hist.spines[_side].set_visible(False)

    # Panel 6: scatter of grn integral (x) vs red & red_corr (y)
    ax_scat = fig.add_subplot(gs[2, 4:6])
    _x = _integrals['grn']
    for _y_ch in ('red', 'red_corr'):
        _y = _integrals[_y_ch]
        _ok = np.isfinite(_x) & np.isfinite(_y)
        ax_scat.scatter(_x[_ok], _y[_ok],
                        facecolors='none',
                        edgecolors=_CHANNEL_COLOURS[_y_ch],
                        s=18, lw=0.8, alpha=0.7,
                        label=f'{_CHANNEL_LABELS[_y_ch]} vs grn')
    # y = x reference line over the joint data range
    _xy_all = np.concatenate([
        _integrals[c][np.isfinite(_integrals[c])]
        for c in ('grn', 'red', 'red_corr')
        if _integrals[c].size])
    if _xy_all.size:
        _lo, _hi = float(np.min(_xy_all)), float(np.max(_xy_all))
        ax_scat.plot([_lo, _hi], [_lo, _hi],
                     color='#bbb', lw=0.5, ls='--', zorder=0)
    ax_scat.set_xlabel('green activity (stim)')
    ax_scat.set_ylabel(
        r'red / red$_\mathrm{corrected}$ activity (stim)')
    ax_scat.legend(fontsize=7, frameon=False)
    for _side in ('top', 'right'):
        ax_scat.spines[_side].set_visible(False)

    # Panel 7: pooled per-cell donut-ring dF/F heatmap, spanning the full
    # row width. Styled to match the panel 2 sector heatmaps (same
    # 'RdBu_r' diverging colormap) so the two heatmap rows read as the
    # same kind of quantity; cell_vlim follows sector_vlim's semantics
    # (None -> derive vlim from the actual pooled cell data, float ->
    # pin an explicit range for cross-cohort comparison).
    if _has_cells:
        ax_cell_hm = fig.add_subplot(gs[3, 0:6])
        _c_vmin, _c_vmax = _symmetric_vlim(cells, vlim=cell_vlim)
        _c_im = ax_cell_hm.imshow(
            cells, aspect='auto', origin='upper',
            extent=[t_common[0], t_common[-1], cells.shape[0], 0],
            cmap='RdBu_r', vmin=_c_vmin, vmax=_c_vmax,
            interpolation='nearest')
        ax_cell_hm.axvline(0, color='#ff000d', lw=0.8, ls='--', alpha=0.8)
        if t_common[0] <= _t_rew_plot <= t_common[-1]:
            ax_cell_hm.axvline(_t_rew_plot, color='#0165fc',
                               lw=0.8, ls='--', alpha=0.8)
        # 'stim -> rew', not 'stim→rew': the publication_ml stylesheet
        # uses Helvetica, which has no U+2194 glyph.
        ax_cell_hm.set_title('cells (sorted by stim-rew response)',
                             color=_CELL_COLOUR)
        ax_cell_hm.set_xlabel('time from stim (s)')
        ax_cell_hm.set_ylabel(f'cell (1..{cells.shape[0]})')
        _c_cb = fig.colorbar(_c_im, ax=ax_cell_hm,
                             fraction=0.04, pad=0.02)
        _c_cb.set_label('dF/F (%)')

    fig.suptitle(
        f'{title}  ({len(records)} recordings, '
        f'sort_by={sort_by}, sec={sector_corr_metric})')

    # Close the style context now that the figure is fully built;
    # any later show/save calls render with the style baked in.
    _style_ctx.__exit__(None, None, None)

    if save_path is not None:
        # Canonical filename: <fname_stem>_<kwargs>.<ext>, so the
        # cohort + kwarg context is baked into every saved file. The
        # user-supplied save_path is treated as a directory (or, if it
        # has an extension, its containing directory is used and only
        # the extension is honoured).
        _kw_parts = [
            f'layer={str(plt_layer).replace("/", "-")}',
            f'sort={sort_by}',
            f'sec={sector_corr_metric}',
            f'tpre={t_pre:g}',
            f'tpost={t_post:g}',
            f'dt={dt:g}',
            f'trew={t_rew:g}',
            f'vlim={_vmax:g}',
            f'norm={int(bool(norm_resp_hist))}',
        ]
        # Only appended when they were actually requested, so files
        # written by plt_grab_ctrl_dualcolour keep their historical name.
        if correct_signal_kwargs is not None:
            _kw_parts.append(f'corr={_cs_method}')
        if _has_cells:
            _kw_parts.append('cells=1')
            _kw_parts.append(f'cellvlim={_c_vmax:g}')
            if drop_flat_cells:
                _kw_parts.append('noflat=1')
        _kw_suffix = '_'.join(_kw_parts)
        _root, _ext = os.path.splitext(save_path)
        if _ext:
            _save_dir = os.path.dirname(save_path) or '.'
        else:
            _save_dir = save_path
            _ext = '.pdf'
        _fname = f'{fname_stem}_{_kw_suffix}{_ext}'
        os.makedirs(_save_dir, exist_ok=True)
        _full_save_path = os.path.join(_save_dir, _fname)
        fig.savefig(_full_save_path, dpi=150)
        print(f'  saved figure -> {_full_save_path}', flush=True)
    if plt_show:
        plt.show()

    stats_out = SimpleNamespace(
        wf_pearson=SimpleNamespace(t=t_wf, p=p_wf,
                                   rg=np.asarray(pear_rg_wf),
                                   cg=np.asarray(pear_cg_wf)),
        sector_corr=SimpleNamespace(
            metric=sector_corr_metric,
            t=t_sec, p=p_sec,
            rg=np.asarray(sec_corr_rg),
            cg=np.asarray(sec_corr_cg)))

    return SimpleNamespace(
        fig=fig, records=records,
        t_common=t_common, wf=wf, sectors=all_sec,
        cells=cells, cell_records=_cell_records,
        sector_integrals=_integrals,
        t_rew=_t_rew_plot,
        stats=stats_out)


def plt_grab_ctrl_dualcolour(csv_path,
            data_folder='/Volumes/T7/5HTCtx',
            sort_by='red_corr',
            sector_corr_metric='pearson',
            t_pre=2.0,
            t_post=6.0,
            dt=0.05,
            t_rew=2.0,
            sector_vlim=None,
            norm_resp_hist=True,
            plt_layer='all',
            figsize=(6.86, 5),
            save_path=None,
            plt_show=True):
    """Across-recording summary plot of QC response_mean outputs.

    For each row in the csv, loads the most recently saved
    `_qc_response_mean*.npy` from ``{data_folder}/{animal}/{date}/
    data_mbl/`` (as written by ``_plt_qc_response_mean``), verifies that
    all loaded files share the same kwarg-suffix, and draws a 6-panel
    figure summarising the cohort.

    See ``plt_summary`` for a variant that additionally targets one
    correction parameterisation, adds a pooled per-cell row, and takes
    explicit dF/F axis limits.

    Panels
    ------
    1.  Whole-frame trial-averaged traces, aligned to stim onset, one
        column per channel (red / green / red_corr). Each recording's
        mean trace is shown at lw=0.5, alpha=0.4; the grand mean is
        drawn over the top at lw=2 with an SEM-shaded band.
    2.  Per-sector trial-averaged response heatmaps (one row per
        sector, pooled across all recordings), one column per channel.
        Sectors are sorted by post-stim peak in the ``sort_by``
        channel; the same row ordering is applied to all three columns.
    3.  Paired column dot plot of whole-frame Pearson correlation
        (red vs grn) and (red_corr vs grn); open circles, joined by a
        thin line per recording. Stats: paired t-test.
    4.  Paired column dot plot of per-sector cross-channel correlation
        (red vs grn) and (red_corr vs grn). The metric is chosen by
        ``sector_corr_metric`` ('pearson' or 'spearman'). Stats: paired
        t-test.
    5.  Histogram of per-sector stim->reward integrals (red, red_corr).
    6.  Scatter of the green integral against red / red_corr.

    Parameters
    ----------
    csv_path : str
        Path to a filelist csv with at least an 'expref' column (see
        ``bulk_run_twop_correction``).
    data_folder : str
        Root data folder; expref maps to
        ``{data_folder}/{animal}/{date}/data_mbl/``.
        Default '/Volumes/T7/5HTCtx'.
    sort_by : str
        Channel to use for sorting the sector heatmaps. One of 'red',
        'grn', 'red_corr'. Default 'red_corr'.
    sector_corr_metric : str
        'pearson' or 'spearman'. Used for panel 4. Default 'pearson'.
    t_pre, t_post : float
        Common-time-grid window around stim onset (seconds) for
        interpolating per-recording response_mean traces.
    dt : float
        Step (seconds) of the common interpolation grid.
    t_rew : float
        Fallback time (s) at which to draw a reward marker on the
        whole-frame traces. Used only if no recording in the cohort
        has a `rew_t_rel` field in its saved summary. Default 2.0.
    sector_vlim : float or None
        Symmetric colour-scale limit for the sector heatmaps
        (vmin = -sector_vlim, vmax = +sector_vlim). None (default)
        derives it from the actual pooled sector data across all three
        channels, as the 95th percentile of |value| (clipping ~5% of
        values rather than stretching to the true max, which a few
        outlier sectors can dominate); pass a float to pin the same
        numeric range across separately-plotted cohorts for comparison.
    norm_resp_hist : bool
        If True (default), z-score each channel's per-sector
        integral distribution before plotting the histogram in
        panel 5 so red and red_corr can be compared on a common
        scale. If False, plot the raw integral values.
    plt_layer : str
        Filter csv rows by the ``dual2p_folder`` tag.
        'all' (default) keeps every row;
        '2/3' keeps only canonical layer 2/3 recordings (t001 / t-001);
        '1'   keeps only layer 1 recordings (t002 / t-002 / t003 /
              t-003).
    figsize : tuple
        Figure size in inches.
    save_path : str or None
        If given, save the figure as a pdf at this path.
    plt_show : bool
        If True, call plt.show() at the end.

    Returns
    -------
    out : SimpleNamespace
        .fig          : matplotlib Figure
        .records      : list of loaded record namespaces
        .t_common     : common time vector (s)
        .wf           : dict {channel: (n_rec, n_t) interpolated traces}
        .sectors      : dict {channel: (n_total_sec, n_t) sorted sector
                              traces, sorted by sort_by}
        .stats        : SimpleNamespace with paired-T results
    """
    return _plt_summary_core(
        csv_path,
        data_folder=data_folder,
        sort_by=sort_by,
        sector_corr_metric=sector_corr_metric,
        t_pre=t_pre, t_post=t_post, dt=dt, t_rew=t_rew,
        sector_vlim=sector_vlim,
        norm_resp_hist=norm_resp_hist,
        plt_layer=plt_layer,
        figsize=figsize,
        save_path=save_path,
        plt_show=plt_show,
        correct_signal_kwargs=None,
        plt_cells=False,
        dff_min=None, dff_max=None,
        fname_stem='fig_dualcolour_grab',
        title='plt_grab_ctrl_dualcolour')


def plt_summary(csv_path,
                data_folder='/Volumes/mbl_data/5HTCtx/',
                sort_by='red_corr',
                sector_corr_metric='pearson',
                t_pre=2.0,
                t_post=6.0,
                dt=0.05,
                t_rew=2.0,
                sector_vlim=None,
                norm_resp_hist=True,
                plt_layer='all',
                correct_signal=True,
                correct_signal_kwargs=None,
                detrend_sigs=False,
                plt_cells=True,
                drop_flat_cells=False,
                dff_min=None,
                dff_max=None,
                cell_vlim=None,
                figsize=(6.86, 7.0),
                save_path=None,
                plt_show=True):
    """Across-recording summary plot, with correction targeting and cells.

    Variant of ``plt_grab_ctrl_dualcolour``: every panel of that figure
    is drawn identically (whole-frame traces, sector heatmaps, the two
    paired-correlation dot plots, the integral histogram and scatter),
    plus three additions:

    1.  ``correct_signal_kwargs`` picks WHICH saved QC output to plot.
        Each ``plt_qc`` run stamps its correction parameterisation into
        the filename (``_corrsig=1_method=full_regress_fit=global...``),
        so a recording swept over several methods holds several
        ``_qc_response_mean*.npy``. Without this argument the newest is
        taken; with it, only the file matching this parameterisation is
        loaded, making a sweep reproducible. Nothing is re-corrected
        here -- if no matching file exists, run ``plt_qc`` /
        ``bulk_run_twop_correction`` with those kwargs first.
    2.  A pooled per-cell heatmap row (panel 7), the cohort-level
        counterpart of the per-recording ``*_qc_cells_event_avg*.pdf``,
        drawn with the same 'RdBu_r' diverging colormap as panel 2.
    3.  Colour/y-axis scaling is data-driven throughout, not a single
        global clamp: panel 1's red/grn axes share one y-lim spanning
        their combined actual data, red_corr gets its own y-lim from
        just its data; panel 2's sector heatmaps and panel 7's cell
        heatmap each derive a symmetric colour range from their own
        actual data. ``sector_vlim`` / ``cell_vlim`` let you override
        either with an explicit number -- e.g. to pin the same numeric
        range across two separately-plotted cohorts for comparison.
        ``dff_min`` / ``dff_max`` are accepted but no longer used (kept
        only so old call sites don't error).

    Correction methods
    ------------------
    ``correct_signal_kwargs['method']`` selects the correction that was
    run; the remaining keys are that function's own kwargs. The full set
    dispatched by ``TwoPRec.correct_signal`` is:

    'full_regress' (default)
        ``signal_correction.correct_full_regress`` -- Martianova-style
        zdFF: airPLS baseline removal, a robust beta fit of the control
        against the signal, then normalised subtraction. Notable kwargs:
        ``fit_mode`` ('global' / 'per_sector' / 'per_pixel'),
        ``f0_level``, ``f0_n_sectors``, ``spatial_avg_sigma_px``,
        ``beta_loss``, ``beta_f_scale``, ``beta_scale``,
        ``regress_type``, ``nn_slope``, ``smooth_window``,
        ``airpls_lam``, ``airpls_porder``, ``airpls_max_iter``,
        ``trim_initial``.
    'two_stage'
        ``correct_two_stage_regress`` -- hierarchical multi-scale
        regression against orthogonalised sector-mean regressors.
        Notable kwargs: ``sector_levels``, ``f0_level``,
        ``orthogonalise``, ``min_block_var``, ``min_block_corr``, plus
        the shared airPLS / beta keys above.
    'pixel_spatial_subtr'
        ``correct_pixel_spatial_subtr`` -- per-pixel dF/F minus a
        Gaussian spatial average of the control, at unit gain (no
        beta). Notable kwargs: ``pixel_gate`` ('gmm' / 'percentile'),
        ``gmm_log``, ``gate_pct``, ``segmentation_type`` ('gmm' /
        'cells'), ``soma_min_darkness``, ``soma_diam_px``,
        ``ring_width_px``, ``sigma_px``, ``den_min``, ``f0_mode``
        ('global' / 'per_trial'), ``n_frames_f0``, ``stim_step_dff``,
        ``aggregate_sectors``.
        **This is the only method that emits per-cell traces** --
        specifically with ``segmentation_type='cells'`` -- so it is the
        only one for which ``plt_cells=True`` produces anything.
    'lms'
        ``correct_lms_adaptive`` -- per-pixel LMS adaptive filter.
        Notable kwargs: ``filter_order``, ``mu``, ``normalized``.
    'pca'
        ``correct_pca_shared_variance`` -- removes principal components
        shared between the two channels. Notable kwargs:
        ``n_components``, ``variance_threshold``,
        ``s1_loading_threshold``, ``spatial_subsample``,
        ``fit_backend``, ``fit_temporal_stride``.
    'ica'
        ``correct_ica_shared_components`` -- same idea via independent
        components. Notable kwargs: ``n_components``,
        ``s1_loading_threshold``, ``max_iter``, ``n_fit_frames``,
        ``random_state``.
    'nmf'
        ``correct_nmf_shared_components`` -- non-negative factorisation
        of the (non-negative) fluorescence. Notable kwargs:
        ``n_components``, ``s1_loading_threshold``, ``max_iter``,
        ``zscore_input``, ``random_state``.

    Every method also accepts the stim light-leak step keys
    (``remove_stim_step``, ``stim_step_edge``, ``stim_step_dff``), and
    these DO affect the saved filename -- so they must match what
    ``plt_qc`` was run with for a file to be found. ``plt_qc``'s own
    default is ``{'method': 'full_regress', 'remove_stim_step': True}``,
    which is what to pass here to target a default QC run.

    Parameters
    ----------
    csv_path, data_folder, sort_by, sector_corr_metric, t_pre, t_post,
    dt, t_rew, sector_vlim, norm_resp_hist, plt_layer, figsize,
    save_path, plt_show
        As in ``plt_grab_ctrl_dualcolour``.
    correct_signal : bool
        Whether the targeted QC run had correction enabled. Only used to
        build the filename fragment (a False run is tagged
        ``_corrsig=0_method=none``). Ignored when
        correct_signal_kwargs is None. Default True.
    correct_signal_kwargs : dict or None
        The correction parameterisation to load, as passed to
        ``plt_qc``. See "Correction methods" above. None (default)
        restores ``plt_grab_ctrl_dualcolour``'s newest-file selection.
    detrend_sigs : bool
        Whether the targeted QC run linearly detrended its traces; also
        part of the filename fragment. Default False.
    plt_cells : bool
        If True (default), add the pooled per-cell row. Silently
        skipped -- with a printed note -- when no recording has a
        ``*_celltraces_trialavg.npy``, or when correct_signal_kwargs
        names a method other than 'pixel_spatial_subtr'.

        Note that the cell-trace file is named from the imaging stem
        alone and so is written once per DATE, not per behaviour
        session: a date with two sessions has one file shared by both
        csv rows (the last correction run for that date wins). Such a
        file is pooled once, not once per row, and the cells row of a
        ``plt_layer='1'`` figure may therefore contain the same cells
        as the ``plt_layer='2/3'`` one for those dates.
    drop_flat_cells : bool
        If True, exclude cells whose saved trace has zero variance.
        ``correct_pixel_spatial_subtr`` writes an all-zero row when a
        soma's donut ring ends up with no usable pixels; those rows are
        failed extractions rather than measured null responses, so
        averaging them in biases the pooled mean toward zero. Default
        False (keep them), which matches the per-recording
        ``qc_cells_event_avg`` figure -- but the count and percentage
        are always printed, so check it before trusting the pooled
        amplitude.
    dff_min, dff_max : float or None
        Unused -- accepted only so existing call sites don't error.
        Panel 1's trace y-axes are now data-driven (see point 3 above);
        use ``sector_vlim`` / ``cell_vlim`` to control the heatmap
        colour scales instead.
    cell_vlim : float or None
        Symmetric colour-scale limit for the panel-7 per-cell heatmap
        (vmin=-cell_vlim, vmax=+cell_vlim), mirroring ``sector_vlim``.
        None (default) derives it from the actual pooled cell data
        (post ``drop_flat_cells`` filtering) as the 95th percentile of
        |dF/F|, clipping ~5% of values rather than stretching to the
        true max; pass a float to pin the same numeric range across
        separately-plotted cohorts for comparison. Units are dF/F (%),
        matching the cells row.

    Returns
    -------
    out : SimpleNamespace
        As ``plt_grab_ctrl_dualcolour``, plus
        .cells        : (n_total_cells, n_t) pooled per-cell dF/F (%),
                        sorted by stim->reward response (empty when the
                        cells row was skipped)
        .cell_records : list of loaded per-cell record namespaces
    """
    return _plt_summary_core(
        csv_path,
        data_folder=data_folder,
        sort_by=sort_by,
        sector_corr_metric=sector_corr_metric,
        t_pre=t_pre, t_post=t_post, dt=dt, t_rew=t_rew,
        sector_vlim=sector_vlim,
        norm_resp_hist=norm_resp_hist,
        plt_layer=plt_layer,
        figsize=figsize,
        save_path=save_path,
        plt_show=plt_show,
        correct_signal=correct_signal,
        correct_signal_kwargs=correct_signal_kwargs,
        detrend_sigs=detrend_sigs,
        plt_cells=plt_cells,
        drop_flat_cells=drop_flat_cells,
        dff_min=dff_min, dff_max=dff_max,
        cell_vlim=cell_vlim,
        fname_stem='fig_summary',
        title='plt_summary')


# ===========================================================================
# Cross-cohort comparison of pooled red_corrected stim-window activity
# ===========================================================================
# Companion to plt_summary: rather than one cohort's full QC panel set,
# reduces each sector's / cell's red_corrected trace to a single scalar
# (its mean over a fixed window from stim onset) and compares the two
# cohorts' distributions via cumulative histograms + a KS test.
# ===========================================================================


_COMPARE_COLOURS = ('#d62728', '#7f7f7f')  # red, grey -- matches
                                            # _GROUP_COLOURS below


def _pooled_red_corr_window_means(csv_path, data_folder,
                                  correct_signal=True,
                                  correct_signal_kwargs=None,
                                  detrend_sigs=False,
                                  plt_layer='all',
                                  t_pre=2.0, t_post=6.0, dt=0.05,
                                  t_int=(0.0, 2.0),
                                  drop_flat_cells=True,
                                  verbose=True):
    """Pooled per-sector and per-cell red_corrected window means.

    Loads a cohort's saved QC outputs the same way ``plt_summary`` does,
    then reduces each sector's and each cell's red_corrected trace to a
    single scalar: its mean over `t_int` (seconds from stim onset).
    Companion to ``plt_summary_compare_cohorts``.

    Parameters
    ----------
    csv_path, data_folder, correct_signal, correct_signal_kwargs,
    detrend_sigs, plt_layer, t_pre, t_post, dt
        As in ``plt_summary``.
    t_int : tuple of float
        (start, end) seconds from stim onset to average each trace over.
        Default (0.0, 2.0) -- the stim -> reward window in this dataset.
    drop_flat_cells : bool
        As in ``plt_summary``; excludes failed-ring-extraction cells
        from the per-cell values.
    verbose : bool
        Print progress, matching ``plt_summary``'s console output.

    Returns
    -------
    out : SimpleNamespace
        .sector_vals : (n_sectors,) per-sector mean red_corrected value
        .cell_vals   : (n_cells,) per-cell mean red_corrected value
        .n_recordings : int
    """
    _corrsig_suffix = None if correct_signal_kwargs is None else \
        corrsig_suffix_from_kwargs(
            correct_signal_kwargs=correct_signal_kwargs,
            correct_signal=correct_signal, detrend_sigs=detrend_sigs)

    records, suffix_counts = _collect_response_mean_records(
        csv_path, data_folder, verbose=verbose, plt_layer=plt_layer,
        corrsig_suffix=_corrsig_suffix)
    if not records:
        raise RuntimeError(
            f'No _qc_response_mean*.npy files were loaded for '
            f'{csv_path!r}; cannot compute window means.')
    if len(suffix_counts) > 1:
        _dom_sfx = max(suffix_counts, key=lambda k: suffix_counts[k])
        records = [r for r in records if r.suffix == _dom_sfx]

    t_common = np.arange(-t_pre, t_post + dt / 2, dt)
    _mask = (t_common >= t_int[0]) & (t_common <= t_int[1])

    # Per-sector red_corrected window means
    _sector_means = []
    for rec in records:
        summary = rec.summary
        t_rel_sec = np.asarray(
            summary.get('t_rel_sec', summary.get('t_rel', [])),
            dtype=np.float64)
        sec_d = summary.get('sectors', {})
        corr_key = _resolve_corr_key(summary)
        sec_arr = sec_d.get(corr_key) if corr_key else None
        if sec_arr is None:
            continue
        sec_arr = np.asarray(sec_arr, dtype=np.float64)
        if sec_arr.ndim != 2 or sec_arr.shape[1] != t_rel_sec.size:
            continue
        _interp = _interp_to_common(t_rel_sec, sec_arr, t_common)
        with warnings.catch_warnings(), np.errstate(invalid='ignore'):
            warnings.simplefilter('ignore', category=RuntimeWarning)
            _sector_means.append(np.nanmean(_interp[:, _mask], axis=1))
    sector_vals = (np.concatenate(_sector_means) if _sector_means
                  else np.array([]))

    # Per-cell red_corrected window means. Cell traces are already the
    # corrected (donut-ring-subtracted) signal -- see plt_summary's
    # plt_cells docs -- so no corr_key resolution is needed here.
    _cell_records = _collect_celltraces_records(
        csv_path, data_folder, verbose=verbose, plt_layer=plt_layer,
        keep_exprefs={r.expref for r in records})
    _cell_means = []
    for _crec in _cell_records:
        _d = _crec.cells
        _ct = _d.get('cell_traces_trialavg')
        _tw = _d.get('t_win')
        if _ct is None or _tw is None:
            continue
        _ct = np.asarray(_ct, dtype=np.float64)
        _tw = np.asarray(_tw, dtype=np.float64)
        if _ct.ndim != 2 or _ct.shape[1] != _tw.size:
            continue
        # Baseline-subtract and scale to % dF/F, matching plt_summary's
        # cells row so the two figures describe the same quantity.
        _pre = _tw < 0
        if np.any(_pre):
            with warnings.catch_warnings(), np.errstate(invalid='ignore'):
                warnings.simplefilter('ignore', category=RuntimeWarning)
                _base = np.nanmean(_ct[:, _pre], axis=1)
            _ct = _ct - _base[:, None]
        with warnings.catch_warnings(), np.errstate(invalid='ignore'):
            warnings.simplefilter('ignore', category=RuntimeWarning)
            _flat = np.nanstd(_ct, axis=1) == 0
        if drop_flat_cells and _flat.any():
            _ct = _ct[~_flat]
        if _ct.shape[0] == 0:
            continue
        _interp = _interp_to_common(_tw, _ct * 100.0, t_common)
        with warnings.catch_warnings(), np.errstate(invalid='ignore'):
            warnings.simplefilter('ignore', category=RuntimeWarning)
            _cell_means.append(np.nanmean(_interp[:, _mask], axis=1))
    cell_vals = (np.concatenate(_cell_means) if _cell_means
                else np.array([]))

    return SimpleNamespace(sector_vals=sector_vals, cell_vals=cell_vals,
                           n_recordings=len(records))


def plt_summary_compare_cohorts(csv_path_a, csv_path_b,
        labels=('grab5ht_gfp', 'grab5htmut_gfp'),
        data_folder='/Volumes/mbl_data/5HTCtx',
        t_pre=2.0, t_post=6.0, dt=0.05,
        t_int=(0.0, 2.0),
        plt_layer='all',
        correct_signal=True,
        correct_signal_kwargs=None,
        detrend_sigs=False,
        drop_flat_cells=True,
        log_prob=False,
        n_bins=100,
        figsize=(7.0, 3.4),
        save_path=None,
        plt_show=True):
    """Compare two cohorts' pooled red_corrected stim-window activity.

    Companion figure to ``plt_summary``: instead of one cohort's full
    QC panel set, this reduces every sector's and every cell's
    red_corrected trace to a single scalar (its mean over `t_int`
    seconds from stim onset -- the stim -> reward window by default)
    and draws two cumulative-distribution subplots -- one across
    sectors, one across cells -- comparing the two cohorts named by
    `csv_path_a` / `csv_path_b`. Colour follows the same red/grey
    convention as the hemisphere-control cohort figure (red = cohort A,
    grey = cohort B); each subplot is annotated with a two-sample
    Kolmogorov-Smirnov test.

    Parameters
    ----------
    csv_path_a, csv_path_b : str
        Filelist csvs for the two cohorts to compare (A, B).
    labels : tuple of str
        Legend / annotation labels for (A, B). Default
        ('grab5ht_gfp', 'grab5htmut_gfp').
    data_folder : str
        As in ``plt_summary``.
    t_pre, t_post, dt : float
        Interpolation grid, as in ``plt_summary``.
    t_int : tuple of float
        (start, end) seconds from stim onset to average each sector's /
        cell's trace over before building the distributions. Default
        (0.0, 2.0).
    plt_layer, correct_signal, correct_signal_kwargs, detrend_sigs
        As in ``plt_summary`` -- must match the QC run being targeted
        for the .npy files to be found.
    drop_flat_cells : bool
        As in ``plt_summary``.
    log_prob : bool
        If True, plot the cumulative-probability y-axis on a log scale
        (reveals tail behaviour that a linear axis compresses); if
        False (default), linear.
    n_bins : int
        Histogram bin count for the cumulative step plot. Default 100.
    figsize : tuple
        Figure size in inches.
    save_path : str or None
        If given, save the figure as a pdf. `save_path` is treated as a
        directory (or, if it has an extension, its containing directory
        is used and only the extension is honoured) -- the filename is
        built from the cohort labels and kwargs, matching
        ``plt_summary``'s save convention.
    plt_show : bool
        If True, call plt.show() at the end.

    Returns
    -------
    out : SimpleNamespace
        .fig       : matplotlib Figure
        .data      : {label: SimpleNamespace(sector_vals, cell_vals,
                                             n_recordings)}
        .ks_sector : (statistic, pvalue) for the sector-level KS test
        .ks_cell   : (statistic, pvalue) for the cell-level KS test
    """
    _cs_method = (correct_signal_kwargs or {}).get('method', 'full_regress')

    _print_banner('PLT_COMPARE')
    print(f'  csv A ({labels[0]}) : {csv_path_a}')
    print(f'  csv B ({labels[1]}) : {csv_path_b}')

    _data = {}
    for _lab, _csv in zip(labels, (csv_path_a, csv_path_b)):
        print(f'  --- {_lab} ---', flush=True)
        _data[_lab] = _pooled_red_corr_window_means(
            _csv, data_folder,
            correct_signal=correct_signal,
            correct_signal_kwargs=correct_signal_kwargs,
            detrend_sigs=detrend_sigs,
            plt_layer=plt_layer,
            t_pre=t_pre, t_post=t_post, dt=dt, t_int=t_int,
            drop_flat_cells=drop_flat_cells, verbose=True)
        print(f'      n recordings = {_data[_lab].n_recordings}, '
              f'n sectors = {_data[_lab].sector_vals.size}, '
              f'n cells = {_data[_lab].cell_vals.size}', flush=True)

    _colours = dict(zip(labels, _COMPARE_COLOURS))

    _style_ctx = plt.style.context('publication_ml')
    _style_ctx.__enter__()  # closed below (search: _style_ctx.__exit__)
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout='constrained')

    _int_lab = f'{t_int[0]:g}-{t_int[1]:g} s'
    _panel_specs = [
        (axes[0], 'sector_vals', 'sectors',
         r'red$_\mathrm{corrected}$ mean dF/F, ' + _int_lab
         + ' (per sector)'),
        (axes[1], 'cell_vals', 'cells',
         r'red$_\mathrm{corrected}$ mean dF/F, ' + _int_lab
         + ' (per cell)'),
    ]

    ks_results = {}
    for ax, _attr, _unit, _xlabel in _panel_specs:
        _vA = getattr(_data[labels[0]], _attr)
        _vB = getattr(_data[labels[1]], _attr)
        _vA = _vA[np.isfinite(_vA)]
        _vB = _vB[np.isfinite(_vB)]

        _all_v = (np.concatenate([_vA, _vB]) if (_vA.size or _vB.size)
                  else np.array([0.0, 1.0]))
        _bins = np.linspace(float(np.min(_all_v)), float(np.max(_all_v)),
                            n_bins + 1)

        for _lab, _v in zip(labels, (_vA, _vB)):
            if _v.size == 0:
                continue
            ax.hist(_v, bins=_bins, density=True, cumulative=True,
                   histtype='step', color=_colours[_lab], lw=1.5,
                   label=f'{_lab} (n={_v.size})')

        if _vA.size >= 2 and _vB.size >= 2:
            _ks_stat, _ks_p = sp_stats.ks_2samp(_vA, _vB)
            _ks_stat, _ks_p = float(_ks_stat), float(_ks_p)
        else:
            _ks_stat, _ks_p = np.nan, np.nan
        ks_results[_unit] = (_ks_stat, _ks_p)

        ax.text(0.03, 0.97, f'KS D = {_ks_stat:.3f}\np = {_ks_p:.2g}',
               transform=ax.transAxes, ha='left', va='top', fontsize=7)
        ax.axhline(0.5, color='#bbb', lw=0.5, ls=':', zorder=0)
        ax.set_xlabel(_xlabel)
        ax.set_ylabel('cumulative probability')
        if log_prob:
            ax.set_yscale('log')
        ax.legend(fontsize=6, frameon=False, loc='lower right')
        ax.set_title(f'across {_unit}', fontsize=9)
        for _side in ('top', 'right'):
            ax.spines[_side].set_visible(False)

    fig.suptitle(
        r'red$_\mathrm{corrected}$ stim-window activity: '
        f'{labels[0]} vs {labels[1]}  ({_int_lab})')

    # Close the style context now that the figure is fully built; any
    # later show/save calls render with the style baked in.
    _style_ctx.__exit__(None, None, None)

    if save_path is not None:
        _kw_parts = [
            f'tint={t_int[0]:g}-{t_int[1]:g}',
            f'logprob={int(bool(log_prob))}',
        ]
        if correct_signal_kwargs is not None:
            _kw_parts.append(f'corr={_cs_method}')
        if drop_flat_cells:
            _kw_parts.append('noflat=1')
        _kw_suffix = '_'.join(_kw_parts)
        _root, _ext = os.path.splitext(save_path)
        if _ext:
            _save_dir = os.path.dirname(save_path) or '.'
            _ext = _ext or '.pdf'
        else:
            _save_dir = save_path
            _ext = '.pdf'
        os.makedirs(_save_dir, exist_ok=True)
        _fname = (f'fig_compare_red_corr_{labels[0]}_vs_{labels[1]}_'
                 f'{_kw_suffix}{_ext}')
        _full_save_path = os.path.join(_save_dir, _fname)
        fig.savefig(_full_save_path, dpi=150)
        print(f'  saved figure -> {_full_save_path} '
              f'(exists={os.path.exists(_full_save_path)})', flush=True)

    if plt_show:
        plt.show()
    else:
        plt.close(fig)

    return SimpleNamespace(
        fig=fig, data=_data,
        ks_sector=ks_results['sectors'], ks_cell=ks_results['cells'])


# ===========================================================================
# Significant-cell detection against a null cohort
# ===========================================================================
# Third companion figure to plt_summary. Uses the control (mutant)
# cohort's per-cell stim-window dF/F distribution as an empirical null,
# z-scores every signal-cohort cell against it, and flags the tail.
# ===========================================================================


def _pooled_cell_window_data(csv_path, data_folder,
                             correct_signal=True,
                             correct_signal_kwargs=None,
                             detrend_sigs=False,
                             plt_layer='all',
                             t_pre=2.0, t_post=6.0, dt=0.05,
                             t_int=(0.0, 2.0),
                             drop_flat_cells=True,
                             verbose=True):
    """Pooled per-cell traces, window means and centroids for a cohort.

    Like ``_pooled_red_corr_window_means`` but keeps the per-cell
    identity needed for the significant-cell figure: each cell's full
    interpolated trace, its scalar window mean, its (row, col) centroid
    and the expref of the recording it came from.

    Parameters
    ----------
    csv_path, data_folder, correct_signal, correct_signal_kwargs,
    detrend_sigs, plt_layer, t_pre, t_post, dt, t_int, drop_flat_cells,
    verbose
        As in ``_pooled_red_corr_window_means``.

    Returns
    -------
    out : SimpleNamespace
        .traces    : (n_cells, n_t) baseline-subtracted dF/F (%)
        .vals      : (n_cells,) mean over t_int
        .centroids : (n_cells, 2) (row, col) in frame pixels
        .exprefs   : (n_cells,) object array of source expref per cell
        .t_common  : (n_t,) time vector
        .order     : list of exprefs in load order
    """
    _corrsig_suffix = None if correct_signal_kwargs is None else \
        corrsig_suffix_from_kwargs(
            correct_signal_kwargs=correct_signal_kwargs,
            correct_signal=correct_signal, detrend_sigs=detrend_sigs)

    records, suffix_counts = _collect_response_mean_records(
        csv_path, data_folder, verbose=False, plt_layer=plt_layer,
        corrsig_suffix=_corrsig_suffix)
    if not records:
        raise RuntimeError(
            f'No _qc_response_mean*.npy files were loaded for '
            f'{csv_path!r}; cannot select cells.')
    if len(suffix_counts) > 1:
        _dom_sfx = max(suffix_counts, key=lambda k: suffix_counts[k])
        records = [r for r in records if r.suffix == _dom_sfx]

    t_common = np.arange(-t_pre, t_post + dt / 2, dt)
    _mask = (t_common >= t_int[0]) & (t_common <= t_int[1])

    _cell_records = _collect_celltraces_records(
        csv_path, data_folder, verbose=verbose, plt_layer=plt_layer,
        keep_exprefs={r.expref for r in records})

    _traces, _cents, _refs, _order = [], [], [], []
    for _crec in _cell_records:
        _d = _crec.cells
        _ct = _d.get('cell_traces_trialavg')
        _tw = _d.get('t_win')
        if _ct is None or _tw is None:
            continue
        _ct = np.asarray(_ct, dtype=np.float64)
        _tw = np.asarray(_tw, dtype=np.float64)
        if _ct.ndim != 2 or _ct.shape[1] != _tw.size:
            continue
        _cen = _d.get('cell_centroids')
        _cen = (np.asarray(_cen, dtype=np.float64) if _cen is not None
                else np.full((_ct.shape[0], 2), np.nan))

        # Same baseline-subtract + scale-to-% convention as plt_summary's
        # cells row, so all three figures describe the same quantity.
        _pre = _tw < 0
        if np.any(_pre):
            with warnings.catch_warnings(), np.errstate(invalid='ignore'):
                warnings.simplefilter('ignore', category=RuntimeWarning)
                _base = np.nanmean(_ct[:, _pre], axis=1)
            _ct = _ct - _base[:, None]

        # Flat rows are failed ring extractions, not null responses --
        # drop centroids alongside so the arrays stay row-aligned.
        with warnings.catch_warnings(), np.errstate(invalid='ignore'):
            warnings.simplefilter('ignore', category=RuntimeWarning)
            _flat = np.nanstd(_ct, axis=1) == 0
        if drop_flat_cells and _flat.any():
            _ct = _ct[~_flat]
            if _cen.shape[0] == _flat.size:
                _cen = _cen[~_flat]
        if _ct.shape[0] == 0:
            continue

        _traces.append(_interp_to_common(_tw, _ct * 100.0, t_common))
        _cents.append(_cen)
        _refs.append(np.array([_crec.expref] * _ct.shape[0], dtype=object))
        _order.append(_crec.expref)

    if not _traces:
        raise RuntimeError(
            f'No per-cell traces were loaded for {csv_path!r}.')

    traces = np.concatenate(_traces, axis=0)
    centroids = np.concatenate(_cents, axis=0)
    exprefs = np.concatenate(_refs, axis=0)
    with warnings.catch_warnings(), np.errstate(invalid='ignore'):
        warnings.simplefilter('ignore', category=RuntimeWarning)
        vals = np.nanmean(traces[:, _mask], axis=1)

    return SimpleNamespace(traces=traces, vals=vals, centroids=centroids,
                           exprefs=exprefs, t_common=t_common,
                           order=_order)


# ---------------------------------------------------------------------------
# Bin-by-bin PDF enrichment
# ---------------------------------------------------------------------------

def _pdf_enrichment_bins(signal_vals, null_vals, n_bins=12, margin=2.0,
                         ci=0.95, side='both', contiguous=True,
                         bonferroni=True, min_pooled=10,
                         binning='quantile'):
    """Flag bins where the signal density exceeds the null density.

    Distribution-shape-agnostic alternative to z-scoring against a null
    cohort's mean/SD, for use when the null is heavy-tailed and so a
    z threshold carries no calibrated meaning.

    Bins are built from the POOLED values, never from the cohort
    labels -- which is what keeps the conditional test below valid.
    With ``binning='quantile'`` (default) they are equal-occupancy, so
    they are narrow where the data is dense and wide in the sparse
    tails and every bin holds a similar number of cells; with
    ``binning='equal'`` they are equal-width across the pooled range.
    Within each bin the density ratio is

        R_i = (k1_i / n1) / (k0_i / n0)

    -- the bin widths cancel out of the densities, leaving a ratio of
    bin proportions.

    Notes
    -----
    The confidence bound is EXACT (conditional binomial), not a
    bootstrap. Conditioning on the pooled count m_i = k1_i + k0_i is
    legitimate for either binning because the edges are derived from
    the pooled VALUES alone and never from the cohort labels; under
    ``binning='quantile'`` m_i is additionally fixed by construction,
    which is why that mode spends its statistical power evenly.
    Conditional on m_i the only random quantity is how those cells
    split between cohorts; writing pi_i = k1_i / m_i,

        R_i = (n0 / n1) * pi_i / (1 - pi_i)

    which is strictly increasing in pi_i, so a Clopper-Pearson lower
    bound on pi_i maps directly to an exact lower bound on R_i.

    A bootstrap must NOT be used here. Resampling cells is multinomial
    resampling of the bin counts, so when a bin holds zero null cells
    every draw returns zero for it: the interval cannot express any
    uncertainty in the null count precisely where that uncertainty
    matters. Empirically, for k1=25, k0=0 (n1=1248, n0=784) a
    percentile bootstrap returns a lower bound of ~20.7 where the exact
    bound is 4.9 -- it would flag the sparsest, least trustworthy bins
    with the greatest apparent confidence.

    Under the usual two-component mixture (the signal cohort being
    responders plus a non-responder component distributed as the null
    cohort -- the same assumption the z-score route already makes), the
    non-responder fraction among flagged cells in bin i is 1 / R_i. So
    `margin` is a per-bin false-discovery ceiling: margin=2 allows at
    most 50% contamination, margin=3 at most 33%, margin=4 at most 25%.
    Using the lower bound makes 1 / r_lo a conservative bound on that.

    The interval is conditional on the bin edges, which are themselves
    estimated from the same pooled cells; edge uncertainty is not
    propagated.

    Parameters
    ----------
    signal_vals, null_vals : np.ndarray
        Per-cell scalar responses for the two cohorts. Non-finite
        entries are excluded from the fit and never flagged.
    n_bins : int
        Requested quantile-bin count. Ties in the pooled quantiles can
        collapse edges, so the effective count may be lower. Keep this
        small: the pooled count per bin is what powers the test, and at
        n_bins=80 with ~2000 cells a bin holds ~25 cells, which needs
        an observed ratio above ~5.7 to clear margin=2 -- effectively
        no bin can ever pass on merit.
    margin : float
        Required lower bound on the density ratio. Must exceed 1.0.
    ci : float
        ONE-SIDED confidence level for the lower bound (the decision
        rule r_lo >= margin is one-sided, so the tail mass is 1 - ci).
    side : str
        Which bins may be flagged: 'upper' (bins above the null
        median), 'lower' (below it), or 'both' (default, any bin).
    contiguous : bool
        If True (default), keep only the unbroken run of flagged bins
        anchored at the OUTERMOST FLAGGED bin and walking inward,
        discarding isolated flagged bins mid-distribution. The result
        is then a threshold on the response value (`thresh_upper` /
        `thresh_lower`) and cells are flagged BY VALUE against it, so
        cells beyond the run -- which under equal-width binning may sit
        in bins too sparse to test -- are still included.
    bonferroni : bool
        If True (default), divide the tail mass by the effective bin
        count, correcting for testing every bin simultaneously.
    min_pooled : int
        Minimum POOLED cells (k1 + k0) for a bin to be eligible. Guards
        against bins too small to support any inference. Note a guard
        on k1 alone would not do this -- a bin with k1=25, k0=0 passes
        a k1 guard while being the least trustworthy bin present. This
        matters far more under ``binning='equal'``, where a heavy tail
        leaves the outermost bins nearly empty.
    binning : str
        'quantile' (default) for equal-occupancy bins, or 'equal' for
        equal-width bins across the pooled range. Equal-width bins keep
        the familiar histogram shape and a linear value axis, but at
        the cost of spending most bins on the sparse tails, where m_i
        falls below `min_pooled` and no test can run.

    Returns
    -------
    out : SimpleNamespace
        .signif       : (n_signal,) bool, per-cell flag
        .edges        : (n_eff + 1,) bin edges
        .n_bins_eff   : int, effective bin count after tie collapse
        .k1, .k0, .m  : (n_eff,) signal / null / pooled counts per bin
        .r_hat        : (n_eff,) point-estimate density ratio (inf when
                        k0 == 0)
        .r_lo         : (n_eff,) exact lower confidence bound on R
        .flag_bin     : (n_eff,) bool, which bins were kept
        .fdr_bound    : (n_eff,) 1 / r_lo, the contamination ceiling
        .thresh_upper : float, lowest value in the kept upper run (nan
                        if none) -- the data-driven response threshold
        .thresh_lower : float, highest value in the kept lower run
        .alpha        : float, per-bin tail mass actually used
        .n_tested     : int, eligible bins (the Bonferroni divisor)
        .n1, .n0      : int, finite cell counts per cohort
        .n_dropped_noncontig : int, bins dropped by the contiguity rule
    """
    if margin <= 1.0:
        raise ValueError(
            f'margin must exceed 1.0 (a ratio floor at or below 1 is '
            f'not an enrichment criterion), got {margin!r}')
    if not (0.0 < ci < 1.0):
        raise ValueError(f'ci must lie in (0, 1), got {ci!r}')
    if side not in ('upper', 'lower', 'both'):
        raise ValueError(
            f"side must be 'upper', 'lower' or 'both', got {side!r}")
    if int(n_bins) < 3:
        raise ValueError(f'n_bins must be >= 3, got {n_bins!r}')
    if binning not in ('quantile', 'equal'):
        raise ValueError(
            f"binning must be 'quantile' or 'equal', got {binning!r}")

    _v1 = np.asarray(signal_vals, dtype=np.float64).ravel()
    _v0 = np.asarray(null_vals, dtype=np.float64).ravel()
    _fin1 = np.isfinite(_v1)
    _fin0 = np.isfinite(_v0)
    _n1, _n0 = int(_fin1.sum()), int(_fin0.sum())
    if _n1 < 2 * n_bins or _n0 < 2 * n_bins:
        raise RuntimeError(
            f'need at least {2 * n_bins} finite values per cohort for '
            f'n_bins={n_bins}; got signal={_n1}, null={_n0}')

    # Edges from the pooled data only (never the labels). np.quantile
    # returns all-NaN if any non-finite value slips through, so the
    # finite masks are applied first either way.
    _pooled = np.concatenate([_v1[_fin1], _v0[_fin0]])
    if binning == 'quantile':
        _edges = np.unique(
            np.quantile(_pooled, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    else:
        _edges = np.linspace(float(_pooled.min()), float(_pooled.max()),
                             int(n_bins) + 1)
    _n_eff = _edges.size - 1
    if _n_eff < 3:
        raise RuntimeError(
            f'quantile edges collapsed to {_n_eff} bins; the pooled '
            f'distribution is too degenerate for this method')
    if _n_eff < int(n_bins):
        warnings.warn(
            f'tied {binning} edges collapsed {n_bins} requested bins '
            f'to {_n_eff}', RuntimeWarning, stacklevel=2)

    # digitize against the INTERIOR edges only: the outermost bins are
    # then implicitly unbounded and absorb everything beyond the
    # observed range, so no cell can fall outside the binning.
    _b1 = np.digitize(_v1[_fin1], _edges[1:-1])
    _b0 = np.digitize(_v0[_fin0], _edges[1:-1])
    _k1 = np.bincount(_b1, minlength=_n_eff).astype(np.float64)
    _k0 = np.bincount(_b0, minlength=_n_eff).astype(np.float64)
    _m = _k1 + _k0

    with np.errstate(divide='ignore', invalid='ignore'):
        _r_hat = np.where(_k0 > 0, (_k1 / _n1) / (_k0 / _n0), np.inf)
        _r_hat = np.where(_k1 > 0, _r_hat, 0.0)

    # Eligibility first: the Bonferroni correction must divide by the
    # number of bins actually TESTED, not by every bin present, or
    # restricting `side` would pay a correction for tests never run.
    _elig = _m >= float(min_pooled)
    _ref = float(np.median(_v0[_fin0]))
    if side == 'upper':
        _elig &= _edges[:-1] >= _ref
    elif side == 'lower':
        _elig &= _edges[1:] <= _ref
    _n_test = max(int(_elig.sum()), 1)

    # Exact one-sided Clopper-Pearson lower bound on pi = k1 / m, then
    # mapped through the monotone R(pi) = (n0/n1) * pi / (1 - pi).
    _alpha = (1.0 - ci) / (_n_test if bonferroni else 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        _pi_lo = np.where(
            _k1 > 0,
            sp_stats.beta.ppf(_alpha, np.maximum(_k1, 1.0),
                              _m - _k1 + 1.0),
            0.0)
        _pi_lo = np.nan_to_num(_pi_lo, nan=0.0)
        _r_lo = (_n0 / _n1) * _pi_lo / np.clip(1.0 - _pi_lo, 1e-12, None)

    _flag = _elig & (_r_lo >= margin)

    # Contiguity: keep the run anchored at the OUTERMOST FLAGGED bin
    # and walk inward. Anchoring on the outermost *eligible* bin
    # instead would be wrong under equal-width binning, where the
    # extreme tail bins hold too few cells to ever reach significance
    # and would break the run before it started.
    _n_drop = 0
    _thr_up, _thr_lo = np.nan, np.nan
    if contiguous and _flag.any():
        # Split the flagged bins into maximal contiguous runs, then
        # keep at most one per side. A run only counts as an UPPER run
        # if it sits entirely above the null median (and vice versa),
        # which stops a single high run from also being read as a
        # lower-tail result under side='both'.
        _fl = np.flatnonzero(_flag)
        _brk = np.flatnonzero(np.diff(_fl) > 1)
        _starts = np.concatenate([[_fl[0]], _fl[_brk + 1]])
        _ends = np.concatenate([_fl[_brk], [_fl[-1]]])
        _keep = np.zeros_like(_flag)
        if side in ('upper', 'both'):
            _s, _e = int(_starts[-1]), int(_ends[-1])
            if _edges[_s] >= _ref:
                _keep[_s:_e + 1] = True
                _thr_up = float(_edges[_s])
        if side in ('lower', 'both'):
            _s, _e = int(_starts[0]), int(_ends[0])
            if _edges[_e + 1] <= _ref:
                _keep[_s:_e + 1] = True
                _thr_lo = float(_edges[_e + 1])
        _n_drop = int(_flag.sum() - _keep.sum())
        if _n_drop:
            warnings.warn(
                f'contiguity rule dropped {_n_drop} flagged bin(s) not '
                f'in the outermost run on their side',
                RuntimeWarning, stacklevel=2)
        _flag = _keep

    # With a contiguous run the result IS a threshold, so flag by value
    # rather than by bin membership. That matters under equal-width
    # binning: cells further out than the run sit in bins too sparse to
    # test, and it would be perverse to flag a cell at 5.5% while
    # leaving a stronger one at 9% unflagged.
    signif = np.zeros(_v1.size, dtype=bool)
    if contiguous and (np.isfinite(_thr_up) or np.isfinite(_thr_lo)):
        _sel = np.zeros(_n1, dtype=bool)
        if np.isfinite(_thr_up):
            _sel |= _v1[_fin1] >= _thr_up
        if np.isfinite(_thr_lo):
            _sel |= _v1[_fin1] <= _thr_lo
        signif[_fin1] = _sel
    else:
        signif[_fin1] = _flag[_b1]

    with np.errstate(divide='ignore', invalid='ignore'):
        _fdr = np.where(_r_lo > 0, 1.0 / _r_lo, np.nan)

    return SimpleNamespace(
        signif=signif, edges=_edges, n_bins_eff=_n_eff,
        k1=_k1, k0=_k0, m=_m, r_hat=_r_hat, r_lo=_r_lo,
        flag_bin=_flag, fdr_bound=_fdr,
        thresh_upper=_thr_up, thresh_lower=_thr_lo,
        alpha=_alpha, n_tested=_n_test, n1=_n1, n0=_n0,
        binning=binning, n_dropped_noncontig=_n_drop)


def plt_summary_significant_cells(csv_path_signal, csv_path_null,
        labels=('grab5ht_gfp', 'grab5htmut_gfp'),
        data_folder='/Volumes/mbl_data/5HTCtx',
        method='zscore',
        zscore_signif=3.0,
        tail='two',
        pdf_n_bins=12,
        pdf_binning='quantile',
        pdf_margin=2.0,
        pdf_ci=0.95,
        pdf_side='both',
        pdf_contiguous=True,
        pdf_bonferroni=True,
        pdf_min_pooled=10,
        t_pre=2.0, t_post=6.0, dt=0.05,
        t_int=(0.0, 2.0),
        plt_layer='all',
        correct_signal=True,
        correct_signal_kwargs=None,
        detrend_sigs=False,
        drop_flat_cells=True,
        cell_vlim=None,
        n_bins=80,
        figsize=(7.0, 8.6),
        save_path=None,
        plt_show=True):
    """Flag signal-cohort cells whose stim response exceeds a null cohort.

    Third companion figure to ``plt_summary``. Each cell's response is
    reduced to its mean dF/F over `t_int` seconds from stim onset, and
    the NULL cohort (`csv_path_null`, the mutant control) is used to
    decide which SIGNAL cohort cells (`csv_path_signal`) respond.
    `method` selects how:

    'zscore'
        The null supplies a mean and standard deviation; each signal
        cell is z-scored against them and flagged past
        `zscore_signif`.
    'pdf_enrichment'
        The two cohorts' PDFs are compared bin by bin on quantile
        (equal-occupancy) bins of the pooled values, and cells are
        kept from bins where the signal density exceeds the null
        density by a margin that survives an exact confidence bound.
        See ``_pdf_enrichment_bins``.

    The enrichment route exists because the null here is strongly
    heavy-tailed: 5.0% of signal cells clear |z| >= 3, where a Gaussian
    null implies ~0.27%, so the z threshold has no calibrated meaning.
    Reducing both distributions to two moments discards exactly the
    shape information that distinguishes them.

    Panels
    ------
    1.  Method-dependent. For 'zscore', a histogram of signal-cohort
        z-scores with the threshold marked and the null's own z-scores
        overlaid in grey (~N(0, 1) by construction). For
        'pdf_enrichment', a two-part panel: per-bin proportions for
        both cohorts, above the per-bin density ratio with its
        confidence band, the `pdf_margin` line and the flagged bins.
    2.  Heatmap of the flagged cells' dF/F traces (RdBu_r, matching
        ``plt_summary``'s panel 2 / 7), sorted by response amplitude.
    3.  Spatial map of cell centroids, one small panel per recording,
        with flagged cells in red over all cells in grey.
    4.  Placeholder for across-trial reliability -- see Notes.

    Notes
    -----
    Panel 4 (per-cell across-trial Pearson r) CANNOT currently be
    computed from the saved artefacts and is drawn as an explanatory
    placeholder. ``correct_pixel_spatial_subtr`` writes only
    trial-AVERAGED per-cell traces (`cell_traces_trialavg`); the
    per-trial data that exists is per-PIXEL, and the donut-ring labels
    needed to map pixels back onto cells are never persisted (only
    `cell_centroids` are). Producing it requires saving per-cell
    per-trial traces at correction time and re-running the correction.

    Parameters
    ----------
    csv_path_signal, csv_path_null : str
        Filelist csvs for the signal cohort (cells to test) and the
        null cohort (supplies the null distribution).
    labels : tuple of str
        (signal, null) names, used in titles and the saved filename.
    data_folder : str
        As in ``plt_summary``.
    method : str
        'zscore' (default, preserving historical behaviour) or
        'pdf_enrichment'. See the summary above.
    zscore_signif : float
        Threshold in null-distribution standard deviations. Default
        3.0. Only used when ``method='zscore'``.
    tail : str
        Which tail counts as significant. 'two' (default) flags
        ``|z| >= zscore_signif``; 'upper' flags ``z >= +thr`` only;
        'lower' flags ``z <= -thr`` only. Two-sided is the default
        because a corrected GRAB trace can deviate in either direction
        and the null is not assumed one-sided. Only used when
        ``method='zscore'``; use `pdf_side` for the enrichment route.
    pdf_n_bins : int
        Bin count for ``method='pdf_enrichment'``. Default 12.
        Deliberately small and deliberately NOT `n_bins`: the pooled
        count per bin is what powers the test, and with ~2000 pooled
        cells a large bin count leaves too few cells per bin for any
        bin to clear `pdf_margin` on merit. Fine binning is more
        defensible under ``pdf_binning='equal'``, where the dense
        centre can afford narrow bins.
    pdf_binning : str
        'quantile' (default, equal-occupancy) or 'equal' (equal-width
        across the pooled range). See ``_pdf_enrichment_bins``.
    pdf_margin : float
        Required lower confidence bound on the per-bin density ratio.
        Default 2.0. Doubles as a per-bin false-discovery ceiling:
        1 / margin bounds the non-responder fraction among flagged
        cells, so 2.0 allows at most 50% contamination.
    pdf_ci : float
        One-sided confidence level for that bound. Default 0.95.
    pdf_side : str
        'both' (default), 'upper' or 'lower' -- which side of the null
        median a bin must sit on to be eligible.
    pdf_contiguous : bool
        Keep only the unbroken run of flagged bins anchored at the
        eligible extreme, so the result stays expressible as a
        threshold on dF/F. Default True.
    pdf_bonferroni : bool
        Correct the tail mass for testing every bin. Default True.
    pdf_min_pooled : int
        Minimum pooled (signal + null) cells for a bin to be eligible.
        Default 10.
    t_pre, t_post, dt, t_int, plt_layer, correct_signal,
    correct_signal_kwargs, detrend_sigs, drop_flat_cells
        As in ``plt_summary`` / ``plt_summary_compare_cohorts``.
    cell_vlim : float or None
        Symmetric colour limit for the panel-2 heatmap. None (default)
        derives it from the flagged cells' data at the 95th percentile
        of |dF/F|, as elsewhere in this module.
    n_bins : int
        Smoothing bin count for the panel-1 z-score histogram only.
        Default 80. Distinct from `pdf_n_bins`.
    figsize : tuple
        Figure size in inches.
    save_path : str or None
        Directory (or a path whose extension is honoured) to save into.
    plt_show : bool
        If True, call plt.show() at the end.

    Returns
    -------
    out : SimpleNamespace
        .fig        : matplotlib Figure
        .method     : str, the method used
        .z          : (n_cells,) signal-cohort z-scores (always
                      computed, so callers reading .z keep working
                      whichever method ran)
        .signif     : (n_cells,) bool flag
        .null_mean, .null_std : float, float
        .signal, .null : the two pooled SimpleNamespaces
        .enrich     : ``_pdf_enrichment_bins`` output for
                      method='pdf_enrichment', else None -- so
                      ``if out.enrich is not None:`` dispatches cleanly
    """
    if method not in ('zscore', 'pdf_enrichment'):
        raise ValueError(
            f"method must be 'zscore' or 'pdf_enrichment', got "
            f"{method!r}")
    if tail not in ('two', 'upper', 'lower'):
        raise ValueError(
            f"tail must be 'two', 'upper' or 'lower', got {tail!r}")
    if method == 'pdf_enrichment' and tail != 'two':
        raise ValueError(
            f"tail={tail!r} has no effect when method='pdf_enrichment'; "
            f"use pdf_side to restrict which bins are eligible")

    _cs_method = (correct_signal_kwargs or {}).get('method', 'full_regress')
    _common = dict(
        data_folder=data_folder, correct_signal=correct_signal,
        correct_signal_kwargs=correct_signal_kwargs,
        detrend_sigs=detrend_sigs, plt_layer=plt_layer,
        t_pre=t_pre, t_post=t_post, dt=dt, t_int=t_int,
        drop_flat_cells=drop_flat_cells)

    _print_banner('PLT_SIGNIF_CELLS')
    print(f'  signal ({labels[0]}) : {csv_path_signal}')
    print(f'  null   ({labels[1]}) : {csv_path_null}')

    print(f'  --- {labels[1]} (null) ---', flush=True)
    null = _pooled_cell_window_data(csv_path_null, verbose=True, **_common)
    print(f'  --- {labels[0]} (signal) ---', flush=True)
    signal = _pooled_cell_window_data(csv_path_signal, verbose=True,
                                      **_common)

    _null_fin = null.vals[np.isfinite(null.vals)]
    null_mean = float(np.mean(_null_fin)) if _null_fin.size else np.nan
    null_std = float(np.std(_null_fin, ddof=1)) if _null_fin.size > 1 \
        else np.nan
    if method == 'zscore' and not (np.isfinite(null_std) and null_std > 0):
        raise RuntimeError(
            f'Null cohort {labels[1]!r} has a degenerate spread '
            f'(std={null_std!r}); cannot z-score against it.')

    # z is computed either way: panel 1's null overlay uses it and the
    # return contract keeps it, so callers stay method-agnostic.
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (signal.vals - null_mean) / null_std
        z_null = (null.vals - null_mean) / null_std

    enrich = None
    if method == 'zscore':
        if tail == 'two':
            signif = np.abs(z) >= zscore_signif
        elif tail == 'upper':
            signif = z >= zscore_signif
        else:
            signif = z <= -zscore_signif
        signif &= np.isfinite(z)
    else:
        enrich = _pdf_enrichment_bins(
            signal.vals, null.vals, n_bins=pdf_n_bins,
            margin=pdf_margin, ci=pdf_ci, side=pdf_side,
            contiguous=pdf_contiguous, bonferroni=pdf_bonferroni,
            min_pooled=pdf_min_pooled, binning=pdf_binning)
        signif = enrich.signif

    _n_sig = int(signif.sum())
    _n_tot = int(np.isfinite(signal.vals).sum())
    _pct = 100.0 * _n_sig / max(_n_tot, 1)

    if method == 'zscore':
        print(f'  null  : mean = {null_mean:.4f}, sd = {null_std:.4f} '
              f'(n = {_null_fin.size} cells)')
        print(f'  signal: {_n_sig} / {_n_tot} cells significant '
              f'({_pct:.1f}%) at |z| >= {zscore_signif:g} (tail={tail})',
              flush=True)
    else:
        print(f'  bins  : {enrich.n_bins_eff} {enrich.binning} bins, '
              f'{enrich.n_tested} eligible '
              f'(side={pdf_side}, min_pooled={pdf_min_pooled}), '
              f'per-bin alpha = {enrich.alpha:.4g} '
              f'(bonferroni={pdf_bonferroni})')
        print(f'  flagged bins ({int(enrich.flag_bin.sum())} of '
              f'{enrich.n_bins_eff}):')
        for _i in np.flatnonzero(enrich.flag_bin):
            print(f'      [{enrich.edges[_i]:7.2f},'
                  f'{enrich.edges[_i + 1]:7.2f}]  '
                  f'k1={int(enrich.k1[_i]):4d} k0={int(enrich.k0[_i]):4d}  '
                  f'R={enrich.r_hat[_i]:6.2f}  '
                  f'R_lo={enrich.r_lo[_i]:5.2f}  '
                  f'FDR<={enrich.fdr_bound[_i]:.2f}')
        if np.isfinite(enrich.thresh_upper):
            print(f'  upper threshold: dF/F >= '
                  f'{enrich.thresh_upper:.2f}%')
        if np.isfinite(enrich.thresh_lower):
            print(f'  lower threshold: dF/F <= '
                  f'{enrich.thresh_lower:.2f}%')
        print(f'  signal: {_n_sig} / {_n_tot} cells flagged '
              f'({_pct:.1f}%) at R_lo >= {pdf_margin:g}', flush=True)

    # Per-recording breakdown: a cohort-level percentage can hide a
    # result that lives almost entirely in one session, so always show
    # where the flagged cells actually came from.
    print('  per-recording flagged cells:')
    _refs_all = signal.exprefs
    for _ref in dict.fromkeys(signal.order):
        _m_ref = _refs_all == _ref
        _n_ref = int(_m_ref.sum())
        _s_ref = int(signif[_m_ref].sum())
        print(f'      {_ref:26s} {_s_ref:4d} / {_n_ref:4d}  '
              f'({100.0 * _s_ref / max(_n_ref, 1):5.1f}%)')
    print('', end='', flush=True)

    _col_sig, _col_null = _COMPARE_COLOURS

    _style_ctx = plt.style.context('publication_ml')
    _style_ctx.__enter__()  # closed below (search: _style_ctx.__exit__)
    fig = plt.figure(figsize=figsize, layout='constrained')
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[1.15, 1.6, 0.45])

    # --- Panel 1: method-dependent ---
    if method == 'zscore':
        ax_h = fig.add_subplot(gs[0, 0])
        _zf = z[np.isfinite(z)]
        _znf = z_null[np.isfinite(z_null)]
        _all_z = np.concatenate([_zf, _znf]) if (_zf.size or _znf.size) \
            else np.array([0.0, 1.0])
        _bins = np.linspace(float(np.min(_all_z)), float(np.max(_all_z)),
                            n_bins + 1)
        ax_h.hist(_znf, bins=_bins, density=True, histtype='stepfilled',
                  color=_col_null, alpha=0.35, lw=1.0,
                  label=f'{labels[1]} (null, n={_znf.size})')
        ax_h.hist(_zf, bins=_bins, density=True, histtype='step',
                  color=_col_sig, lw=1.5,
                  label=f'{labels[0]} (n={_zf.size})')
        for _s in ((1, -1) if tail == 'two'
                   else ((1,) if tail == 'upper' else (-1,))):
            ax_h.axvline(_s * zscore_signif, color='k', lw=1.0, ls='--',
                         alpha=0.8)
        ax_h.set_xlabel('z-score vs null cohort')
        ax_h.set_ylabel('probability density')
        ax_h.set_title(
            f'z-scores  ({_n_sig}/{_n_tot} signif, {_pct:.1f}%)',
            fontsize=8)
        ax_h.legend(fontsize=5.5, frameon=False)
        for _side in ('top', 'right'):
            ax_h.spines[_side].set_visible(False)
    else:
        # Two stacked sub-axes sharing a BIN-INDEX x-axis. Plotting
        # against value would be unreadable: equal-occupancy bins are
        # orders of magnitude wider in the tails, so the enriched
        # region -- the whole point -- would collapse to a sliver.
        _p_spec = gs[0, 0].subgridspec(2, 1, height_ratios=[0.9, 1.1],
                                       hspace=0.32)
        ax_h = fig.add_subplot(_p_spec[0])
        ax_r = fig.add_subplot(_p_spec[1], sharex=ax_h)
        _idx = np.arange(enrich.n_bins_eff)

        ax_h.step(_idx, enrich.k1 / enrich.n1, where='mid',
                  color=_col_sig, lw=1.4,
                  label=f'{labels[0]} (n={enrich.n1})')
        ax_h.step(_idx, enrich.k0 / enrich.n0, where='mid',
                  color=_col_null, lw=1.4,
                  label=f'{labels[1]} (null, n={enrich.n0})')
        ax_h.set_ylabel('fraction of cells')
        ax_h.set_title(
            f'per-bin PDFs  ({_n_sig}/{_n_tot} flagged, {_pct:.1f}%)',
            fontsize=8)
        ax_h.legend(fontsize=5.5, frameon=False)
        ax_h.tick_params(labelbottom=False)

        _r_plot = np.where(np.isfinite(enrich.r_hat), enrich.r_hat,
                           np.nan)
        ax_r.step(_idx, _r_plot, where='mid', color='k', lw=1.2)
        ax_r.fill_between(_idx, enrich.r_lo, _r_plot, step='mid',
                          color='k', alpha=0.15, lw=0)
        ax_r.axhline(1.0, color='#888', lw=0.7, ls='-', alpha=0.7)
        ax_r.axhline(pdf_margin, color='k', lw=1.0, ls='--', alpha=0.85)
        if enrich.flag_bin.any():
            ax_r.scatter(_idx[enrich.flag_bin],
                         np.clip(_r_plot[enrich.flag_bin], None, 1e4),
                         s=14, color=_col_sig, zorder=3,
                         label=f'flagged ({int(enrich.flag_bin.sum())})')
            ax_r.legend(fontsize=5.5, frameon=False, loc='upper left')
        ax_r.set_yscale('log')
        ax_r.set_ylabel('density ratio\n(signal / null)')
        ax_r.set_xlabel(f'pooled {enrich.binning} bin '
                        f'(dF/F %, bin edges)')
        # Label a few bins with their actual dF/F edge so amplitudes
        # remain readable off a bin-index axis.
        _tick = np.linspace(0, enrich.n_bins_eff - 1, 5).astype(int)
        ax_r.set_xticks(_tick)
        ax_r.set_xticklabels([f'{enrich.edges[_t]:.1f}' for _t in _tick],
                             fontsize=5)
        for _ax in (ax_h, ax_r):
            for _side in ('top', 'right'):
                _ax.spines[_side].set_visible(False)

    # --- Panel 2: heatmap of flagged cells only ---
    ax_hm = fig.add_subplot(gs[0, 1])
    _sig_tr = signal.traces[signif]
    if _sig_tr.shape[0]:
        _sig_vals = signal.vals[signif]
        _sig_tr = _sig_tr[np.argsort(
            -np.nan_to_num(_sig_vals, nan=-np.inf))]
        _vmin, _vmax = _symmetric_vlim(_sig_tr, vlim=cell_vlim)
        _im = ax_hm.imshow(
            _sig_tr, aspect='auto', origin='upper',
            extent=[signal.t_common[0], signal.t_common[-1],
                    _sig_tr.shape[0], 0],
            cmap='RdBu_r', vmin=_vmin, vmax=_vmax,
            interpolation='nearest')
        ax_hm.axvline(0, color='#ff000d', lw=0.8, ls='--', alpha=0.8)
        if signal.t_common[0] <= t_int[1] <= signal.t_common[-1]:
            ax_hm.axvline(t_int[1], color='#0165fc', lw=0.8, ls='--',
                          alpha=0.8)
        _cb = fig.colorbar(_im, ax=ax_hm, fraction=0.04, pad=0.02)
        _cb.set_label('dF/F (%)')
        ax_hm.set_ylabel(f'significant cell (1..{_sig_tr.shape[0]})')
    else:
        ax_hm.text(0.5, 0.5, 'no significant cells', ha='center',
                   va='center', transform=ax_hm.transAxes, fontsize=8,
                   color='grey')
    ax_hm.set_xlabel('time from stim (s)')
    ax_hm.set_title(f'significant cells (n={_n_sig})', fontsize=8,
                    color=_col_sig)

    # --- Panel 3: spatial map, one small panel per recording ---
    # All panels share one square extent (the largest centroid seen
    # across the cohort) so cell positions are comparable panel to
    # panel. aspect is fixed with adjustable='box' -- 'datalim' would
    # inflate the limits to fill these short wide axes and collapse the
    # scatter onto a line.
    _recs = list(dict.fromkeys(signal.order))
    _cen_all = signal.centroids[np.isfinite(signal.centroids).all(axis=1)]
    _lim = (float(np.max(_cen_all)) * 1.02 if _cen_all.size else 1.0)
    _ncol = min(4, max(1, len(_recs)))
    _nrow = int(np.ceil(len(_recs) / _ncol)) if _recs else 1
    sp_spec = gs[1, :].subgridspec(_nrow, _ncol, hspace=0.75, wspace=0.3)
    for _i, _ref in enumerate(_recs):
        _ax = fig.add_subplot(sp_spec[_i // _ncol, _i % _ncol])
        _m = signal.exprefs == _ref
        _c = signal.centroids[_m]
        _s = signif[_m]
        if _c.size and np.any(np.isfinite(_c)):
            # centroids are (row, col); plot col as x, row as y, with y
            # inverted so the map matches image orientation.
            _ax.scatter(_c[~_s, 1], _c[~_s, 0], s=2, facecolors='none',
                        edgecolors='#c8c8c8', lw=0.3, alpha=0.8)
            _ax.scatter(_c[_s, 1], _c[_s, 0], s=9, facecolors='none',
                        edgecolors=_col_sig, lw=0.8, alpha=0.95)
            _ax.set_xlim(0, _lim)
            _ax.set_ylim(_lim, 0)
            _ax.set_aspect('equal', adjustable='box')
        else:
            _ax.text(0.5, 0.5, 'no centroids', ha='center', va='center',
                     transform=_ax.transAxes, fontsize=5, color='grey')
        _ax.set_title(f'{_ref}\n{int(_s.sum())}/{int(_m.sum())} signif',
                      fontsize=5)
        _ax.set_xticks([])
        _ax.set_yticks([])
        for _side in ('top', 'right'):
            _ax.spines[_side].set_visible(False)

    # --- Panel 4: across-trial reliability placeholder ---
    # Deliberately a placeholder, not a substitute metric: computing it
    # needs per-cell per-trial traces that the correction does not save
    # (see this function's Notes).
    ax_rel = fig.add_subplot(gs[2, :])
    ax_rel.text(
        0.5, 0.5,
        'across-trial reliability (Pearson r per cell): NOT COMPUTED\n'
        'requires per-cell per-trial traces; correct_pixel_spatial_subtr '
        'saves only trial-averaged cell traces,\nand the donut-ring '
        'labels needed to map the per-pixel trial cube onto cells are '
        'not persisted.',
        ha='center', va='center', transform=ax_rel.transAxes,
        fontsize=6, color='#993333',
        bbox=dict(boxstyle='round', facecolor='#f7eeee',
                  edgecolor='#cc9999', lw=0.6))
    ax_rel.set_xticks([])
    ax_rel.set_yticks([])
    for _side in ('top', 'right', 'bottom', 'left'):
        ax_rel.spines[_side].set_visible(False)

    if method == 'zscore':
        _crit = f'|z| >= {zscore_signif:g}'
    else:
        _crit = f'R_lo >= {pdf_margin:g}'
        if np.isfinite(enrich.thresh_upper):
            _crit += f', dF/F >= {enrich.thresh_upper:.2f}%'
        if np.isfinite(enrich.thresh_lower):
            _crit += f', dF/F <= {enrich.thresh_lower:.2f}%'
    fig.suptitle(
        f'significant cells: {labels[0]} vs {labels[1]} null  '
        f'({_crit}, {t_int[0]:g}-{t_int[1]:g} s)')

    _style_ctx.__exit__(None, None, None)

    if save_path is not None:
        # The zscore branch keeps its historical parts verbatim so
        # existing filenames stay reproducible; the enrichment branch
        # leads with a distinct token so the two can never collide.
        if method == 'zscore':
            _kw_parts = [
                f'z={zscore_signif:g}',
                f'tail={tail}',
                f'tint={t_int[0]:g}-{t_int[1]:g}',
            ]
        else:
            _kw_parts = [
                'pdfenr',
                f'bin={pdf_binning}',
                f'm={pdf_margin:g}',
                f'nb={int(pdf_n_bins)}',
                f'ci={pdf_ci:g}',
                f'side={pdf_side}',
                f'tint={t_int[0]:g}-{t_int[1]:g}',
            ]
            if pdf_contiguous:
                _kw_parts.append('contig')
            if pdf_bonferroni:
                _kw_parts.append('bonf')
        if correct_signal_kwargs is not None:
            _kw_parts.append(f'corr={_cs_method}')
        if drop_flat_cells:
            _kw_parts.append('noflat=1')
        _kw_suffix = '_'.join(_kw_parts)
        _root, _ext = os.path.splitext(save_path)
        if _ext:
            _save_dir = os.path.dirname(save_path) or '.'
        else:
            _save_dir = save_path
            _ext = '.pdf'
        os.makedirs(_save_dir, exist_ok=True)
        _fname = (f'fig_signifcells_{labels[0]}_vs_{labels[1]}_'
                  f'{_kw_suffix}{_ext}')
        _full = os.path.join(_save_dir, _fname)
        fig.savefig(_full, dpi=150)
        print(f'  saved figure -> {_full} '
              f'(exists={os.path.exists(_full)})', flush=True)

    if plt_show:
        plt.show()
    else:
        plt.close(fig)

    return SimpleNamespace(
        fig=fig, method=method, z=z, signif=signif,
        null_mean=null_mean, null_std=null_std,
        signal=signal, null=null, enrich=enrich)


# ===========================================================================
# Hemisphere-control across-recording summary plot
# ===========================================================================
# Designed for single-channel cohorts run with split_lr=True (see
# bulk_run_twop_hemispherectrl). Each recording's .npy carries per-
# hemisphere whole-frame + sector entries (`{ch}_left`, `{ch}_right`).
# A filelist csv column `hemisphere_with_grab5ht` ('left' or 'right')
# assigns hemisphere → group: the named side is the grab5ht hemisphere,
# the other side is grab5ht_mut.
# ===========================================================================


_GROUP_COLOURS = {
    'grab5ht': '#d62728',       # red
    'grab5ht_mut': '#7f7f7f',   # grey
}
_GROUP_LABELS = {
    'grab5ht': 'GRAB5HT',
    'grab5ht_mut': r'GRAB5HT$_\mathrm{mut}$',
}


def _parse_hemisphere_side(val):
    """Normalize a hemisphere_with_grab5ht csv cell to 'left'/'right'."""
    if val is None:
        return None
    try:
        s = str(val).strip().lower()
    except Exception:
        return None
    if s in ('l', 'left'):
        return 'left'
    if s in ('r', 'right'):
        return 'right'
    return None


def _collect_hemispherectrl_records(csv_path, data_folder,
                                     channel='red',
                                     verbose=True, plt_layer='all'):
    """Read csv and load the latest response_mean .npy per row, tagging
    each record with the hemisphere_with_grab5ht side and the channel
    base label used when the .npy was written.

    Skips rows where:
        - `hemisphere_with_grab5ht` is missing / unparseable
        - the .npy cannot be found / loaded
        - the loaded summary has no `{channel}_left` or `{channel}_right`
          entry in `whole_frame` (e.g. split_lr=False at save time)
    """
    filelist = pd.read_csv(csv_path)
    records = []
    suffix_counts = {}

    for i, row in enumerate(filelist.to_dict('records')):
        try:
            expref = str(row['expref'])
        except Exception:
            expref = f'row_{i}'

        if plt_layer != 'all':
            _dual_tag = str(row.get('dual2p_folder', '')).strip().lower()
            if plt_layer == '2/3':
                _allowed = _L23_TAGS
            elif plt_layer == '1':
                _allowed = _L1_TAGS
            else:
                raise ValueError(
                    f"plt_layer must be 'all', '2/3', or '1'; "
                    f'got {plt_layer!r}')
            if _dual_tag and _dual_tag not in _allowed:
                if verbose:
                    print(f'  - {expref}: skipped (plt_layer='
                          f'{plt_layer!r}, dual2p_folder='
                          f'{_dual_tag!r})')
                continue

        # Accept either 'hemisphere_with_grab5ht' or
        # 'hemisphere with grab5ht' (space-separated) as the column
        # name; some filelists use one style and some the other.
        _side_val = row.get('hemisphere_with_grab5ht',
                            row.get('hemisphere with grab5ht'))
        side = _parse_hemisphere_side(_side_val)
        if side is None:
            if verbose:
                print(f'  x {expref}: missing/invalid '
                      f'hemisphere_with_grab5ht')
            continue

        try:
            animal, date, folder_beh = _parse_expref(expref)
        except Exception as e:
            if verbose:
                print(f'  x {expref}: could not parse expref ({e})')
            continue

        data_dir = os.path.join(data_folder, animal, date, 'data_mbl')
        try:
            fpath, suffix, _ = _find_latest_response_mean_npy(
                data_dir, beh_folder=folder_beh)
        except Exception as e:
            if verbose:
                print(f'  x {expref}: error scanning {data_dir} ({e})')
            continue
        if fpath is None:
            if verbose:
                if not os.path.isdir(data_dir):
                    print(f'  x {expref}: data_mbl dir does not '
                          f'exist: {data_dir}')
                else:
                    _other = [e for e in os.listdir(data_dir)
                              if '_qc_response_mean' in e
                              and e.endswith('.npy')]
                    if _other:
                        print(f'  x {expref}: no _qc_response_mean*.npy '
                              f'matching beh={folder_beh!r} in '
                              f'{data_dir}; found {len(_other)} '
                              f'other(s) for different beh folders: '
                              f'{_other[:3]}')
                    else:
                        print(f'  x {expref}: no _qc_response_mean*.npy '
                              f'in {data_dir}')
            continue

        try:
            summary = np.load(fpath, allow_pickle=True).item()
        except Exception as e:
            if verbose:
                print(f'  x {expref}: failed to load {fpath} ({e})')
            continue

        _wf = summary.get('whole_frame', {})
        if (f'{channel}_left' not in _wf
                or f'{channel}_right' not in _wf):
            if verbose:
                print(f'  x {expref}: {os.path.basename(fpath)} has no '
                      f'{channel}_left/{channel}_right entries '
                      f'(was plt_qc run with split_lr=True?)')
            continue

        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        records.append(SimpleNamespace(
            expref=expref, fpath=fpath, suffix=suffix,
            summary=summary, grab5ht_side=side, channel=channel))
        if verbose:
            print(f'  + {expref}: {os.path.basename(fpath)} '
                  f'(grab5ht={side})')

    return records, suffix_counts


def plt_all_hemispherectrl(csv_path,
                            data_folder='/Volumes/T7/5HTCtx',
                            channel='red',
                            sort_by='grab5ht',
                            t_pre=2.0,
                            t_post=6.0,
                            dt=0.05,
                            t_rew=2.0,
                            sector_vlim=5.0,
                            norm_resp_hist=True,
                            plt_layer='all',
                            figsize=(6.0, 5.0),
                            save_path=None,
                            plt_show=True):
    """Across-recording hemisphere-control summary plot.

    Compares grab5ht vs grab5ht_mut hemisphere responses pooled across
    a single-channel split_lr cohort. Hemisphere → group is inferred
    per recording from the csv column ``hemisphere_with_grab5ht``:
    rows where it is 'left' assign ``{channel}_left`` → grab5ht and
    ``{channel}_right`` → grab5ht_mut; rows where it is 'right'
    assign the opposite.

    Panels
    ------
    1.  Whole-frame trial-averaged traces, one column per group. Each
        recording's mean trace is shown at lw=0.5, alpha=0.4; the
        grand mean ± SEM band is drawn over the top.
    2.  Per-sector trial-averaged response heatmaps, one column per
        group. Sectors are pooled across recordings within each group
        and sorted by post-stim peak. Hemispheres index different
        physical sectors so each column is sorted independently.
    3.  Paired dot plot (per recording) of whole-frame stim → reward
        integrated response, grab5ht vs grab5ht_mut.
    4.  Paired dot plot (per recording) of mean per-sector stim →
        reward integrated response, grab5ht vs grab5ht_mut.
    5.  Histogram of pooled per-sector stim → reward integrals
        (both groups overlaid).

    Parameters
    ----------
    csv_path : str
        Path to a filelist csv with at least columns 'expref',
        'dual2p_folder', and 'hemisphere_with_grab5ht'.
    data_folder : str
        Root data folder. Default '/Volumes/T7/5HTCtx'.
    channel : str
        Channel base label used when the .npy was saved (e.g. 'red').
        Default 'red'.
    sort_by : str
        Reserved for future use — each heatmap column is currently
        sorted by its own post-stim peak because the two groups index
        different physical sectors. Default 'grab5ht'.
    t_pre, t_post : float
        Common-time-grid window around stim onset (seconds).
    dt : float
        Step (seconds) of the common interpolation grid.
    t_rew : float
        Fallback time (s) at which to draw a reward marker, used only
        if no recording has a saved `rew_t_rel`. Also bounds the
        stim → reward integration window. Default 2.0.
    sector_vlim : float
        Symmetric colour-scale limit for the sector heatmaps.
        Default 5.0.
    norm_resp_hist : bool
        If True, z-score each group's per-sector integral distribution
        before plotting the histogram. Default True.
    plt_layer : str
        'all' / '2/3' / '1' — same filter as plt_all.
    figsize : tuple
        Figure size in inches.
    save_path : str or None
        If given, save the figure as a pdf at this path (kwargs are
        baked into the filename).
    plt_show : bool
        If True, call plt.show() at the end.

    Returns
    -------
    out : SimpleNamespace
        .fig                 : matplotlib Figure
        .records             : list of loaded record namespaces (with
                               .grab5ht_side attached)
        .t_common            : common time vector (s)
        .wf                  : dict {group: (n_rec, n_t) WF means}
        .sectors             : dict {group: (n_total_sec, n_t) sorted
                               sector traces}
        .sector_integrals    : dict {group: per-sector stim→rew
                               integrals (pooled, sorted)}
        .wf_integrals        : dict {group: per-recording WF integral}
        .sec_mean_integrals  : dict {group: per-recording mean sector
                               integral}
        .stats               : SimpleNamespace with paired-T results
    """
    groups = ('grab5ht', 'grab5ht_mut')
    if sort_by not in groups:
        raise ValueError(
            f'sort_by must be one of {groups}, got {sort_by!r}')

    _print_banner('PLT_ALL_HEMISPHERECTRL')
    print(f'  csv         : {csv_path}')
    print(f'  data_folder : {data_folder}')
    print(f'  channel     : {channel}')
    print(f'  sort_by     : {sort_by}')
    print(f'  plt_layer   : {plt_layer!r}', flush=True)

    records, suffix_counts = _collect_hemispherectrl_records(
        csv_path, data_folder, channel=channel,
        verbose=True, plt_layer=plt_layer)
    if not records:
        raise RuntimeError(
            'No usable _qc_response_mean*.npy files were loaded; '
            'cannot plot.')

    if len(suffix_counts) > 1:
        print('  ! WARNING: inconsistent kwarg suffixes across .npy:')
        for sfx, n in sorted(suffix_counts.items(),
                             key=lambda kv: -kv[1]):
            print(f'      ({n:3d}) {sfx!r}')
        _dom_sfx = max(suffix_counts, key=lambda k: suffix_counts[k])
        print(f'  ! using dominant suffix: {_dom_sfx!r}')
        records = [r for r in records if r.suffix == _dom_sfx]
    else:
        print(f'  suffix      : {next(iter(suffix_counts))!r}')

    print(f'  n loaded    : {len(records)}', flush=True)

    # Common time grid
    t_common = np.arange(-t_pre, t_post + dt / 2, dt)
    n_t = t_common.size

    # ----------------------------------------------------------
    # Per-recording traces onto t_common (whole-frame + sectors).
    # Per record, resolve which saved key feeds each group from its
    # grab5ht_side ('left'/'right').
    # ----------------------------------------------------------
    wf = {g: [] for g in groups}
    sec_per_rec = {g: [] for g in groups}

    for rec in records:
        summary = rec.summary
        t_rel = np.asarray(summary.get('t_rel', []), dtype=np.float64)
        t_rel_sec = np.asarray(
            summary.get('t_rel_sec', t_rel), dtype=np.float64)
        wf_d = summary.get('whole_frame', {})
        sec_d = summary.get('sectors', {})

        _other = 'right' if rec.grab5ht_side == 'left' else 'left'
        group_to_key = {
            'grab5ht': f'{channel}_{rec.grab5ht_side}',
            'grab5ht_mut': f'{channel}_{_other}',
        }

        for g in groups:
            _key = group_to_key[g]
            entry = wf_d.get(_key)
            if (entry is None or 'mean' not in entry
                    or entry['mean'] is None):
                wf[g].append(np.full(n_t, np.nan))
            else:
                wf[g].append(_interp_to_common(
                    t_rel, np.asarray(entry['mean']), t_common))

            sec_arr = sec_d.get(_key)
            if sec_arr is None:
                sec_per_rec[g].append(np.full((0, n_t), np.nan))
            else:
                sec_arr = np.asarray(sec_arr, dtype=np.float64)
                if (sec_arr.ndim != 2
                        or sec_arr.shape[1] != t_rel_sec.size):
                    sec_per_rec[g].append(np.full((0, n_t), np.nan))
                else:
                    sec_per_rec[g].append(
                        _interp_to_common(t_rel_sec, sec_arr, t_common))

    wf = {g: np.asarray(v, dtype=np.float64) for g, v in wf.items()}

    _modes = [r.summary.get('mode') for r in records
              if r.summary.get('mode') is not None]
    _mode_dom = max(set(_modes), key=_modes.count) if _modes else None
    if _mode_dom == 'dff_sig':
        _wf_ylabel = 'dF/F'
    elif _mode_dom == 'iti_zscore':
        _wf_ylabel = 'z-score'
    else:
        _wf_ylabel = 'response'

    _rew_ts = [float(r.summary.get('rew_t_rel'))
               for r in records
               if r.summary.get('rew_t_rel') is not None
               and np.isfinite(float(r.summary.get('rew_t_rel')))]
    _t_rew_plot = float(np.median(_rew_ts)) if _rew_ts else float(t_rew)

    # Pool sectors across recordings within each group. Hemispheres
    # index different physical sectors so we do NOT enforce cross-group
    # row alignment; each group's row count and ordering are
    # independent.
    # ----------
    all_sec = {}
    for g in groups:
        per_rec = sec_per_rec[g]
        if per_rec:
            all_sec[g] = np.concatenate(per_rec, axis=0)
        else:
            all_sec[g] = np.zeros((0, n_t), dtype=np.float64)

    _post = t_common >= 0
    for g in groups:
        mat = all_sec[g]
        if mat.shape[0] == 0:
            continue
        with np.errstate(invalid='ignore'):
            if np.any(_post):
                _peak = np.nanmax(mat[:, _post], axis=1)
            else:
                _peak = np.nanmax(mat, axis=1)
        _order_g = np.argsort(np.where(np.isfinite(_peak),
                                       _peak, -np.inf))[::-1]
        all_sec[g] = mat[_order_g]

    # Per-recording scalar metrics: integrate stim → reward.
    # ----------
    _int_t0 = 0.0
    _int_t1 = float(_t_rew_plot) if _t_rew_plot > 0 else float(t_post)
    _mask_int = (t_common >= _int_t0) & (t_common <= _int_t1)
    _t_int = t_common[_mask_int]

    wf_integrals = {}
    sec_mean_integrals = {}
    for g in groups:
        if _mask_int.sum() >= 2:
            with warnings.catch_warnings(), np.errstate(invalid='ignore'):
                warnings.simplefilter('ignore', category=RuntimeWarning)
                wf_integrals[g] = np.trapz(
                    wf[g][:, _mask_int], _t_int, axis=1)
        else:
            wf_integrals[g] = np.full(wf[g].shape[0], np.nan)

        _per_rec = sec_per_rec[g]
        _means = []
        for _mat in _per_rec:
            if _mat.shape[0] == 0 or _mask_int.sum() < 2:
                _means.append(np.nan)
                continue
            with warnings.catch_warnings(), np.errstate(invalid='ignore'):
                warnings.simplefilter('ignore', category=RuntimeWarning)
                _ints = np.trapz(_mat[:, _mask_int], _t_int, axis=1)
                _means.append(float(np.nanmean(_ints)))
        sec_mean_integrals[g] = np.asarray(_means, dtype=np.float64)

    sector_integrals = {}
    for g in groups:
        mat = all_sec[g]
        if mat.shape[0] == 0 or _mask_int.sum() < 2:
            sector_integrals[g] = np.full(mat.shape[0], np.nan)
        else:
            with warnings.catch_warnings(), np.errstate(invalid='ignore'):
                warnings.simplefilter('ignore', category=RuntimeWarning)
                sector_integrals[g] = np.trapz(
                    mat[:, _mask_int], _t_int, axis=1)

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------
    _style_ctx = plt.style.context('publication_ml')
    _style_ctx.__enter__()
    fig = plt.figure(figsize=figsize, layout='constrained')
    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        height_ratios=[1.0, 2.0, 1.2])

    # Panel 1: whole-frame trial-averaged traces (2 columns)
    axes_wf = [fig.add_subplot(gs[0, 2 * i:2 * i + 2])
               for i in range(2)]
    for ax, g in zip(axes_wf, groups):
        col = _GROUP_COLOURS[g]
        traces = wf[g]
        for row in traces:
            if np.any(np.isfinite(row)):
                ax.plot(t_common, row, color=col, lw=0.5, alpha=0.4)
        with warnings.catch_warnings(), np.errstate(invalid='ignore'):
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean = np.nanmean(traces, axis=0)
            n_per_t = np.sum(np.isfinite(traces), axis=0)
            sd = np.nanstd(traces, axis=0)
            sem = sd / np.sqrt(np.maximum(n_per_t, 1))
        ax.fill_between(t_common, mean - sem, mean + sem,
                        color=col, alpha=0.25, lw=0)
        ax.plot(t_common, mean, color=col, lw=2, alpha=1.0)
        ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)
        if t_common[0] <= _t_rew_plot <= t_common[-1]:
            ax.axvline(_t_rew_plot, color='#1f77b4',
                       lw=0.5, ls='--', alpha=0.7)
        ax.set_title(f'{_GROUP_LABELS[g]}  (n={traces.shape[0]})',
                     color=col)
        ax.set_xlabel('time from stim (s)')
        ax.set_ylabel(_wf_ylabel)
        ax.set_xlim(t_common[0], t_common[-1])
        for _side in ('top', 'right'):
            ax.spines[_side].set_visible(False)

    # Panel 2: per-sector heatmaps (2 columns, sorted independently)
    axes_sec = [fig.add_subplot(gs[1, 2 * i:2 * i + 2])
                for i in range(2)]
    _vmax = float(sector_vlim)
    _vmin = -_vmax
    _last_im = None
    for ax, g in zip(axes_sec, groups):
        mat = all_sec[g]
        if mat.size == 0:
            ax.text(0.5, 0.5, 'no sectors', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        im = ax.imshow(
            mat, aspect='auto', origin='upper',
            extent=[t_common[0], t_common[-1], mat.shape[0], 0],
            cmap='RdBu_r', vmin=_vmin, vmax=_vmax,
            interpolation='nearest')
        _last_im = im
        ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.6)
        if t_common[0] <= _t_rew_plot <= t_common[-1]:
            ax.axvline(_t_rew_plot, color='#1f77b4',
                       lw=0.5, ls='--', alpha=0.7)
        ax.set_title(f'{_GROUP_LABELS[g]} sectors',
                     color=_GROUP_COLOURS[g])
        ax.set_xlabel('time from stim (s)')
        ax.set_ylabel('sector (sorted by own peak)')
    if _last_im is not None:
        fig.colorbar(_last_im, ax=axes_sec[-1],
                     fraction=0.04, pad=0.02)

    # Panel 3: paired dot plot — per-recording whole-frame integral
    ax_wf_dot = fig.add_subplot(gs[2, 0])
    t_wf_int, p_wf_int = _paired_dot_plot(
        ax_wf_dot,
        wf_integrals['grab5ht'], wf_integrals['grab5ht_mut'],
        labels=(_GROUP_LABELS['grab5ht'],
                _GROUP_LABELS['grab5ht_mut']),
        colour_left=_GROUP_COLOURS['grab5ht'],
        colour_right=_GROUP_COLOURS['grab5ht_mut'])
    ax_wf_dot.set_ylabel('whole-frame\nstim→rew integral')
    ax_wf_dot.axhline(0, color='#bbb', lw=0.5, ls='--', zorder=0)
    ax_wf_dot.set_box_aspect(3.0)
    ax_wf_dot.spines['bottom'].set_visible(False)
    ax_wf_dot.tick_params(axis='x', bottom=False)

    # Panel 4: paired dot plot — per-recording mean sector integral
    ax_sec_dot = fig.add_subplot(gs[2, 1])
    t_sec_int, p_sec_int = _paired_dot_plot(
        ax_sec_dot,
        sec_mean_integrals['grab5ht'],
        sec_mean_integrals['grab5ht_mut'],
        labels=(_GROUP_LABELS['grab5ht'],
                _GROUP_LABELS['grab5ht_mut']),
        colour_left=_GROUP_COLOURS['grab5ht'],
        colour_right=_GROUP_COLOURS['grab5ht_mut'])
    ax_sec_dot.set_ylabel('mean sector\nstim→rew integral')
    ax_sec_dot.axhline(0, color='#bbb', lw=0.5, ls='--', zorder=0)
    ax_sec_dot.set_box_aspect(3.0)
    ax_sec_dot.spines['bottom'].set_visible(False)
    ax_sec_dot.tick_params(axis='x', bottom=False)

    # Panel 5: histogram of pooled per-sector integrals (both groups)
    ax_hist = fig.add_subplot(gs[2, 2:4])
    _hist_data = {}
    for g in groups:
        _vals = sector_integrals[g]
        _vals = _vals[np.isfinite(_vals)]
        if norm_resp_hist and _vals.size >= 2:
            _sd = float(np.std(_vals))
            if _sd > 0:
                _vals = (_vals - float(np.mean(_vals))) / _sd
        _hist_data[g] = _vals

    _all_h = np.concatenate(
        [v for v in _hist_data.values() if v.size]) \
        if any(v.size for v in _hist_data.values()) else np.array([])
    if _all_h.size:
        _bins = np.linspace(float(np.nanmin(_all_h)),
                            float(np.nanmax(_all_h)), 51)
    else:
        _bins = 50

    for g in groups:
        _vals = _hist_data[g]
        if _vals.size == 0:
            continue
        ax_hist.hist(_vals, bins=_bins, histtype='stepfilled',
                     color=_GROUP_COLOURS[g],
                     edgecolor=_GROUP_COLOURS[g],
                     alpha=0.4, lw=1.0, density=True,
                     label=_GROUP_LABELS[g])
    _hist_xlab = ('sector activity (stim) (z)'
                  if norm_resp_hist else 'sector activity (stim)')
    ax_hist.set_xlabel(_hist_xlab)
    ax_hist.set_ylabel('probability density')
    ax_hist.legend(fontsize=7, frameon=False)
    for _side in ('top', 'right'):
        ax_hist.spines[_side].set_visible(False)

    fig.suptitle(
        f'plt_all_hemispherectrl  ({len(records)} recordings, '
        f'channel={channel})')

    _style_ctx.__exit__(None, None, None)

    if save_path is not None:
        _kw_parts = [
            f'layer={str(plt_layer).replace("/", "-")}',
            f'ch={channel}',
            f'sort={sort_by}',
            f'tpre={t_pre:g}',
            f'tpost={t_post:g}',
            f'dt={dt:g}',
            f'trew={t_rew:g}',
            f'vlim={sector_vlim:g}',
            f'norm={int(bool(norm_resp_hist))}',
        ]
        _kw_suffix = '_' + '_'.join(_kw_parts)
        _root, _ext = os.path.splitext(save_path)
        if not _ext:
            _ext = '.pdf'
        _full_save_path = f'{_root}{_kw_suffix}{_ext}'
        fig.savefig(_full_save_path, dpi=150)
        print(f'  saved figure -> {_full_save_path}', flush=True)
    if plt_show:
        plt.show()

    stats_out = SimpleNamespace(
        wf_integral=SimpleNamespace(
            t=t_wf_int, p=p_wf_int,
            grab5ht=np.asarray(wf_integrals['grab5ht']),
            grab5ht_mut=np.asarray(wf_integrals['grab5ht_mut'])),
        sector_mean_integral=SimpleNamespace(
            t=t_sec_int, p=p_sec_int,
            grab5ht=np.asarray(sec_mean_integrals['grab5ht']),
            grab5ht_mut=np.asarray(sec_mean_integrals['grab5ht_mut'])))

    return SimpleNamespace(
        fig=fig, records=records,
        t_common=t_common, wf=wf, sectors=all_sec,
        sector_integrals=sector_integrals,
        wf_integrals=wf_integrals,
        sec_mean_integrals=sec_mean_integrals,
        t_rew=_t_rew_plot,
        stats=stats_out)
