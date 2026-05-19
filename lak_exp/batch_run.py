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
    ``figs_mbl/`` folder by ``plt_qc`` and draws a 4-panel cohort
    figure (whole-frame averages, per-sector heatmaps, and paired
    Pearson / Spearman dot plots).

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


def _find_latest_response_mean_npy(figs_dir, beh_folder=None):
    """Return the most-recently-modified _qc_response_mean*.npy in figs_dir.

    Parameters
    ----------
    figs_dir : str
        Path to a recording's figs_mbl/ folder.
    beh_folder : str or None
        If given, only consider files whose embedded behaviour folder
        matches (so multi-beh enclosing folders pick the right file).

    Returns
    -------
    fpath : str or None
    suffix : str or None
        The suffix fragment of the chosen filename (the part after
        '_qc_response_mean' and before '.npy'). Used to verify that
        kwargs are consistent across recordings.
    """
    if not os.path.isdir(figs_dir):
        return None, None

    candidates = []
    for entry in os.listdir(figs_dir):
        m = _NPY_PATTERN_RE.match(entry)
        if m is None:
            continue
        if beh_folder is not None and m.group('beh') != str(beh_folder):
            continue
        fpath = os.path.join(figs_dir, entry)
        candidates.append((os.path.getmtime(fpath), fpath, m.group('suffix')))

    if not candidates:
        return None, None

    candidates.sort(reverse=True)
    _, fpath, suffix = candidates[0]
    return fpath, suffix


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


def _collect_response_mean_records(csv_path, data_folder, verbose=True,
                                   plt_layer='all'):
    """Read the csv and load the latest response_mean .npy per row.

    Returns
    -------
    records : list of SimpleNamespace
        One entry per successfully-loaded recording, with .expref,
        .fpath, .suffix, .summary.
    suffix_counts : dict
        {suffix: count} of all observed filename suffixes; used to
        check kwarg consistency across recordings.
    """
    filelist = pd.read_csv(csv_path)
    records = []
    suffix_counts = {}

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

        figs_dir = os.path.join(data_folder, animal, date, 'figs_mbl')
        try:
            fpath, suffix = _find_latest_response_mean_npy(
                figs_dir, beh_folder=folder_beh)
        except Exception as e:
            if verbose:
                print(f'  x {expref}: error scanning {figs_dir} ({e})')
            continue
        if fpath is None:
            if verbose:
                print(f'  x {expref}: no _qc_response_mean*.npy in '
                      f'{figs_dir}')
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


def plt_grab_ctrl_dualcolour(csv_path,
            data_folder='/Volumes/T7/5HTCtx',
            sort_by='red_corr',
            sector_corr_metric='pearson',
            t_pre=2.0,
            t_post=6.0,
            dt=0.05,
            t_rew=2.0,
            sector_vlim=5.0,
            norm_resp_hist=True,
            plt_layer='all',
            figsize=(6.86, 5),
            save_path=None,
            plt_show=True):
    """Across-recording summary plot of QC response_mean outputs.

    For each row in the csv, loads the most recently saved
    `_qc_response_mean*.npy` from ``{data_folder}/{animal}/{date}/
    figs_mbl/`` (as written by ``_plt_qc_response_mean``), verifies that
    all loaded files share the same kwarg-suffix, and draws a 4-panel
    figure summarising the cohort.

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

    Parameters
    ----------
    csv_path : str
        Path to a filelist csv with at least an 'expref' column (see
        ``bulk_run_twop_correction``).
    data_folder : str
        Root data folder; expref maps to
        ``{data_folder}/{animal}/{date}/figs_mbl/``.
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
    sector_vlim : float
        Symmetric colour-scale limit for the sector heatmaps
        (vmin = -sector_vlim, vmax = +sector_vlim). Default 5.0.
    plt_layer : str
        Filter csv rows by the ``dual2p_folder`` tag.
        'all' (default) keeps every row;
        '2/3' keeps only canonical layer 2/3 recordings (t001 / t-001);
        '1'   keeps only layer 1 recordings (t002 / t-002 / t003 /
              t-003).
    norm_resp_hist : bool
        If True (default), z-score each channel's per-sector
        integral distribution before plotting the histogram in
        panel 5 so red and red_corr can be compared on a common
        scale. If False, plot the raw integral values.
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
    if sector_corr_metric not in ('pearson', 'spearman'):
        raise ValueError(
            f'sector_corr_metric must be "pearson" or "spearman", got '
            f'{sector_corr_metric!r}')

    _print_banner('PLT_ALL')
    print(f'  csv         : {csv_path}')
    print(f'  data_folder : {data_folder}')
    print(f'  sort_by     : {sort_by}')
    print(f'  sec metric  : {sector_corr_metric}')
    print(f'  plt_layer   : {plt_layer!r}', flush=True)

    records, suffix_counts = _collect_response_mean_records(
        csv_path, data_folder, verbose=True, plt_layer=plt_layer)
    if not records:
        raise RuntimeError(
            'No _qc_response_mean*.npy files were loaded; cannot plot.')

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

        # Panel 3 scalars: whole-frame Pearson from saved stats
        stats = summary.get('stats', {})
        wf_p = stats.get('whole_frame_pearson', {})
        rg = wf_p.get('red_vs_grn', {})
        pear_rg_wf.append(float(rg.get('r', np.nan)))
        if corr_key is not None:
            cg = wf_p.get(f'{corr_key}_vs_grn', {})
            pear_cg_wf.append(float(cg.get('r', np.nan)))
        else:
            pear_cg_wf.append(np.nan)

        # Panel 4 scalars: per-sector pearson/spearman from saved stats
        sec_stats = stats.get(_stat_key, {})
        rg_sec = sec_stats.get('red_vs_grn', {})
        sec_corr_rg.append(float(rg_sec.get(_stat_val_key, np.nan)))
        if corr_key is not None:
            cg_sec = sec_stats.get(f'{corr_key}_vs_grn', {})
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
    gs = gridspec.GridSpec(
        3, 6, figure=fig,
        height_ratios=[1.0, 2.0, 1.2])

    # Panel 1: whole-frame trial-averaged traces (3 columns)
    axes_wf = [fig.add_subplot(gs[0, 2 * i:2 * i + 2])
               for i in range(3)]
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
        for _side in ('top', 'right'):
            ax.spines[_side].set_visible(False)

    # Panel 2: per-sector heatmaps (3 columns, shared sorting).
    # Fixed symmetric colour scale (vmin=-sector_vlim, vmax=+sector_vlim);
    # only the rightmost panel carries a colorbar since all three share
    # the same scale.
    axes_sec = [fig.add_subplot(gs[1, 2 * i:2 * i + 2])
                for i in range(3)]
    _vmax = float(sector_vlim)
    _vmin = -_vmax

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

    fig.suptitle(
        f'plt_grab_ctrl_dualcolour  ({len(records)} recordings, '
        f'sort_by={sort_by}, sec={sector_corr_metric})')

    # Close the style context now that the figure is fully built;
    # any later show/save calls render with the style baked in.
    _style_ctx.__exit__(None, None, None)

    if save_path is not None:
        # Canonical filename: fig_dualcolour_grab_<kwargs>.<ext>, so the
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
            f'vlim={sector_vlim:g}',
            f'norm={int(bool(norm_resp_hist))}',
        ]
        _kw_suffix = '_'.join(_kw_parts)
        _root, _ext = os.path.splitext(save_path)
        if _ext:
            _save_dir = os.path.dirname(save_path) or '.'
        else:
            _save_dir = save_path
            _ext = '.pdf'
        _fname = f'fig_dualcolour_grab_{_kw_suffix}{_ext}'
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
        sector_integrals=_integrals,
        t_rew=_t_rew_plot,
        stats=stats_out)


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

        figs_dir = os.path.join(data_folder, animal, date, 'figs_mbl')
        try:
            fpath, suffix = _find_latest_response_mean_npy(
                figs_dir, beh_folder=folder_beh)
        except Exception as e:
            if verbose:
                print(f'  x {expref}: error scanning {figs_dir} ({e})')
            continue
        if fpath is None:
            if verbose:
                if not os.path.isdir(figs_dir):
                    print(f'  x {expref}: figs_mbl dir does not '
                          f'exist: {figs_dir}')
                else:
                    _other = [e for e in os.listdir(figs_dir)
                              if '_qc_response_mean' in e
                              and e.endswith('.npy')]
                    if _other:
                        print(f'  x {expref}: no _qc_response_mean*.npy '
                              f'matching beh={folder_beh!r} in '
                              f'{figs_dir}; found {len(_other)} '
                              f'other(s) for different beh folders: '
                              f'{_other[:3]}')
                    else:
                        print(f'  x {expref}: no _qc_response_mean*.npy '
                              f'in {figs_dir}')
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
