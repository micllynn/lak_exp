"""
batch_run.py  –  Utilities for batch-loading and processing Marko-style
2-photon experiments (rec_type='paqio').

Usage example
-------------
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
    'MBL002_2025-02-05': dict(
        enclosing_folder='/path/to/MBL002/2025-02-05/',
        folder_beh='2',
        folder_img='TwoP/2025-02-05_t-001',
        fname_img='2025-02-05_t-001_Cycle00001_Ch2.tif',
        beh_type='auditory_pavlov',
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
"""

import traceback
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
