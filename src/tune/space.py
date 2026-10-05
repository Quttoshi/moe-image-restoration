"""Turn a YAML search-space description into Optuna suggestions."""
from __future__ import annotations

import optuna


def suggest(trial: optuna.Trial, space: dict) -> dict:
    hp = {}
    for name, s in space.items():
        if s["type"] == "float":
            hp[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
        elif s["type"] == "int":
            hp[name] = trial.suggest_int(name, s["low"], s["high"], log=s.get("log", False))
        elif s["type"] == "categorical":
            hp[name] = trial.suggest_categorical(name, s["choices"])
        else:
            raise ValueError(f"unknown search-space type {s['type']!r} for {name}")
    return hp


def best_or_defaults(study_name: str, storage: str, defaults: dict) -> tuple[dict, bool]:
    """Best params of a finished study, else the config defaults."""
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        return dict(defaults) | study.best_params, True
    except (KeyError, ValueError):
        return dict(defaults), False


def remaining_trials(study: optuna.Study, n_trials: int) -> int:
    """Make a resumed study safe: fail trials left RUNNING by a killed process and
    return how many trials are still needed to reach ``n_trials`` finished ones."""
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
    done = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)))
    return max(0, n_trials - done)
