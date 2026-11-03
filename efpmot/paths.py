# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 In Ho Jeung
"""Path configuration: every file-system location comes from configs/paths.yaml, never from the code.

Lookup order of the configuration file:
  1. the file named by the environment variable EFPMOT_PATHS,
  2. <repo>/configs/paths.yaml,
  3. <repo>/configs/paths.example.yaml (defaults relative to the repository; a warning is printed once).
Relative paths are resolved against the repository root. Keys are dotted, e.g. get("data.dsec_det").

    from efpmot import paths
    root = paths.require("data.dsec_det")         # absolute path, error if the key is null
    runs = paths.out("runs", "dsec_v8n_416s3x_s0")  # <outputs.runs>/dsec_v8n_416s3x_s0, created on demand
"""
from __future__ import annotations

import os
import warnings
from typing import Any

import yaml

ENV_VAR = "EFPMOT_PATHS"
_CFG: dict | None = None
_CFG_FILE: str | None = None


def repo_root() -> str:
    """the repository root (the folder that holds configs/ and efpmot/)"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_file() -> str:
    """the configuration file that load() reads (see the module docstring for the lookup order)"""
    env = os.environ.get(ENV_VAR)
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(f"{ENV_VAR}={env}: no such file")
        return os.path.abspath(env)
    local = os.path.join(repo_root(), "configs", "paths.yaml")
    if os.path.isfile(local):
        return local
    return os.path.join(repo_root(), "configs", "paths.example.yaml")


def load(path: str | None = None, reload: bool = False) -> dict:
    """parsed configuration (cached). path: read this file instead of the lookup order (also cached)."""
    global _CFG, _CFG_FILE
    if _CFG is not None and not reload and (path is None or os.path.abspath(path) == _CFG_FILE):
        return _CFG
    f = os.path.abspath(path) if path else config_file()
    if path is None and os.path.basename(f) == "paths.example.yaml" and not os.environ.get(ENV_VAR):
        warnings.warn("configs/paths.yaml not found: using configs/paths.example.yaml (copy and edit it)", stacklevel=2)
    with open(f, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{f}: top level must be a mapping")
    _CFG, _CFG_FILE = cfg, f
    return cfg


def resolve(p: str | None) -> str | None:
    """expand ~ and environment variables; relative -> absolute against the repository root; None stays None"""
    if p is None:
        return None
    p = os.path.expandvars(os.path.expanduser(str(p)))
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(repo_root(), p))


def raw(key: str, default: Any = None) -> Any:
    """the unresolved value of a dotted key (default if any part is missing)"""
    node: Any = load()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def get(key: str, default: str | None = None) -> str | None:
    """absolute path of a dotted key, None if the key is null or missing (and no default is given)"""
    v = raw(key, None)
    return resolve(v) if v is not None else resolve(default)


def require(key: str) -> str:
    """absolute path of a dotted key; raises KeyError naming the config file if it is null or missing"""
    v = get(key)
    if v is None:
        raise KeyError(f"path '{key}' is not set in {_CFG_FILE or config_file()} (see configs/paths.example.yaml)")
    return v


def out(kind: str, *parts: str, make: bool = True) -> str:
    """<outputs.kind>/<parts...> (kind: derived | runs | cache | results); the directory is created when make"""
    d = os.path.join(require(f"outputs.{kind}"), *parts)
    if make:
        os.makedirs(d, exist_ok=True)
    return d


def first_dir(*candidates: str | None) -> str | None:
    """first candidate that is an existing directory (None entries skipped), else the last non-None candidate"""
    last = None
    for c in candidates:
        if c is None:
            continue
        last = c
        if os.path.isdir(c):
            return c
    return last
