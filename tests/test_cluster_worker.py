"""Dimensionnement local du worker cluster.

Ces tests restent hors réseau et hors self-play : ils valident seulement la logique
qui transforme CLI/env/job serveur en parallélisme effectif.
"""

from __future__ import annotations

import argparse

import pytest

from sanchess.cluster import protocol as P
from sanchess.cluster import worker


def test_auto_workers_use_all_but_reserved_cores():
    assert worker._resolve_workers("auto", reserve_cores=1, cpu_count=16) == 15


def test_auto_workers_keep_at_least_one_worker():
    assert worker._resolve_workers("auto", reserve_cores=8, cpu_count=2) == 1


def test_explicit_workers_are_respected():
    assert worker._resolve_workers("4", reserve_cores=1, cpu_count=16) == 4


def test_auto_threads_default_to_one_per_worker():
    assert worker._resolve_threads("auto") == 1


def test_invalid_worker_count_is_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        worker._resolve_workers("0", reserve_cores=1, cpu_count=16)


def test_cuda_gpu_games_are_split_across_processes():
    args = argparse.Namespace(workers=8, gpu_games_total=None,
                              gpu_leaves_per_game=None)
    job = P.JobSpec(gpu_games=64, gpu_leaves_per_game=8)
    out = worker._apply_local_job_overrides(job, args, "cuda")
    assert out.gpu_games == 8
    assert out.gpu_leaves_per_game == 8


def test_cuda_gpu_games_and_leaves_can_be_overridden_locally():
    args = argparse.Namespace(workers=6, gpu_games_total=512,
                              gpu_leaves_per_game=12)
    job = P.JobSpec(gpu_games=64, gpu_leaves_per_game=8)
    out = worker._apply_local_job_overrides(job, args, "cuda")
    assert out.gpu_games == 86
    assert out.gpu_leaves_per_game == 12


def test_cpu_job_keeps_server_gpu_budget_untouched():
    args = argparse.Namespace(workers=8, gpu_games_total=512,
                              gpu_leaves_per_game=12)
    job = P.JobSpec(gpu_games=64, gpu_leaves_per_game=8)
    out = worker._apply_local_job_overrides(job, args, "cpu")
    assert out.gpu_games == 64
    assert out.gpu_leaves_per_game == 8
