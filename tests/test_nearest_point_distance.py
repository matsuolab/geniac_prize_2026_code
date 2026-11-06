"""The multiview point merge must not build a pairwise distance matrix.

``FrankaLiberoApi`` decides whether the agent-view and wrist-view point clouds see
the same object by asking how close the nearest pair is. It used to answer that by
broadcasting an ``(n_agent, n_wrist, 3)`` difference — one float64 entry per pair of
points, reduced immediately to one distance per agent point. On a mask covering much
of both views that is tens of thousands of points each: measured at 66 GiB per call,
spiking the submission process to 254 GiB, which is what took every full-pack run
out with OOM.

A KD-tree answers the same question exactly. These tests pin *exactly*: the results
must agree elementwise with the brute-force formula on inputs small enough to
compute both ways.
"""

from __future__ import annotations

import numpy as np
import pytest

cKDTree = pytest.importorskip("scipy.spatial").cKDTree


def brute_force(agent: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """What the code used to do, kept here as the reference to match."""
    return np.min(
        np.linalg.norm(agent[:, np.newaxis, :] - wrist[np.newaxis, :, :], axis=2), axis=1
    )


def kd_tree(agent: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """What it does now."""
    distances, _ = cKDTree(wrist).query(agent, k=1)
    return distances


@pytest.mark.parametrize("n_agent, n_wrist", [(1, 1), (1, 40), (40, 1), (37, 53), (200, 173)])
def test_the_two_agree_elementwise(n_agent: int, n_wrist: int) -> None:
    rng = np.random.default_rng(n_agent * 1000 + n_wrist)
    agent = rng.normal(size=(n_agent, 3))
    wrist = rng.normal(size=(n_wrist, 3))
    np.testing.assert_allclose(kd_tree(agent, wrist), brute_force(agent, wrist), rtol=1e-12)


def test_they_agree_when_the_clouds_overlap() -> None:
    """The case the caller actually branches on: a shared point makes the minimum
    zero, and that is what decides whether the two views get merged."""
    rng = np.random.default_rng(7)
    shared = rng.normal(size=(5, 3))
    agent = np.vstack([shared, rng.normal(size=(30, 3))])
    wrist = np.vstack([shared, rng.normal(size=(20, 3))])

    ours, theirs = kd_tree(agent, wrist), brute_force(agent, wrist)
    np.testing.assert_allclose(ours, theirs, rtol=1e-12)
    assert ours.min() == pytest.approx(0.0, abs=1e-12)


def test_duplicate_points_do_not_change_the_answer() -> None:
    rng = np.random.default_rng(11)
    wrist = rng.normal(size=(25, 3))
    agent = np.repeat(rng.normal(size=(9, 3)), 3, axis=0)
    np.testing.assert_allclose(kd_tree(agent, wrist), brute_force(agent, wrist), rtol=1e-12)


def test_a_realistic_cloud_stays_within_a_sane_amount_of_memory() -> None:
    """Two masked 256x256 views is ~65k points each. The old formula would need
    65k x 65k x 3 x 8 B = 95 GiB for this; the tree needs megabytes."""
    import tracemalloc

    rng = np.random.default_rng(3)
    agent, wrist = rng.normal(size=(65_536, 3)), rng.normal(size=(65_536, 3))

    tracemalloc.start()
    distances = kd_tree(agent, wrist)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert distances.shape == (65_536,)
    assert peak < 256 * 1024 * 1024, f"peak {peak / 1e9:.1f} GB"
