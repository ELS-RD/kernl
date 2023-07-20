from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TritonMeasure:
    two_tiles: bool
    sm: int
    disc: Optional[float] = None
    triton_ms: Optional[float] = None


@dataclass
class Measure:
    m: int
    n: int
    k: int
    triton: List[TritonMeasure]
    pytorch_ms: Optional[float] = None
    speedup: Optional[float] = None

    def number_of_tiles(self, blk_m: int, blk_n: int) -> int:
        return (self.m // blk_m) * (self.n // blk_n)

    def iter_per_tile(self, blk_k: int) -> int:
        return self.k // blk_k

    def get_minimum_triton_measure(self) -> TritonMeasure:
        return min(self.triton, key=lambda x: x.triton_ms)


def get_timings(measures: List[TritonMeasure]) -> List[float]:
    xp_timings = list()
    for triton in measures:
        xp_timings.append(triton.triton_ms)
    return xp_timings


def get_features(measures: List[TritonMeasure], total_tiles: int, iters_per_tile: int) -> List[List[float]]:
    xp_features = list()
    for triton in measures:
        total_programs_streamk = triton.sm
        total_tiles_streamk = total_tiles % total_programs_streamk
        # for two-tile Stream-K + data-parallel from original paper
        if triton.two_tiles and total_tiles - total_tiles_streamk > total_programs_streamk:
            total_tiles_streamk += total_programs_streamk
        # remaining tiles are computed using classical blocking
        total_blocking_tiles = total_tiles - total_tiles_streamk
        total_iters_streamk = total_tiles_streamk * iters_per_tile

        # values used for prediction
        nb_sync_stream_k = triton.sm  # there is 2 syncs per SM in stream-k
        nb_store = total_blocking_tiles  # there is 1 store per tile in blocking loop
        nb_iter_stream_k = total_iters_streamk  # includes loading
        nb_iter_blocking = total_blocking_tiles * iters_per_tile  # includes loading

        xp_features.append([nb_sync_stream_k, nb_iter_stream_k, nb_iter_blocking, nb_store])

    return xp_features
