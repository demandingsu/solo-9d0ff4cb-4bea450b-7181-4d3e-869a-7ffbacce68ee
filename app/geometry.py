"""走航轨迹 -> (UTC 日, 整度经纬格) 覆盖分片的切分逻辑。

规则:
- 相邻观测点之间视为持续观测, 位置随时间线性变化。
- 经度走短弧; 若某航段短弧经差恰为 180°, 整次提交拒收。
- 按 UTC 午夜与整度经线/纬线切分; 西、南边界归本格。
- ±180° 视为同一条经线 (经度 180 归入西边界为 -180 的格)。
- 只保留有时长的分片。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

DAY_SECONDS = 86400.0
# 同一切点的浮点容差 (占整段时长的比例)
FRACTION_EPS = 1e-12
# 合并/比较时刻时的浮点容差 (秒)
TIME_EPS = 1e-9


class TrackValidationError(ValueError):
    """轨迹数据不合法 (对应 HTTP 400)。"""


@dataclass(frozen=True)
class InputPoint:
    ts: datetime  # 已转为 UTC 的 aware datetime
    lat: float
    lon: float


@dataclass(frozen=True)
class Shard:
    day: str  # YYYY-MM-DD (UTC)
    lat_cell: int  # 格的南边界纬度, [-90, 89]
    lon_cell: int  # 格的西边界经度, [-180, 179]
    start_epoch: float
    end_epoch: float


def _wrap_longitude_delta(delta: float) -> float:
    """把经差归一化到短弧 (-180, 180]。

    取模本身落在 [-180, 180); -180 与 +180 是同一条弧,
    统一映射到 +180 以配合 "恰为 180° 拒收" 的判定。
    """
    wrapped = (delta + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _integer_boundary_fractions(a: float, b: float) -> list[float]:
    """开区间 (a, b) 内所有整数值对应的线性比例 f=(k-a)/(b-a)。"""
    if b == a:
        return []
    lo, hi = min(a, b), max(a, b)
    k_first = math.floor(lo) + 1
    k_last = math.ceil(hi) - 1
    return [(k - a) / (b - a) for k in range(k_first, k_last + 1)]


def _midnight_fractions(t0: float, t1: float, dt: float) -> list[float]:
    """开时间区间 (t0, t1) 内所有 UTC 午夜对应的线性比例。"""
    first_midnight = (math.floor(t0 / DAY_SECONDS) + 1) * DAY_SECONDS
    out: list[float] = []
    m = first_midnight
    while m < t1:
        out.append((m - t0) / dt)
        m += DAY_SECONDS
    return out


def _lon_cell(lon: float) -> int:
    """经度 -> 格西边界索引; ±180 同线, 180 归入 [-180,-179) 格。"""
    wrapped = (lon + 180.0) % 360.0 - 180.0
    return math.floor(wrapped)


def _lat_cell(lat: float) -> int:
    """纬度 -> 格南边界索引; 极点 90 夹到最北一格 [89,90]。"""
    return max(-90, min(89, math.floor(lat)))


def split_track(points: list[InputPoint]) -> list[Shard]:
    """把整条轨迹切分为带日/格归属的覆盖分片。

    任一校验失败抛 TrackValidationError, 调用方不得写入任何数据。
    """
    if len(points) < 2:
        raise TrackValidationError("至少需要 2 个观测点才能构成走航观测")

    shards: list[Shard] = []

    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]

        t0 = p0.ts.timestamp()
        t1 = p1.ts.timestamp()
        dt = t1 - t0
        if dt <= 0:
            raise TrackValidationError(
                f"第 {i + 1}、{i + 2} 个点的 UTC 时刻不是严格递增"
            )

        dlon = _wrap_longitude_delta(p1.lon - p0.lon)
        if abs(dlon) == 180.0:
            raise TrackValidationError(
                f"第 {i + 1}->{i + 2} 点航段短弧经差恰为 180°, 整次提交拒收"
            )

        # 短弧展开后的经度 (可能落在 [-180,180] 之外, 仅用于线性插值)
        u0, u1 = p0.lon, p0.lon + dlon
        lat0, lat1 = p0.lat, p1.lat

        fractions = {0.0, 1.0}
        fractions.update(_integer_boundary_fractions(u0, u1))
        fractions.update(_integer_boundary_fractions(lat0, lat1))
        fractions.update(_midnight_fractions(t0, t1, dt))

        ordered = sorted(fractions)
        # 合并浮点意义上重合的切点
        unique: list[float] = []
        for f in ordered:
            if not unique or f - unique[-1] > FRACTION_EPS:
                unique.append(f)

        for fa, fb in zip(unique, unique[1:]):
            start = t0 + fa * dt
            end = t0 + fb * dt
            if end - start <= TIME_EPS:
                continue  # 只保留有时长的分片

            # 切点保证开区间 (fa, fb) 内不再有任何整度边界,
            # 故用中点定格: 朝西/南运动且起点压线时, 分片内部实际属于另一侧的格。
            fm = (fa + fb) / 2.0
            lon_mid = u0 + fm * dlon
            lat_mid = lat0 + fm * (lat1 - lat0)
            day = datetime.fromtimestamp(start, tz=timezone.utc).date().isoformat()
            shards.append(
                Shard(
                    day=day,
                    lat_cell=_lat_cell(lat_mid),
                    lon_cell=_lon_cell(lon_mid),
                    start_epoch=start,
                    end_epoch=end,
                )
            )

    return shards


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """合并同一日/格内重叠或首尾相接的时段。"""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: x[0])
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + TIME_EPS:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]
