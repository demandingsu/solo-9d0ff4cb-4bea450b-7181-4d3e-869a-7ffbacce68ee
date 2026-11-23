"""走航覆盖服务 FastAPI 入口。

端点:
- PUT  /voyages/{voyage_id}/coverage  上传/重交航次走航点列 (整船覆盖原子替换)
- GET  /coverage                       查询某日某整度格的覆盖时段 (合并后)
- GET  /voyages                        辅助: 查看已存航次分片概况
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field, field_validator

from . import database
from .geometry import InputPoint, TrackValidationError, split_track


@asynccontextmanager
async def lifespan(_app: FastAPI):
    database.init_db()
    yield


app = FastAPI(
    title="走航观测覆盖服务",
    description=(
        "接收航次走航点列, 按 UTC 日与整度经纬格切分覆盖并存入 SQLite; "
        "支持按日/格查询合并后的观测时段, 用于与卫星记录对照。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PointIn(BaseModel):
    ts: datetime = Field(..., description="观测时刻, 必须带时区 (建议直接用 UTC, 如 2026-11-20T00:00:00Z)")
    lat: float = Field(..., ge=-90.0, le=90.0, description="纬度, [-90, 90]")
    lon: float = Field(..., ge=-180.0, le=180.0, description="经度, [-180, 180]; ±180 视为同一经线")

    @field_validator("ts")
    @classmethod
    def _ts_must_be_aware_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("时刻必须带时区信息 (例如 2026-11-20T00:00:00Z)")
        return v.astimezone(timezone.utc)


class VoyageUpload(BaseModel):
    points: list[PointIn] = Field(..., min_length=2, description="带严格递增 UTC 时刻的经纬度点列, 至少 2 个点")


class IntervalOut(BaseModel):
    start: str
    end: str


class CoverageResponse(BaseModel):
    day: str
    lat_cell: int
    lon_cell: int
    merged: bool
    intervals: list[IntervalOut]


class UploadResponse(BaseModel):
    voyage_id: str
    status: Literal["replaced"]
    shard_count: int
    day_cell_count: int


def _epoch_to_iso(epoch: float) -> str:
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@app.put(
    "/voyages/{voyage_id}/coverage",
    response_model=UploadResponse,
    summary="上传或重交航次走航点列",
)
def upload_coverage(
    voyage_id: str = Path(..., min_length=1, description="航次号"),
    payload: VoyageUpload = ...,
) -> UploadResponse:
    points = [InputPoint(ts=p.ts, lat=p.lat, lon=p.lon) for p in payload.points]

    # 先完整计算; 计算失败时不会触碰数据库, 旧结果照常可查
    try:
        shards = split_track(points)
    except TrackValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 单事务删除旧分片 + 插入新分片; 保存失败整体回滚
    try:
        written = database.replace_voyage(voyage_id, shards)
    except Exception as exc:  # pragma: no cover - 防御性
        raise HTTPException(status_code=500, detail=f"保存失败, 旧覆盖保持不变: {exc}") from exc

    return UploadResponse(
        voyage_id=voyage_id,
        status="replaced",
        shard_count=written,
        day_cell_count=len({(s.day, s.lat_cell, s.lon_cell) for s in shards}),
    )


@app.get("/coverage", response_model=CoverageResponse, summary="查询某日某整度格的覆盖时段")
def query_coverage(
    day: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="UTC 日期 YYYY-MM-DD"),
    lat_cell: int = Query(..., ge=-90, le=89, description="格的南边界纬度, 范围 -90..89"),
    lon_cell: int = Query(..., ge=-180, le=179, description="格的西边界经度, 范围 -180..179"),
) -> CoverageResponse:
    try:
        datetime.strptime(day, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"非法日期: {day}") from exc

    intervals = database.query_coverage(day, lat_cell, lon_cell)
    return CoverageResponse(
        day=day,
        lat_cell=lat_cell,
        lon_cell=lon_cell,
        merged=True,
        intervals=[
            IntervalOut(start=_epoch_to_iso(s), end=_epoch_to_iso(e)) for s, e in intervals
        ],
    )


@app.get("/voyages", summary="已存航次分片概况 (辅助查看)")
def voyages() -> dict:
    return {"rows": database.list_voyages()}
