"""端到端自测: FastAPI TestClient + 临时 SQLite。

运行:
    PYTHONUSERBASE=/tmp/pyuser python3 -m pytest tests/ -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import database
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("COVERAGE_DB", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def _point(ts: str, lat: float, lon: float) -> dict:
    return {"ts": ts, "lat": lat, "lon": lon}


# ---------------------------------------------------------------- 基本切分

def test_basic_day_and_grid_split(client: TestClient) -> None:
    """2026-11-20 23:30 -> 2026-11-21 00:30, 从 (10.2,10.2) 到 (11.8,11.8)。
    应同时按午夜、整 11° 经纬线切开, 西南边界归本格。"""
    r = client.put(
        "/voyages/V001/coverage",
        json={
            "points": [
                _point("2026-11-20T23:30:00Z", 10.2, 10.2),
                _point("2026-11-21T00:30:00Z", 11.8, 11.8),
            ]
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "replaced"
    # 午夜与 11° 经纬线在同一时刻 (f=0.5) 切到, 共 2 个分片:
    # 20 日 10/10 格 -> 21 日 11/11 格
    assert body["shard_count"] == 2

    # 起点所在格 (lat 10, lon 10) 只在 20 日有覆盖
    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": 10, "lon_cell": 10})
    intervals = r.json()["intervals"]
    assert len(intervals) == 1
    assert intervals[0]["start"] == "2026-11-20T23:30:00.000000Z"

    # 21 日 11/11 格存在覆盖, 终点时刻在其中
    r = client.get("/coverage", params={"day": "2026-11-21", "lat_cell": 11, "lon_cell": 11})
    intervals = r.json()["intervals"]
    assert len(intervals) == 1
    assert intervals[0]["end"] == "2026-11-21T00:30:00.000000Z"

    # 没有覆盖的格返回空
    r = client.get("/coverage", params={"day": "2026-11-21", "lat_cell": 0, "lon_cell": 0})
    assert r.json()["intervals"] == []


def test_west_south_boundary_belongs_to_cell(client: TestClient) -> None:
    """恰好落在整度线上的起点: 西/南边界归本格, 且零长度起点不产生空分片。"""
    r = client.put(
        "/voyages/V002/coverage",
        json={
            "points": [
                _point("2026-11-20T00:00:00Z", 10.0, 10.0),
                _point("2026-11-20T01:00:00Z", 10.5, 10.5),
            ]
        },
    )
    assert r.status_code == 200
    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": 10, "lon_cell": 10})
    assert len(r.json()["intervals"]) == 1


# ---------------------------------------------------------------- 跨 180°

def test_antimeridian_crossing_short_arc(client: TestClient) -> None:
    """179 -> -179 走短弧 (向东 2°), 跨 ±180 同一条经线, 两格各得 30 分钟。"""
    r = client.put(
        "/voyages/V003/coverage",
        json={
            "points": [
                _point("2026-11-20T00:00:00Z", 0.0, 179.0),
                _point("2026-11-20T01:00:00Z", 0.0, -179.0),
            ]
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["shard_count"] == 2

    east = client.get(
        "/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": 179}
    ).json()["intervals"]
    west = client.get(
        "/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": -180}
    ).json()["intervals"]
    assert len(east) == 1 and len(west) == 1
    assert east[0]["end"] == west[0]["start"] == "2026-11-20T00:30:00.000000Z"


def test_longitude_180_merges_with_minus_180(client: TestClient) -> None:
    """经度 180 的观测点与 -180 同线, 归入西边界 -180 的格。"""
    r = client.put(
        "/voyages/V004/coverage",
        json={
            "points": [
                _point("2026-11-20T00:00:00Z", 0.0, 180.0),
                _point("2026-11-20T00:30:00Z", 0.0, -179.5),
            ]
        },
    )
    assert r.status_code == 200
    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": -180})
    assert len(r.json()["intervals"]) == 1
    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": 179})
    assert r.json()["intervals"] == []


def test_antimeridian_reverse_direction(client: TestClient) -> None:
    """-179 -> 179 走短弧 (向西 2°), 同样切成 -180 格与 179 格两段。"""
    r = client.put(
        "/voyages/V003B/coverage",
        json={
            "points": [
                _point("2026-11-20T00:00:00Z", 0.0, -179.0),
                _point("2026-11-20T01:00:00Z", 0.0, 179.0),
            ]
        },
    )
    assert r.status_code == 200
    assert r.json()["shard_count"] == 2
    west = client.get(
        "/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": -180}
    ).json()["intervals"]
    east = client.get(
        "/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": 179}
    ).json()["intervals"]
    assert west[0]["end"] == east[0]["start"] == "2026-11-20T00:30:00.000000Z"
    assert west[0]["start"] == "2026-11-20T00:00:00.000000Z"
    assert east[0]["end"] == "2026-11-20T01:00:00.000000Z"


def test_exact_180_difference_rejected(client: TestClient) -> None:
    """航段两端短弧经差恰为 180° -> 整次拒收, 库里无该航次任何数据。"""
    r = client.put(
        "/voyages/V005/coverage",
        json={
            "points": [
                _point("2026-11-20T00:00:00Z", 0.0, 90.0),
                _point("2026-11-20T01:00:00Z", 0.0, -90.0),
            ]
        },
    )
    assert r.status_code == 400
    assert "180" in r.json()["detail"]
    assert client.get("/voyages").json()["rows"] == []


# ---------------------------------------------------------------- 输入校验

def test_non_increasing_timestamp_rejected(client: TestClient) -> None:
    r = client.put(
        "/voyages/V006/coverage",
        json={
            "points": [
                _point("2026-11-20T01:00:00Z", 0.0, 0.0),
                _point("2026-11-20T00:00:00Z", 1.0, 1.0),
            ]
        },
    )
    assert r.status_code == 400
    assert "递增" in r.json()["detail"]


def test_naive_timestamp_rejected(client: TestClient) -> None:
    r = client.put(
        "/voyages/V007/coverage",
        json={"points": [{"ts": "2026-11-20T00:00:00", "lat": 0, "lon": 0},
                         {"ts": "2026-11-20T01:00:00", "lat": 1, "lon": 1}]},
    )
    assert r.status_code == 422


def test_single_point_rejected(client: TestClient) -> None:
    r = client.put(
        "/voyages/V008/coverage",
        json={"points": [_point("2026-11-20T00:00:00Z", 0.0, 0.0)]},
    )
    assert r.status_code == 422


def test_out_of_range_coords_rejected(client: TestClient) -> None:
    r = client.put(
        "/voyages/V009/coverage",
        json={
            "points": [
                _point("2026-11-20T00:00:00Z", 0.0, 181.0),
                _point("2026-11-20T01:00:00Z", 0.0, 180.0),
            ]
        },
    )
    assert r.status_code == 422


# ---------------------------------------------------------------- 重交更正

def test_resubmit_replaces_all_old_coverage(client: TestClient) -> None:
    payload_v1 = {
        "points": [
            _point("2026-11-20T00:00:00Z", 0.0, 0.0),
            _point("2026-11-20T02:00:00Z", 0.0, 4.0),
        ]
    }
    assert client.put("/voyages/V010/coverage", json=payload_v1).status_code == 200
    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": 0})
    assert len(r.json()["intervals"]) == 1

    # 重交: 航线改到南半球, 旧格覆盖应全部消失
    payload_v2 = {
        "points": [
            _point("2026-11-20T00:00:00Z", -10.0, 0.0),
            _point("2026-11-20T02:00:00Z", -10.0, 4.0),
        ]
    }
    r = client.put("/voyages/V010/coverage", json=payload_v2)
    assert r.status_code == 200
    assert client.get(
        "/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": 0}
    ).json()["intervals"] == []
    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": -10, "lon_cell": 0})
    assert len(r.json()["intervals"]) == 1


def test_failed_resubmit_keeps_old_result(client: TestClient, monkeypatch) -> None:
    payload_v1 = {
        "points": [
            _point("2026-11-20T00:00:00Z", 0.0, 0.0),
            _point("2026-11-20T02:00:00Z", 0.0, 2.0),
        ]
    }
    client.put("/voyages/V011/coverage", json=payload_v1)

    # 非法重交: 计算阶段即失败, 旧数据完好
    bad = {
        "points": [
            _point("2026-11-20T00:00:00Z", 0.0, 0.0),
            _point("2026-11-20T02:00:00Z", 0.0, 180.0),  # 经差 180
        ]
    }
    assert client.put("/voyages/V011/coverage", json=bad).status_code == 400
    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": 0})
    assert len(r.json()["intervals"]) == 1

    # 保存阶段失败 (注入异常): 事务回滚, 旧数据同样完好
    real_replace = database.replace_voyage

    def boom(voyage_id, shards):  # noqa: ANN001
        raise RuntimeError("disk full")

    monkeypatch.setattr(database, "replace_voyage", boom)
    good = {
        "points": [
            _point("2026-11-20T00:00:00Z", 5.0, 5.0),
            _point("2026-11-20T02:00:00Z", 5.0, 7.0),
        ]
    }
    assert client.put("/voyages/V011/coverage", json=good).status_code == 500
    monkeypatch.setattr(database, "replace_voyage", real_replace)

    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": 0})
    assert len(r.json()["intervals"]) == 1
    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": 5, "lon_cell": 5})
    assert r.json()["intervals"] == []


# ---------------------------------------------------------------- 多航次与合并

def test_multiple_voyages_and_merge(client: TestClient) -> None:
    # 航次 A: 00:00-01:00 在格 (0,0)
    client.put(
        "/voyages/VA/coverage",
        json={
            "points": [
                _point("2026-11-20T00:00:00Z", 0.2, 0.2),
                _point("2026-11-20T01:00:00Z", 0.4, 0.4),
            ]
        },
    )
    # 航次 B: 00:30-02:00 同格, 与 A 重叠; 02:00 恰好抵达 1° 边界驶出,
    # 返航段在 04:29:20 越过 1° 线重新进入本格, 形成独立时段
    # (西/南边界归相邻格, 边界点本身不属本格)
    client.put(
        "/voyages/VB/coverage",
        json={
            "points": [
                _point("2026-11-20T00:30:00Z", 0.2, 0.2),
                _point("2026-11-20T02:00:00Z", 1.0, 1.0),
                _point("2026-11-20T03:00:00Z", 5.2, 5.2),
                _point("2026-11-20T04:00:00Z", 5.4, 5.4),
                _point("2026-11-20T04:30:00Z", 0.9, 0.9),
                _point("2026-11-20T05:00:00Z", 0.2, 0.2),
            ]
        },
    )
    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": 0})
    intervals = r.json()["intervals"]
    assert intervals == [
        {"start": "2026-11-20T00:00:00.000000Z", "end": "2026-11-20T02:00:00.000000Z"},
        {"start": "2026-11-20T04:29:20.000000Z", "end": "2026-11-20T05:00:00.000000Z"},
    ]

    # 替换航次 A 不影响航次 B
    client.put(
        "/voyages/VA/coverage",
        json={
            "points": [
                _point("2026-11-20T00:00:00Z", 50.0, 50.0),
                _point("2026-11-20T01:00:00Z", 50.4, 50.4),
            ]
        },
    )
    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": 0})
    assert r.json()["intervals"] == [
        {"start": "2026-11-20T00:30:00.000000Z", "end": "2026-11-20T02:00:00.000000Z"},
        {"start": "2026-11-20T04:29:20.000000Z", "end": "2026-11-20T05:00:00.000000Z"},
    ]


def test_query_uses_committed_snapshot(client: TestClient) -> None:
    """查询只取已提交状态: 未提交事务里的写入对查询不可见。"""
    client.put(
        "/voyages/VC/coverage",
        json={
            "points": [
                _point("2026-11-20T00:00:00Z", 0.2, 0.2),
                _point("2026-11-20T01:00:00Z", 0.4, 0.4),
            ]
        },
    )
    import sqlite3

    conn = sqlite3.connect(database._db_path())
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO coverage "
            "(voyage_id, day, lat_cell, lon_cell, start_epoch, end_epoch) "
            "VALUES ('UNCOMMITTED','2026-11-20',0,0,0,1)"
        )
        r = client.get(
            "/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": 0}
        )
        # 未提交的行不可见; 且 WAL 下读不被写阻塞
        assert all(i["start"].startswith("2026-11-20T00:00") for i in r.json()["intervals"])
        conn.rollback()
    finally:
        conn.close()


def test_southward_boundary_labeling(client: TestClient) -> None:
    """朝南运动且起点压整度纬线: 11 -> 9 (1 小时), 中点定格保证两段归属正确。"""
    r = client.put(
        "/voyages/V002B/coverage",
        json={
            "points": [
                _point("2026-11-20T00:00:00Z", 11.0, 0.0),
                _point("2026-11-20T01:00:00Z", 9.0, 0.0),
            ]
        },
    )
    assert r.status_code == 200
    assert r.json()["shard_count"] == 2
    north = client.get(
        "/coverage", params={"day": "2026-11-20", "lat_cell": 10, "lon_cell": 0}
    ).json()["intervals"]
    south = client.get(
        "/coverage", params={"day": "2026-11-20", "lat_cell": 9, "lon_cell": 0}
    ).json()["intervals"]
    assert north == [
        {"start": "2026-11-20T00:00:00.000000Z", "end": "2026-11-20T00:30:00.000000Z"}
    ]
    assert south == [
        {"start": "2026-11-20T00:30:00.000000Z", "end": "2026-11-20T01:00:00.000000Z"}
    ]


def test_invalid_query_params_rejected(client: TestClient) -> None:
    # 错误日期格式
    r = client.get("/coverage", params={"day": "2026/11/20", "lat_cell": 0, "lon_cell": 0})
    assert r.status_code == 422
    # 格号越界
    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": 90, "lon_cell": 0})
    assert r.status_code == 422
    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": 0, "lon_cell": 180})
    assert r.status_code == 422


def test_docs_page_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_pole_cell_clamped(client: TestClient) -> None:
    """北纬 90 极点夹入最北格 [89,90], 不产生越界格号。"""
    r = client.put(
        "/voyages/VD/coverage",
        json={
            "points": [
                _point("2026-11-20T00:00:00Z", 89.6, 0.0),
                _point("2026-11-20T01:00:00Z", 90.0, 0.0),
            ]
        },
    )
    assert r.status_code == 200
    r = client.get("/coverage", params={"day": "2026-11-20", "lat_cell": 89, "lon_cell": 0})
    assert len(r.json()["intervals"]) == 1
    r = client.get(
        "/coverage", params={"day": "2026-11-20", "lat_cell": 88, "lon_cell": 0}
    )
    # 89° 纬线以南的格在该段只占 89.6 之前的部分, 本段起点已在 89 格内, 故 88 格为空
    assert r.json()["intervals"] == []
