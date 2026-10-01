"""照度上报批次更正（作废）：仅作废收据中实际插入且仍归属该档案的读数。

覆盖：
- 作废立即从剂量查询 / 采集质量诊断 / 读数列表排除，并按剩余时间线重算；
- 收据内容、序号、链哈希不变，查询收据标出走废状态；
- 重复 request_key 重放原结果，换档案 / 收据 / 原因 409；同收据二次作废 409；
- 收据链断链/篡改、归并状态变化（来源旧链）、读数归属/内容变化 → 409 无部分变更；
- 只作废 inserted 读数（duplicate/conflict 不算）；整次在单事务内幂等落库。
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

BASE = 1_760_000_000


def iso(offset: float) -> str:
    return datetime.fromtimestamp(BASE + offset, tz=timezone.utc).isoformat()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    from app.main import app

    with TestClient(app) as c:
        yield c, db_file


def make_exhibit(client: TestClient, eid: str | None = None,
                 limit: float = 1000.0, interval: float = 3600.0) -> str:
    body = {"name": "唐代绢画", "dose_limit_luxh": limit, "max_interval_seconds": interval}
    if eid is not None:
        body["id"] = eid
    resp = client.post("/exhibits", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["canonical_id"]


def send(client: TestClient, exhibit_id: str, readings: list[dict]) -> dict:
    resp = client.post(f"/exhibits/{exhibit_id}/readings", json={"readings": readings})
    assert resp.status_code == 200, resp.text
    return resp.json()


def correct(client: TestClient, exhibit_id: str, receipt_id: str,
            request_key: str, reason: str = "传感器故障批次作废"):
    return client.post(
        f"/exhibits/{exhibit_id}/readings/correct",
        json={"receipt_id": receipt_id, "request_key": request_key, "reason": reason},
    )


def reading_ids(client: TestClient, exhibit_id: str) -> list[str]:
    return [r["id"] for r in client.get(f"/exhibits/{exhibit_id}/readings").json()["readings"]]


# ---------- 基本作废语义 ----------

def test_correct_voids_only_inserted_readings_and_marks_receipt(env):
    client, _ = env
    eid = make_exhibit(client)
    first = send(client, eid, [
        {"id": "a", "ts": iso(0), "lux": 100},
        {"id": "b", "ts": iso(3600), "lux": 100},
    ])
    second = send(client, eid, [
        {"id": "c", "ts": iso(7200), "lux": 100},      # inserted
        {"id": "a", "ts": iso(0), "lux": 100},          # duplicate
        {"id": "b", "ts": iso(3600), "lux": 999},       # conflict
    ])
    assert reading_ids(client, eid) == ["a", "b", "c"]

    resp = correct(client, eid, first["receipt"]["receipt_id"], "key-1")
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["already_committed"] is False
    assert out["voided_count"] == 2
    assert out["voided_reading_ids"] == ["a", "b"]
    assert out["chain_id"] == out["canonical_id"] == eid
    assert out["seq"] == 1

    # 响应内嵌的收据视图标出作废状态
    rv = out["receipt"]
    assert rv["voided"] is True
    assert rv["voided_reading_ids"] == ["a", "b"]
    assert rv["correction"]["request_key"] == "key-1"
    assert rv["correction"]["canonical_id"] == eid
    assert rv["correction"]["voided_reading_ids"] == ["a", "b"]
    assert rv["correction"]["reason"] == "传感器故障批次作废"

    # 读数列表立即排除
    assert reading_ids(client, eid) == ["c"]


def test_correct_receipt_content_seq_and_hashes_unchanged(env):
    client, _ = env
    eid = make_exhibit(client)
    first = send(client, eid, [{"id": "a", "ts": iso(0), "lux": 100}])
    rid = first["receipt"]["receipt_id"]

    out = correct(client, eid, rid, "key-1").json()
    rv = out["receipt"]
    # 内容 / 序号 / 哈希完全不变
    assert rv["seq"] == 1
    assert rv["content"] == first["receipt"]["content"]
    assert rv["prev_hash"] == first["receipt"]["prev_hash"]
    assert rv["content_hash"] == first["receipt"]["content_hash"]
    assert rv["chain_hash"] == first["receipt"]["chain_hash"]
    assert rv["receipt_id"] == rid

    # 链尾继续衔接原哈希：作废后再上报，序号与链衔接照旧
    third = send(client, eid, [{"id": "z", "ts": iso(10), "lux": 1}])
    assert third["receipt"]["seq"] == 2
    assert third["receipt"]["prev_hash"] == first["receipt"]["chain_hash"]
    assert third["receipt"]["chain_hash"] != first["receipt"]["chain_hash"]


def test_receipt_queries_show_voided_status_everywhere(env):
    client, _ = env
    eid = make_exhibit(client)
    first = send(client, eid, [{"id": "a", "ts": iso(0), "lux": 100}])
    send(client, eid, [{"id": "b", "ts": iso(1), "lux": 100}])
    rid = first["receipt"]["receipt_id"]
    correct(client, eid, rid, "key-1")

    # 展品维度单张
    one = client.get(f"/exhibits/{eid}/receipts/1").json()
    assert one["voided"] is True and one["voided_reading_ids"] == ["a"]
    assert one["correction"] is not None and one["correction"]["request_key"] == "key-1"

    # 展品维度列表
    lst = client.get(f"/exhibits/{eid}/receipts").json()["receipts"]
    assert [(r["seq"], r["voided"], r["voided_reading_ids"]) for r in lst] == [
        (1, True, ["a"]), (2, False, []),
    ]
    assert lst[1]["correction"] is None

    # 链标识直查
    chain = client.get(f"/receipts/chains/{eid}").json()["receipts"]
    assert chain[0]["voided"] is True and chain[1]["voided"] is False

    # 整链校验仍通过（哈希没有被改写）
    v = client.get(f"/exhibits/{eid}/receipts/verify").json()
    assert v["ok"] is True and v["checked"] == 2


def test_dose_recomputed_on_remaining_timeline_after_void(env):
    client, _ = env
    # 上限 1000 lux·h，间隔上限 3600s
    eid = make_exhibit(client, limit=1000.0, interval=3600.0)
    send(client, eid, [
        {"id": "r1", "ts": iso(0), "lux": 100},
        {"id": "r2", "ts": iso(3600), "lux": 100},
    ])
    send(client, eid, [{"id": "r3", "ts": iso(7200), "lux": 100}])

    def dose():
        return client.get(
            f"/exhibits/{eid}/dose",
            params={"start": iso(0), "end": iso(10800)},
        ).json()

    before = dose()
    assert before["reading_count"] == 3
    assert before["known_dose_luxh"] == pytest.approx(200.0)   # 2 段 × 100 lux·h

    rid1 = client.get(f"/exhibits/{eid}/receipts/1").json()["receipt_id"]
    correct(client, eid, rid1, "key-1")

    after = dose()
    # r1/r2 作废：窗口内只剩 r3 一个点，无可积分段
    assert after["reading_count"] == 1
    assert after["known_dose_luxh"] == pytest.approx(0.0)
    assert after["covered_seconds"] == pytest.approx(0.0)
    assert after["unknown_seconds"] == pytest.approx(10800.0)


def test_diagnose_recomputed_on_remaining_timeline_after_void(env):
    client, _ = env
    eid = make_exhibit(client, interval=3600.0)
    send(client, eid, [
        {"id": "r1", "ts": iso(0), "lux": 100},
        {"id": "r2", "ts": iso(7200), "lux": 100},   # 与 r1 间隔 7200 > 3600 → gap
    ])
    rid1 = client.get(f"/exhibits/{eid}/receipts/1").json()["receipt_id"]

    body = {
        "start": iso(-3600), "end": iso(10800),
        "max_rate_lux_per_second": 1000.0,
        "stuck_tolerance_lux": 0.0,
        "min_stuck_seconds": 1.0,
    }
    before = client.post(f"/exhibits/{eid}/readings/diagnose", json=body).json()
    assert [e["type"] for e in before["events"]] == ["gap"]

    correct(client, eid, rid1, "key-1")
    after = client.post(f"/exhibits/{eid}/readings/diagnose", json=body).json()
    # r1 作废后只剩 r2 一个点，gap 事件随之消失
    assert after["events"] == []


# ---------- 幂等：request_key 重放与冲突 ----------

def test_same_request_key_replays_original_result(env):
    client, _ = env
    eid = make_exhibit(client)
    first = send(client, eid, [{"id": "a", "ts": iso(0), "lux": 100}])
    rid = first["receipt"]["receipt_id"]

    out1 = correct(client, eid, rid, "key-1", "原始原因").json()
    out2 = correct(client, eid, rid, "key-1", "原始原因").json()
    assert out1["already_committed"] is False
    assert out2["already_committed"] is True
    # 回放首次结果（作废集合、时间、收据视图一致）
    assert out2["voided_reading_ids"] == out1["voided_reading_ids"]
    assert out2["corrected_at"] == out1["corrected_at"]
    assert out2["reason"] == "原始原因"
    assert out2["receipt"]["correction"]["request_key"] == "key-1"
    # 只作废一次：库里仍只有一条读数被标记
    conn = sqlite3.connect(str(env[1]))
    n = conn.execute("SELECT COUNT(*) FROM readings WHERE voided_at IS NOT NULL").fetchone()[0]
    rows = conn.execute("SELECT COUNT(*) FROM reading_corrections").fetchone()[0]
    conn.close()
    assert n == 1 and rows == 1


def test_same_request_key_replay_via_alias_equivalent(env):
    client, _ = env
    src = make_exhibit(client, "src")
    tgt = make_exhibit(client, "tgt")
    first = send(client, tgt, [{"id": "t1", "ts": iso(0), "lux": 10}])
    token = client.post("/merges/preview", json={"source_id": "src", "target_id": "tgt"}).json()["token"]
    assert client.post("/merges/commit", json={"token": token, "resolutions": []}).status_code == 200

    rid = first["receipt"]["receipt_id"]
    out1 = correct(client, "tgt", rid, "key-1").json()
    # 用来源别名提交同一请求：解析后同一档案/收据/原因 → 重放
    out2 = correct(client, "src", rid, "key-1").json()
    assert out1["already_committed"] is False
    assert out2["already_committed"] is True
    assert out2["canonical_id"] == "tgt"
    assert out2["requested_id"] == "tgt"  # 回放的是首次结果（首次以 tgt 提交）


def test_same_request_key_different_target_or_reason_conflicts(env):
    client, _ = env
    eid = make_exhibit(client)
    r1 = send(client, eid, [{"id": "a", "ts": iso(0), "lux": 100}])
    r2 = send(client, eid, [{"id": "b", "ts": iso(1), "lux": 100}])

    assert correct(client, eid, r1["receipt"]["receipt_id"], "key-1").status_code == 200
    # 换收据
    resp = correct(client, eid, r2["receipt"]["receipt_id"], "key-1")
    assert resp.status_code == 409 and "request_key" in resp.json()["detail"]
    # 换原因
    resp = correct(client, eid, r1["receipt"]["receipt_id"], "key-1", reason="另一个原因")
    assert resp.status_code == 409
    # 换档案（同一 request_key 拿去作废另一档案的收据）
    other = make_exhibit(client)
    o1 = send(client, other, [{"id": "o", "ts": iso(0), "lux": 1}])
    resp = correct(client, other, o1["receipt"]["receipt_id"], "key-1")
    assert resp.status_code == 409
    # 被波及的另一档案读数没有被作废
    assert reading_ids(client, other) == ["o"]


def test_correcting_same_receipt_twice_with_new_key_conflicts(env):
    client, _ = env
    eid = make_exhibit(client)
    first = send(client, eid, [{"id": "a", "ts": iso(0), "lux": 100}])
    rid = first["receipt"]["receipt_id"]
    assert correct(client, eid, rid, "key-1").status_code == 200
    resp = correct(client, eid, rid, "key-2", "再次作废")
    assert resp.status_code == 409
    # 仍是原更正记录
    conn = sqlite3.connect(str(env[1]))
    keys = [r[0] for r in conn.execute(
        "SELECT request_key FROM reading_corrections WHERE receipt_id = ?", (rid,))]
    conn.close()
    assert keys == ["key-1"]


def test_concurrent_corrections_same_receipt_only_one_wins(env):
    client, _ = env
    eid = make_exhibit(client)
    send(client, eid, [{"id": "a", "ts": iso(0), "lux": 100}])
    rid = client.get(f"/exhibits/{eid}/receipts/1").json()["receipt_id"]

    def post(i: int) -> int:
        return correct(client, eid, rid, f"key-{i}").status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(post, range(8)))
    # 恰好一个 200，其余全部 409
    assert sorted(statuses) == [200] + [409] * 7
    conn = sqlite3.connect(str(env[1]))
    assert conn.execute("SELECT COUNT(*) FROM reading_corrections").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM readings WHERE voided_at IS NOT NULL"
    ).fetchone()[0] == 1
    conn.close()


# ---------- 冲突：链 / 归并 / 归属变化 ----------

def test_broken_chain_conflicts_and_leaves_no_partial_change(env):
    client, db_file = env
    eid = make_exhibit(client)
    first = send(client, eid, [{"id": "a", "ts": iso(0), "lux": 100}])
    rid = first["receipt"]["receipt_id"]

    conn = sqlite3.connect(str(db_file))
    conn.execute(
        "UPDATE ingest_receipts SET chain_hash = ? WHERE receipt_id = ?",
        ("f" * 64, rid),
    )
    conn.commit()
    conn.close()

    resp = correct(client, eid, rid, "key-1")
    assert resp.status_code == 409
    assert "链" in resp.json()["detail"]
    # 无部分变更：读数仍有效、无更正记录
    assert reading_ids(client, eid) == ["a"]
    conn = sqlite3.connect(str(db_file))
    assert conn.execute("SELECT COUNT(*) FROM reading_corrections").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM readings WHERE voided_at IS NOT NULL"
    ).fetchone()[0] == 0
    conn.close()
    # request_key 未被占用：链修复后同一 request_key 可正常使用（重放语义不被失败毒化）


def test_receipt_from_merged_source_chain_conflicts(env):
    client, _ = env
    src = make_exhibit(client, "src")
    tgt = make_exhibit(client, "tgt")
    s1 = send(client, src, [{"id": "s1", "ts": iso(0), "lux": 10}])
    send(client, tgt, [{"id": "t1", "ts": iso(1), "lux": 10}])
    token = client.post("/merges/preview", json={"source_id": "src", "target_id": "tgt"}).json()["token"]
    assert client.post("/merges/commit", json={"token": token, "resolutions": []}).status_code == 200

    # 用来源别名去作废来源旧链上的收据 → 归并状态已变化，409
    resp = correct(client, "src", s1["receipt"]["receipt_id"], "key-1")
    assert resp.status_code == 409
    assert "归并" in resp.json()["detail"]
    # 来源读数搬迁后仍有效
    assert reading_ids(client, "src") == ["s1", "t1"]

    # 来源链仍原样可查（未被作废标注）
    chain = client.get("/receipts/chains/src").json()["receipts"]
    assert all(r["voided"] is False for r in chain)


def test_correct_target_chain_via_alias_after_merge_works(env):
    client, _ = env
    src = make_exhibit(client, "src")
    tgt = make_exhibit(client, "tgt")
    t1 = send(client, tgt, [{"id": "t1", "ts": iso(0), "lux": 10}])
    token = client.post("/merges/preview", json={"source_id": "src", "target_id": "tgt"}).json()["token"]
    client.post("/merges/commit", json={"token": token, "resolutions": []})

    resp = correct(client, "src", t1["receipt"]["receipt_id"], "key-1")
    assert resp.status_code == 200
    out = resp.json()
    assert out["canonical_id"] == "tgt"
    assert out["requested_id"] == "src"
    assert out["voided_reading_ids"] == ["t1"]
    assert reading_ids(client, "src") == []


def test_reading_moved_by_merge_then_correction_conflicts_on_target_receipt(env):
    """读数归属变化：目标链收据记录的读数被归并搬迁后，再作废该收据 → 409。

    构造：来源与目标都有读数 x 且内容一致（自动去重场景不改变目标行），
    这里用 keep_source 覆盖目标读数内容，使目标旧收据中的 x 内容已变化。
    """
    client, _ = env
    src = make_exhibit(client, "src")
    tgt = make_exhibit(client, "tgt")
    t_recv = send(client, tgt, [{"id": "x", "ts": iso(0), "lux": 200}])
    send(client, src, [{"id": "x", "ts": iso(0), "lux": 100}])

    pv = client.post("/merges/preview", json={"source_id": "src", "target_id": "tgt"}).json()
    assert pv["conflict_count"] == 1
    token = pv["token"]
    committed = client.post("/merges/commit", json={
        "token": token,
        "resolutions": [{"reading_id": "x", "action": "keep_source"}],
    })
    assert committed.status_code == 200, committed.text

    # 目标旧收据中的 x 已被来源内容覆盖（lux 200→100）：读数内容/归属变化 → 409
    resp = correct(client, "tgt", t_recv["receipt"]["receipt_id"], "key-1")
    assert resp.status_code == 409
    assert "归属" in resp.json()["detail"] or "变化" in resp.json()["detail"]
    # 读数仍有效（keep_source 后以 100 lux 存在）
    rows = client.get("/exhibits/tgt/readings").json()["readings"]
    assert [r["id"] for r in rows] == ["x"] and rows[0]["lux"] == 100.0


def test_direct_db_delete_of_reading_makes_correction_conflict(env):
    client, db_file = env
    eid = make_exhibit(client)
    first = send(client, eid, [{"id": "a", "ts": iso(0), "lux": 100}])
    rid = first["receipt"]["receipt_id"]

    conn = sqlite3.connect(str(db_file))
    conn.execute("DELETE FROM readings WHERE exhibit_id = ? AND reading_id = ?", (eid, "a"))
    conn.commit()
    conn.close()

    resp = correct(client, eid, rid, "key-1")
    assert resp.status_code == 409
    assert "不在规范档案" in resp.json()["detail"] or "归属" in resp.json()["detail"]


def test_reading_already_voided_by_another_correction_conflicts(env):
    client, _ = env
    eid = make_exhibit(client)
    # 两张收据各插入一条；第一条作废成功后，构造“读数已作废”状态，
    # 再用另一条新 request_key 作废同一张收据 → 409（已由 re-void 用例覆盖，
    # 这里额外验证读数被其他途径直接置 voided_at 时整次拒绝、无部分变更）。
    first = send(client, eid, [
        {"id": "a", "ts": iso(0), "lux": 100},
        {"id": "b", "ts": iso(1), "lux": 100},
    ])
    rid = first["receipt"]["receipt_id"]
    # 直接在库中只把 a 置为作废（模拟 a 已被作废、b 仍有效）
    conn = sqlite3.connect(str(env[1]))
    conn.execute(
        "UPDATE readings SET voided_at = '2026-01-01T00:00:00.000Z' "
        "WHERE exhibit_id = ? AND reading_id = ?", (eid, "a"),
    )
    conn.commit()
    conn.close()

    resp = correct(client, eid, rid, "key-1")
    assert resp.status_code == 409
    # b 没有被部分作废
    conn = sqlite3.connect(str(env[1]))
    n = conn.execute(
        "SELECT COUNT(*) FROM readings WHERE exhibit_id = ? AND voided_at IS NOT NULL",
        (eid,),
    ).fetchone()[0]
    conn.close()
    assert n == 1  # 只有直接置位的 a，b 未被波及


# ---------- 404 / 422 ----------

def test_correct_unknown_exhibit_or_receipt_returns_404(env):
    client, _ = env
    eid = make_exhibit(client)
    first = send(client, eid, [{"id": "a", "ts": iso(0), "lux": 100}])

    assert correct(client, "ghost", first["receipt"]["receipt_id"], "k").status_code == 404
    assert correct(client, eid, "no-such-receipt", "k").status_code == 404


def test_receipt_without_inserted_readings_returns_422(env):
    client, _ = env
    eid = make_exhibit(client)
    send(client, eid, [{"id": "a", "ts": iso(0), "lux": 100}])
    second = send(client, eid, [
        {"id": "a", "ts": iso(0), "lux": 100},   # duplicate
        {"id": "a", "ts": iso(0), "lux": 9},     # conflict
    ])
    resp = correct(client, eid, second["receipt"]["receipt_id"], "key-1")
    assert resp.status_code == 422
    assert "inserted" in resp.json()["detail"]


def test_correct_request_validation(env):
    client, _ = env
    eid = make_exhibit(client)
    first = send(client, eid, [{"id": "a", "ts": iso(0), "lux": 100}])
    base = {"receipt_id": first["receipt"]["receipt_id"], "request_key": "k", "reason": "r"}
    for field in ("receipt_id", "request_key", "reason"):
        bad = dict(base)
        bad[field] = ""
        resp = client.post(f"/exhibits/{eid}/readings/correct", json=bad)
        assert resp.status_code == 422


# ---------- 与既有行为共存 ----------

def test_voided_reading_excluded_from_merge_bookkeeping(env):
    """作废读数不参与归并：不搬迁、不碰撞、不计入 final_reading_count。"""
    client, _ = env
    src = make_exhibit(client, "src")
    tgt = make_exhibit(client, "tgt")
    s1_recv = send(client, src, [{"id": "s1", "ts": iso(0), "lux": 10}])
    send(client, src, [{"id": "s2", "ts": iso(1), "lux": 10}])
    send(client, tgt, [{"id": "t1", "ts": iso(2), "lux": 10}])

    # 作废 s1（归并前在来源侧，单独一张收据）
    assert correct(client, "src", s1_recv["receipt"]["receipt_id"], "key-1").status_code == 200

    pv = client.post("/merges/preview", json={"source_id": "src", "target_id": "tgt"}).json()
    assert pv["source_reading_count"] == 1   # 只剩 s2
    token = pv["token"]
    out = client.post("/merges/commit", json={"token": token, "resolutions": []}).json()
    assert out["moved_count"] == 1
    assert out["final_reading_count"] == 2   # s2 + t1（作废行不计入）
    assert reading_ids(client, "tgt") == ["s2", "t1"]


def test_ingest_after_void_still_idempotent_and_chain_unaffected(env):
    """作废后继续上报：新收据正常生成并衔接原链尾，已作废读数保持作废。"""
    client, _ = env
    eid = make_exhibit(client)
    first = send(client, eid, [{"id": "a", "ts": iso(0), "lux": 100}])
    correct(client, eid, first["receipt"]["receipt_id"], "key-1")

    # 同 ID 同数据再报：行仍在（带作废标记），主键冲突 → duplicate，不会“复活”
    again = send(client, eid, [{"id": "a", "ts": iso(0), "lux": 100}])
    assert again["inserted"] == 0 and again["duplicates"] == 1
    assert reading_ids(client, eid) == []
    assert again["receipt"]["seq"] == 2

    # 同 ID 不同数据 → conflict
    conflict = send(client, eid, [{"id": "a", "ts": iso(0), "lux": 1}])
    assert conflict["conflicts"] == 1
    assert reading_ids(client, eid) == []
