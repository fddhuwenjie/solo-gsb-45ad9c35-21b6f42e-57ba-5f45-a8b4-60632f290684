"""照度上报批次更正：作废旧收据中实际插入、仍归属本档案的读数。

语义要点：
- 调用方提交规范档案 ID（可用归并别名）、收据 ID（``receipt_id``）、
  幂等键 ``request_key`` 与作废原因；服务只作废该收据 outcome 中标记为
  ``inserted``、且当前**仍归属该规范档案**（时间/照度与收据请求内容一致）
  的读数。
- 整次操作是**全有或全无**的冲突校验：收据链已变化（断链/内容不符/序号
  重排）、归并状态已变化（收据链不再属于当前规范档案）、或读数归属/内容
  已变化（被归并搬迁、内容被 keep_source 覆盖、已被其它更正作废）→
  整次返回 **409**，不产生任何部分变更，也不留更正记录。
- 作废（把 readings.voided_at 置位）与更正记录在**同一个
  ``BEGIN IMMEDIATE`` 事务**内幂等落库；失败整体回滚。读数行不删除，
  剂量查询、采集质量诊断与读数列表立即排除作废读数并按剩余时间线重算；
  存证收据的内容、序号与链哈希保持不变，仅查询收据时标出作废状态。
- ``request_key`` 全局唯一：同一 request_key 重放返回首次结果
  （``already_committed=true``），用同一 request_key 换档案 / 换收据 /
  换原因提交返回 **409**。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from fastapi import HTTPException

from .receipts import load_chain, verify_chain_rows

# ---------- 冲突异常（整次作废的唯一失败口径） ----------

class CorrectionConflict(Exception):
    """收据链、归并状态或读数归属已变化；整次作废返回 409。"""


class CorrectionEmpty(Exception):
    """收据中没有可作废的实际插入读数；422。"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _find_receipt_by_id(
    conn: sqlite3.Connection, receipt_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM ingest_receipts WHERE receipt_id = ?", (receipt_id,)
    ).fetchone()


def _aliases_of(conn: sqlite3.Connection, canonical_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT alias_id FROM exhibit_aliases WHERE canonical_id = ?",
        (canonical_id,),
    ).fetchall()
    return [r["alias_id"] for r in rows]


def _parse_requested_reading(row: sqlite3.Row) -> dict[str, object]:
    """从收据请求内容取出按 (id -> {ts epoch, lux}) 的规范化读数表。"""
    content = json.loads(row["payload_json"])
    out: dict[str, object] = {}
    for r in content["request"]["readings"]:
        out[r["id"]] = {
            "ts": datetime.fromisoformat(r["ts"]).timestamp(),
            "lux": r["lux"],
        }
    return out


def _inserted_ids(row: sqlite3.Row) -> list[str]:
    """收据 outcome 中实际标记为 inserted 的读数 ID（按请求顺序）。"""
    content = json.loads(row["payload_json"])
    return [r["id"] for r in content["outcome"]["results"] if r["status"] == "inserted"]


def _same_target(
    existing: sqlite3.Row,
    *,
    canonical_id: str,
    receipt_id: str,
    reason: str,
) -> bool:
    """重放请求与首次作废是否指向同一档案 / 同一张收据 / 同一原因。"""
    return (
        existing["canonical_id"] == canonical_id
        and existing["receipt_id"] == receipt_id
        and existing["reason"] == reason
    )


def _replay_conflict(existing: sqlite3.Row, request_key: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=f"request_key {request_key} 已用于另一次批次更正"
               f"（档案 {existing['canonical_id']} 的收据 {existing['receipt_id']}），"
               f"不能换档案、换收据或换原因重提",
    )


def correct_batch(
    conn: sqlite3.Connection,
    *,
    requested_id: str,
    canonical_id: str,
    receipt_id: str,
    request_key: str,
    reason: str,
    receipt_out_builder,
) -> tuple[dict, bool]:
    """作废一张上报收据中本档案实际插入、仍归属本档案的读数。

    返回 (BatchCorrectionOut 的 dict, 是否为重放)。任何冲突抛
    ``CorrectionConflict``（路由层转 409）；无可作废读数抛
    ``CorrectionEmpty``（422）；收据/档案不存在由调用方先判 404。

    作废与更正记录在同一写事务内幂等落库。
    """
    # 先在事务外看重放：request_key 是全局唯一幂等键
    existing = conn.execute(
        "SELECT canonical_id, chain_id, seq, receipt_id, reason, result_json "
        "FROM reading_corrections WHERE request_key = ?",
        (request_key,),
    ).fetchone()
    if existing is not None:
        if not _same_target(existing, canonical_id=canonical_id,
                            receipt_id=receipt_id, reason=reason):
            raise _replay_conflict(existing, request_key)
        # 同一目标（含以别名提交）：回放首次结果
        return json.loads(existing["result_json"]), True

    # BEGIN IMMEDIATE：校验 + 作废 + 更正记录同事务原子完成
    conn.execute("BEGIN IMMEDIATE")
    try:
        # 锁内重读：并发请求可能已用同一 request_key 完成作废
        existing = conn.execute(
            "SELECT canonical_id, chain_id, seq, receipt_id, reason, result_json "
            "FROM reading_corrections WHERE request_key = ?",
            (request_key,),
        ).fetchone()
        if existing is not None:
            if not _same_target(existing, canonical_id=canonical_id,
                                receipt_id=receipt_id, reason=reason):
                raise _replay_conflict(existing, request_key)
            conn.commit()  # 尚未写入，仅释放写锁
            return json.loads(existing["result_json"]), True

        receipt_row = _find_receipt_by_id(conn, receipt_id)
        if receipt_row is None:
            raise HTTPException(
                status_code=404, detail=f"收据不存在: {receipt_id}"
            )
        chain_id, seq = receipt_row["chain_id"], receipt_row["seq"]

        # 1) 归并状态校验：收据链必须就是当前规范档案的链。
        #    链属于当前档案的某个归并别名（来源旧链）→ 归并状态已变化，409。
        if chain_id != canonical_id:
            if chain_id in _aliases_of(conn, canonical_id):
                raise CorrectionConflict(
                    f"收据 {receipt_id} 属于来源档案 {chain_id} 的旧链，该档案已"
                    f"归并入 {canonical_id}；归并状态已变化，不能凭该收据作废"
                )
            # 链不属于该档案：归属不符（属别的档案/已删除档案）
            raise CorrectionConflict(
                f"收据 {receipt_id} 属于链 {chain_id}，不归属规范档案 "
                f"{canonical_id}；读数归属已变化，整次作废被拒绝"
            )

        # 2) 收据链校验：链被追加是正常的；但断链、内容不符、序号重排 → 409
        verdict = verify_chain_rows(load_chain(conn, chain_id))
        if not verdict.ok:
            failure = verdict.first_failure or {}
            raise CorrectionConflict(
                f"收据链 {chain_id} 校验失败：第 {failure.get('seq')} 张"
                f"（{failure.get('reason')}），链状态已变化，整次作废被拒绝"
            )

        # 3) 读数归属校验：只允许作废该收据实际 inserted、当前仍归属该档案、
        #    且时间/照度与收据请求内容一致（未被归并 keep_source 改写）的读数。
        inserted = _inserted_ids(receipt_row)
        if not inserted:
            raise CorrectionEmpty(
                f"收据 {receipt_id} 中没有实际插入（inserted）的读数，无可作废对象"
            )
        requested = _parse_requested_reading(receipt_row)
        placeholders = ",".join("?" for _ in inserted)
        rows = conn.execute(
            f"SELECT reading_id, ts, lux, voided_at FROM readings "
            f"WHERE exhibit_id = ? AND reading_id IN ({placeholders})",
            (canonical_id, *inserted),
        ).fetchall()
        live = {r["reading_id"]: r for r in rows}

        voided_ids: list[str] = []
        for rid in inserted:
            cur = live.get(rid)
            if cur is None:
                raise CorrectionConflict(
                    f"收据 {receipt_id} 中的读数 {rid} 已不在规范档案 "
                    f"{canonical_id} 名下（被搬迁或删除），读数归属已变化，"
                    f"整次作废被拒绝"
                )
            spec = requested[rid]
            if cur["ts"] != spec["ts"] or cur["lux"] != spec["lux"]:
                raise CorrectionConflict(
                    f"收据 {receipt_id} 中的读数 {rid} 的时间/照度已变化"
                    f"（归并覆盖等），读数归属已变化，整次作废被拒绝"
                )
            if cur["voided_at"] is not None:
                raise CorrectionConflict(
                    f"收据 {receipt_id} 中的读数 {rid} 已被作废，"
                    f"收据作废状态已变化，整次作废被拒绝"
                )
            voided_ids.append(rid)
        voided_ids.sort()

        # 4) 作废读数 + 写更正记录（同事务）。UPDATE 与 INSERT 任一失败整体回滚。
        corrected_at = _utc_now_iso()
        cur = conn.execute(
            f"UPDATE readings SET voided_at = ? "
            f"WHERE exhibit_id = ? AND reading_id IN ({placeholders}) "
            f"AND voided_at IS NULL",
            (corrected_at, canonical_id, *voided_ids),
        )
        if cur.rowcount != len(voided_ids):
            # 锁内已逐条确认非作废；行数不符说明状态在极端情况下变化，拒绝整次
            raise CorrectionConflict(
                f"收据 {receipt_id} 的读数作废状态在提交时发生变化，"
                f"整次作废被拒绝"
            )

        result_stub = {
            "requested_id": requested_id,
            "canonical_id": canonical_id,
            "chain_id": chain_id,
            "seq": seq,
            "receipt_id": receipt_id,
            "request_key": request_key,
            "reason": reason,
            "voided_count": len(voided_ids),
            "voided_reading_ids": voided_ids,
            "corrected_at": corrected_at,
        }

        # 先落更正记录（result_json 占位），使下面构造的收据视图已能带出
        # voided / correction 标注；随后再回填最终结果（同一事务内）。
        conn.execute(
            """INSERT INTO reading_corrections
               (request_key, chain_id, seq, receipt_id, canonical_id, reason,
                voided_ids_json, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request_key, chain_id, seq, receipt_id, canonical_id, reason,
                json.dumps(voided_ids, ensure_ascii=False),
                "",  # 占位，收据视图构造完成后回填
            ),
        )

        # 收据视图在作废与更正记录均已写入后构造（带出 voided 标记与更正记录）；
        # 收据内容/序号/哈希不受影响，仍是原行。
        receipt_view = receipt_out_builder(
            conn.execute(
                "SELECT * FROM ingest_receipts WHERE chain_id = ? AND seq = ?",
                (chain_id, seq),
            ).fetchone(),
            conn,
        )
        result = dict(result_stub)
        result["receipt"] = receipt_view.model_dump()

        conn.execute(
            "UPDATE reading_corrections SET result_json = ? WHERE request_key = ?",
            (json.dumps(result, ensure_ascii=False, sort_keys=True), request_key),
        )
        try:
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
        return result, False

    except HTTPException:
        conn.rollback()
        raise
    except (CorrectionConflict, CorrectionEmpty):
        conn.rollback()
        raise
    except sqlite3.IntegrityError:
        # (chain_id, seq) 唯一索引兜底：并发下另一请求已更正该收据
        conn.rollback()
        raise CorrectionConflict(
            f"收据 {receipt_id} 已被并发更正，作废状态已变化，整次作废被拒绝"
        )
    except Exception:
        conn.rollback()
        raise
