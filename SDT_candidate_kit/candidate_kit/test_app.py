"""Tests for the pure rules (R1-R5) and the CRUD/punch/regularize flows.
Uses mongomock (an in-memory fake), so the aggregation endpoints are NOT covered here:
run those against a real MongoDB (see README)."""
from datetime import datetime, timezone

import mongomock
import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import IST, attendance_date, compute_late_minutes, compute_overtime, compute_work_hours, shift_bounds


def ist(y, mo, d, h, mi, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=IST)


def ms(dt):
    return main.dt_to_ms(dt)


# ---- rules -------------------------------------------------------------------
def test_late_rule_examples():
    start, _ = shift_bounds("2026-07-06", "09:30", "18:30")
    assert compute_late_minutes(ist(2026, 7, 6, 9, 40, 0), start) == 0
    assert compute_late_minutes(ist(2026, 7, 6, 9, 40, 1), start) == 10
    assert compute_late_minutes(ist(2026, 7, 6, 10, 15, 59), start) == 45
    assert compute_late_minutes(ist(2026, 7, 6, 9, 0, 0), start) == 0


def test_overtime():
    _, end = shift_bounds("2026-07-06", "09:30", "18:30")
    assert compute_overtime(ist(2026, 7, 6, 19, 0, 0), end) == 30
    assert compute_overtime(ist(2026, 7, 6, 18, 59, 59), end) == 0
    assert compute_overtime(ist(2026, 7, 6, 19, 10, 0), end) == 40
    assert compute_overtime(ist(2026, 7, 6, 17, 0, 0), end) == 0


def test_work_hours_half_up():
    a = ist(2026, 7, 6, 9, 0, 0)
    assert compute_work_hours(a, ist(2026, 7, 6, 18, 7, 12)) == 9.12  # 9.12 exactly
    assert compute_work_hours(a, ist(2026, 7, 6, 12, 30, 0)) == 3.5
    # 1.125h = 4050s -> 1.13 half-up (Python round(1.125, 2) gives 1.12)
    assert compute_work_hours(a, datetime.fromtimestamp(a.timestamp() + 4050, tz=IST)) == 1.13


def test_overnight_date_and_bounds():
    assert attendance_date(ist(2026, 7, 7, 5, 0), "22:00", "06:00").isoformat() == "2026-07-06"
    assert attendance_date(ist(2026, 7, 6, 21, 55), "22:00", "06:00").isoformat() == "2026-07-06"
    assert attendance_date(ist(2026, 7, 7, 6, 0), "22:00", "06:00").isoformat() == "2026-07-07"
    assert attendance_date(ist(2026, 7, 7, 0, 10), "09:30", "18:30").isoformat() == "2026-07-07"
    s, e = shift_bounds("2026-07-06", "22:00", "06:00")
    assert e.date().isoformat() == "2026-07-07" and s.date().isoformat() == "2026-07-06"
    _, e = shift_bounds("2026-07-06", "22:00", "06:00")
    assert compute_overtime(ist(2026, 7, 7, 6, 40), e) == 40


# ---- API flows ---------------------------------------------------------------
@pytest.fixture()
def api():
    main.db = mongomock.MongoClient(tz_aware=True)["t"]
    main._indexes_ready = False
    main.ensure_indexes()
    return TestClient(main.app)


EMP = {"emp_code": "EMP0001", "name": "A", "email": "a@x.com", "department": "Eng", "joined_on": "2026-01-05"}
NIGHT = {**EMP, "emp_code": "EMP0005", "shift_start": "22:00", "shift_end": "06:00"}


def test_employee_create_validate_list(api):
    r = api.post("/employees", json=EMP)
    assert r.status_code == 201 and isinstance(r.json()["created_at"], int) and "_id" not in r.json()
    assert api.post("/employees", json=EMP).status_code == 409
    assert api.post("/employees", json={**EMP, "emp_code": "X1"}).status_code == 422
    assert api.post("/employees", json={**EMP, "emp_code": "EMP0002", "shift_end": "09:30"}).status_code == 422
    assert api.post("/employees", json={**EMP, "emp_code": "EMP0002", "joined_on": "2026-02-30"}).status_code == 422
    for i in range(2, 5):
        api.post("/employees", json={**EMP, "emp_code": f"EMP000{i}", "department": "Ops" if i > 2 else "Eng"})
    r = api.get("/employees", params={"department": "Ops", "page": 1, "page_size": 1}).json()
    assert r["total"] == 2 and [e["emp_code"] for e in r["items"]] == ["EMP0003"]
    assert api.get("/employees", params={"page": 0}).status_code == 422
    assert api.get("/employees", params={"page_size": 101}).status_code == 422
    assert api.get("/employees", params={"page": 2, "page_size": 2}).json()["items"][0]["emp_code"] == "EMP0003"


def test_punch_flow(api):
    api.post("/employees", json=EMP)
    t_in = ms(ist(2026, 7, 6, 10, 5, 0)) + 900  # 900 ms of noise must be truncated
    r = api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": t_in})
    assert r.status_code == 201
    b = r.json()
    assert b["date"] == "2026-07-06" and b["late_minutes"] == 35 and b["punch_in"] == t_in - 900
    assert b["work_hours"] is None and b["history"] == []
    assert api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": t_in}).status_code == 409
    assert api.post("/attendance/punch-in", json={"emp_code": "NOPE", "punched_at": t_in}).status_code == 404
    for bad in (1783312500, 1.5e12, "1783312500000", True):
        assert api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": bad}).status_code == 422
    assert api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "status": "ABSENT"}).status_code == 422

    out = ms(ist(2026, 7, 6, 19, 10, 0))
    assert api.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": t_in - 900}).status_code == 422
    r = api.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": out})
    assert r.status_code == 200
    b = r.json()
    assert b["overtime_minutes"] == 40 and b["work_hours"] == 9.08 and b["half_day"] is False
    assert api.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": out}).status_code == 409
    assert api.post("/attendance/punch-out", json={"emp_code": "NOPE", "punched_at": out}).status_code == 404


def test_overnight_punch_across_midnight(api):
    api.post("/employees", json=NIGHT)
    r = api.post("/attendance/punch-in", json={"emp_code": "EMP0005", "punched_at": ms(ist(2026, 7, 6, 21, 55))})
    assert r.json()["date"] == "2026-07-06" and r.json()["late_minutes"] == 0
    r = api.post("/attendance/punch-out", json={"emp_code": "EMP0005", "punched_at": ms(ist(2026, 7, 7, 6, 40))})
    assert r.status_code == 200 and r.json()["overtime_minutes"] == 40 and r.json()["work_hours"] == 8.75
    # more than 24h after punch-in -> 422
    api.post("/attendance/punch-in", json={"emp_code": "EMP0005", "punched_at": ms(ist(2026, 7, 8, 21, 55))})
    r = api.post("/attendance/punch-out", json={"emp_code": "EMP0005", "punched_at": ms(ist(2026, 7, 9, 22, 0))})
    assert r.status_code == 422
    # a 05:00 punch-in belongs to the previous day's shift
    r = api.post("/attendance/punch-in", json={"emp_code": "EMP0005", "punched_at": ms(ist(2026, 7, 11, 5, 0))})
    assert r.json()["date"] == "2026-07-10"


def test_list_attendance(api):
    api.post("/employees", json=EMP)
    api.post("/employees", json={**EMP, "emp_code": "EMP0002"})
    for c in ("EMP0002", "EMP0001"):
        for d in (6, 7):
            api.post("/attendance/punch-in", json={"emp_code": c, "punched_at": ms(ist(2026, 7, d, 9, 30))})
    r = api.get("/attendance").json()
    assert r["total"] == 4
    assert [(i["date"], i["emp_code"]) for i in r["items"]] == [
        ("2026-07-07", "EMP0001"), ("2026-07-07", "EMP0002"), ("2026-07-06", "EMP0001"), ("2026-07-06", "EMP0002")]
    r = api.get("/attendance", params={"emp_code": "EMP0001", "date_from": "2026-07-07"}).json()
    assert r["total"] == 1 and "id" not in r["items"][0] and "_id" not in r["items"][0]
    assert api.get("/attendance", params={"date_from": "2026-07-08", "date_to": "2026-07-01"}).status_code == 422
    assert api.get("/attendance", params={"date_from": "2026-7-8"}).status_code == 422
    assert api.get("/attendance", params={"status": "BOGUS"}).status_code == 422
    assert api.get("/attendance", params={"page": 1, "page_size": 3}).json()["total"] == 4


def seed_closed_day(api):
    api.post("/employees", json=EMP)
    api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms(ist(2026, 7, 7, 10, 5))})
    api.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(ist(2026, 7, 7, 18, 30))})


def test_regularize(api):
    seed_closed_day(api)
    url = "/attendance/EMP0001/2026-07-07"
    body = {"punch_in": ms(ist(2026, 7, 7, 9, 28)), "reason": "biometric glitch", "regularized_by": "hr", "late_minutes": 99}
    r = api.patch(url, json=body)
    assert r.status_code == 200
    b = r.json()
    assert b["late_minutes"] == 0 and b["work_hours"] == 9.03
    h = b["history"][0]
    assert h["by"] == "hr" and set(h["changes"]) == {"punch_in", "late_minutes", "work_hours"}  # only what changed
    assert h["changes"]["punch_in"]["from"] == ms(ist(2026, 7, 7, 10, 5)) and isinstance(h["at"], int)
    assert h["changes"]["late_minutes"] == {"from": 35, "to": 0}
    assert api.patch(url, json=body).status_code == 422  # changes nothing
    # second correction appends, never overwrites
    r = api.patch(url, json={"punch_out": ms(ist(2026, 7, 7, 19, 15)), "reason": "forgot to punch", "regularized_by": "hr2"})
    assert r.status_code == 200 and len(r.json()["history"]) == 2 and r.json()["overtime_minutes"] == 45
    # validation
    base = {"reason": "because", "regularized_by": "hr"}
    assert api.patch(url, json={**base, "status": "LEAVE", "punch_in": ms(ist(2026, 7, 7, 9, 0))}).status_code == 422
    assert api.patch(url, json={**base, "punch_in": ms(ist(2026, 7, 8, 9, 0))}).status_code == 422  # other date
    assert api.patch(url, json={**base, "punch_out": ms(ist(2026, 7, 7, 8, 0))}).status_code == 422  # before in
    assert api.patch(url, json={**base, "reason": "no"}).status_code == 422
    assert api.patch("/attendance/EMP0001/2026-07-09", json={**base, "status": "LEAVE"}).status_code == 404
    assert api.patch("/attendance/NOPE/2026-07-07", json={**base, "status": "LEAVE"}).status_code == 404
    assert api.patch("/attendance/EMP0001/07-07-2026", json={**base, "status": "LEAVE"}).status_code == 422
    # LEAVE clears times; then a presence status without punch_in is 422
    r = api.patch(url, json={**base, "status": "LEAVE"})
    b = r.json()
    assert r.status_code == 200 and b["punch_in"] is None and b["work_hours"] is None and b["late_minutes"] == 0
    assert set(b["history"][-1]["changes"]) >= {"status", "punch_in", "punch_out", "work_hours"}
    assert api.patch(url, json={**base, "status": "WFH"}).status_code == 422
    assert api.patch(url, json={**base, "status": "WFH", "punch_in": ms(ist(2026, 7, 7, 9, 0))}).status_code == 200


def test_regularize_conflict_on_stale_read(api):
    """If the record changes between our read and our write, we must return 409 and not lose history."""
    seed_closed_day(api)
    url = "/attendance/EMP0001/2026-07-07"
    real = main.db.attendance_logs.find_one_and_update
    calls = {"n": 0}

    class Wrapper:
        def __getattr__(self, k):
            return getattr(main.db.attendance_logs, k)

    def sneaky(*a, **k):  # simulate a concurrent writer winning first
        if calls["n"] == 0:
            calls["n"] += 1
            main.db.attendance_logs.update_one({"emp_code": "EMP0001"}, {"$push": {"history": {"at": datetime.now(timezone.utc), "by": "x", "reason": "y", "changes": {}}}})
        return real(*a, **k)

    main.db.attendance_logs.find_one_and_update = sneaky
    try:
        r = api.patch(url, json={"punch_in": ms(ist(2026, 7, 7, 9, 28)), "reason": "glitch!", "regularized_by": "hr"})
    finally:
        main.db.attendance_logs.find_one_and_update = real
    assert r.status_code == 409
    assert len(main.db.attendance_logs.find_one({"emp_code": "EMP0001"})["history"]) == 1


def test_legacy_record_defaults(api):
    api.post("/employees", json=EMP)
    main.db.attendance_logs.insert_one({"emp_code": "EMP0001", "date": "2026-07-14", "status": "PRESENT",
        "punch_in": datetime(2026, 7, 14, 4, 2, tzinfo=timezone.utc), "punch_out": datetime(2026, 7, 14, 13, 1, tzinfo=timezone.utc),
        "work_hours": 8.98, "late_minutes": 0, "overtime_minutes": 0})
    b = api.get("/attendance").json()["items"][0]
    assert b["half_day"] is False and b["history"] == []
    r = api.patch("/attendance/EMP0001/2026-07-14", json={"status": "WFH", "reason": "worked home", "regularized_by": "hr"})
    assert r.status_code == 200 and len(r.json()["history"]) == 1


def test_health(api):
    assert api.get("/health").json() == {"status": "ok"}
