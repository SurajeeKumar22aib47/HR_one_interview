"""
Employee Attendance & Analytics API  (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB  (a local .env is loaded, but real environment variables win)

Layout of this file:
  1. configuration / database
  2. time + business-rule helpers (R1-R5, R8)  - pure functions, no database
  3. request models and serialisation helpers
  4. employees, attendance (punch-in, punch-out, list, regularize)
  5. aggregation pipeline builders + analytics endpoints
  6. explain endpoint (reuses the same builders as the real endpoints)
"""
import json
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Literal, Optional

from bson import SON, json_util
from bson.decimal128 import Decimal128
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

load_dotenv()  # override=False (default): real environment variables win over .env
log = logging.getLogger("attendance")

# --------------------------------------------------------------------------- #
# 1. Configuration / database
# --------------------------------------------------------------------------- #
# tz_aware=True: PyMongo returns timezone-aware (UTC) datetimes instead of naive ones.
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

PRESENCE = ["PRESENT", "WFH", "ON_DUTY"]  # R6
NON_PRESENCE = ("ABSENT", "LEAVE")
GRACE_SECONDS = 10 * 60  # R2
OVERTIME_MIN = 30  # R3
HALF_DAY_LIMIT = 4.5  # R5
MAX_SHIFT = timedelta(hours=24)

MIN_MS, MAX_MS = 100_000_000_000, 4_102_444_800_000  # from openapi.yaml EpochMillis

_indexes_ready = False
_index_lock = threading.Lock()


def ensure_indexes() -> None:
    """Idempotent. Called at startup (and lazily from /health if Mongo was down at startup)."""
    global _indexes_ready
    with _index_lock:
        if _indexes_ready:
            return
        emp, logs = db.employees, db.attendance_logs
        # emp_code must be unique -> the database enforces it (race-safe create)
        emp.create_index([("emp_code", ASCENDING)], unique=True, name="uq_emp_code")
        # GET /employees?department=..  (filter + sort emp_code), trend/summary department match
        emp.create_index([("department", ASCENDING), ("emp_code", ASCENDING)], name="ix_emp_dept_code")
        # department summary without a department filter: joined_on <= end of month
        emp.create_index([("joined_on", ASCENDING)], name="ix_emp_joined_on")
        # one record per (emp_code, date): race-safe punch-in; serves per-employee lists and monthly analytics
        logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="uq_log_emp_date")
        # GET /attendance sort (date desc, emp_code asc) with no emp_code filter; month range scans (leaderboard)
        logs.create_index([("date", DESCENDING), ("emp_code", ASCENDING)], name="ix_log_date_emp")
        # GET /attendance?status=..
        logs.create_index(
            [("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)], name="ix_log_status_date_emp"
        )
        # punch-out: most recent record with punch_in <= punched_at for an employee
        logs.create_index([("emp_code", ASCENDING), ("punch_in", DESCENDING)], name="ix_log_emp_punch_in")
        _indexes_ready = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ensure_indexes()
    except Exception:  # Mongo not reachable yet: /health will report 503 and retry
        log.exception("could not create indexes at startup")
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# 2. Time + business-rule helpers (pure functions)
# --------------------------------------------------------------------------- #
def ms_to_dt(ms: int) -> datetime:
    """Epoch ms -> aware UTC datetime, truncated to whole seconds (R1)."""
    return datetime.fromtimestamp(ms // 1000, tz=UTC)


def aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    return (aware(dt) - EPOCH) // timedelta(milliseconds=1)  # exact integer arithmetic, no float


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def is_overnight(shift_start: str, shift_end: str) -> bool:
    return shift_end <= shift_start  # zero-padded HH:MM compares correctly as text


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> date:
    """R1: IST calendar date of the punch-in; an overnight shift's early-morning punch-in belongs to the previous day."""
    ist = aware(punch_in).astimezone(IST)
    d = ist.date()
    if is_overnight(shift_start, shift_end) and ist.time() < parse_hhmm(shift_end):
        d -= timedelta(days=1)
    return d


def shift_bounds(date_str: str, shift_start: str, shift_end: str) -> tuple[datetime, datetime]:
    """Start and end instants (IST) of the shift that belongs to attendance date `date_str`."""
    d = date.fromisoformat(date_str)
    end_day = d + timedelta(days=1) if is_overnight(shift_start, shift_end) else d
    return (
        datetime.combine(d, parse_hhmm(shift_start), tzinfo=IST),
        datetime.combine(end_day, parse_hhmm(shift_end), tzinfo=IST),
    )


def compute_late_minutes(punch_in: datetime, shift_start_dt: datetime) -> int:
    """R2: late only if strictly more than 10:00 after shift start; then whole minutes since shift start."""
    secs = int((punch_in - shift_start_dt).total_seconds())
    return secs // 60 if secs > GRACE_SECONDS else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: hours rounded to 2 decimals HALF-UP (Decimal, not float round())."""
    secs = int((punch_out - punch_in).total_seconds())
    return float((Decimal(secs) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute_overtime(punch_out: datetime, shift_end_dt: datetime) -> int:
    """R3: whole minutes after shift end, only when >= 30."""
    mins = int((punch_out - shift_end_dt).total_seconds()) // 60  # floor
    return mins if mins >= OVERTIME_MIN else 0


def derive(emp: dict, date_str: str, punch_in: Optional[datetime], punch_out: Optional[datetime]) -> dict:
    """All derived fields for a presence record (R2-R5)."""
    out = {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    if punch_in is None:
        return out
    start, end = shift_bounds(date_str, emp.get("shift_start", "09:30"), emp.get("shift_end", "18:30"))
    out["late_minutes"] = compute_late_minutes(punch_in, start)
    if punch_out is not None:
        out["work_hours"] = compute_work_hours(punch_in, punch_out)
        out["overtime_minutes"] = compute_overtime(punch_out, end)
        out["half_day"] = out["work_hours"] < HALF_DAY_LIMIT
    return out


def round_half_up(x, places: int) -> float:
    q = Decimal(1).scaleb(-places)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


# --------------------------------------------------------------------------- #
# 3. Models, validation helpers, serialisation
# --------------------------------------------------------------------------- #
HHMM = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"
DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
MONTH_RE = re.compile(r"[0-9]{4}-(0[1-9]|1[0-2])")

# strict=True: floats, strings and booleans are rejected (422); range rejects seconds-looking values
EpochMs = Annotated[int, Field(strict=True, ge=MIN_MS, le=MAX_MS)]
Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]


def unprocessable(msg: str) -> HTTPException:
    return HTTPException(422, msg)


def parse_date(s: str, name: str = "date") -> date:
    if not DATE_RE.fullmatch(s):
        raise unprocessable(f"{name} must be YYYY-MM-DD")
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise unprocessable(f"{name} is not a real calendar date")


def parse_month(s: str) -> tuple[date, date]:
    """'YYYY-MM' -> (first day, last day)."""
    if not MONTH_RE.fullmatch(s):
        raise unprocessable("month must be YYYY-MM")
    y, m = int(s[:4]), int(s[5:])
    first = date(y, m, 1)
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return first, nxt - timedelta(days=1)


def require(value, name: str):
    if value is None:
        raise unprocessable(f"query parameter '{name}' is required")
    return value


class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP[0-9]{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field("09:30", pattern=HHMM)
    shift_end: str = Field("18:30", pattern=HHMM)
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def _joined_on_is_date(cls, v: str) -> str:
        if not DATE_RE.fullmatch(v):
            raise ValueError("joined_on must be YYYY-MM-DD")
        date.fromisoformat(v)  # raises ValueError for 2026-02-30
        return v

    @model_validator(mode="after")
    def _shift_differs(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: PresenceStatus = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    status: Optional[Status] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)
    # derived fields sent by a client are ignored (pydantic ignores unknown keys by default)


def employee_out(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc.get("shift_start", "09:30"),
        "shift_end": doc.get("shift_end", "18:30"),
        "joined_on": doc["joined_on"],
        "created_at": dt_to_ms(doc.get("created_at")),
    }


def _history_value(field: str, v):
    return dt_to_ms(v) if field in ("punch_in", "punch_out") and v is not None else v


def history_out(h: dict) -> dict:
    return {
        "at": dt_to_ms(h["at"]),
        "by": h.get("by"),
        "reason": h.get("reason"),
        "changes": {
            f: {"from": _history_value(f, c.get("from")), "to": _history_value(f, c.get("to"))}
            for f, c in (h.get("changes") or {}).items()
        },
    }


def record_out(doc: dict) -> dict:
    """Mongo document -> API record. Applies the 'missing means default' rules; never exposes _id."""
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_ms(doc.get("punch_in")),
        "punch_out": dt_to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day", False)),
        "history": [history_out(h) for h in (doc.get("history") or [])],
    }


def get_employee_or_404(emp_code: str) -> dict:
    emp = db.employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    return emp


# --------------------------------------------------------------------------- #
# 4. Employees and attendance
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        db.command("ping")
        ensure_indexes()
    except Exception:
        return JSONResponse(status_code=503, content={"detail": "database unavailable"})
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = now_utc()
    try:
        db.employees.insert_one(doc)  # unique index decides the winner of concurrent creates
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = {"department": department} if department else {}
    total = db.employees.count_documents(q)  # total follows the filter (R10)
    cur = db.employees.find(q).sort("emp_code", ASCENDING).skip((page - 1) * page_size).limit(page_size)
    return {"items": [employee_out(d) for d in cur], "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = get_employee_or_404(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()
    d = attendance_date(ts, emp.get("shift_start", "09:30"), emp.get("shift_end", "18:30")).isoformat()
    doc = {
        "emp_code": body.emp_code,
        "date": d,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        **derive(emp, d, ts, None),
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)  # unique (emp_code, date) index: exactly one winner
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return record_out(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = get_employee_or_404(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()
    # most recent record whose punch_in is at or before punched_at (works across midnight)
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts}}, sort=[("punch_in", DESCENDING)]
    )
    if rec is None:
        raise HTTPException(404, "no punch-in found")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")
    pin = aware(rec["punch_in"])
    if ts <= pin:
        raise unprocessable("punched_at must be after the punch-in")
    if ts - pin > MAX_SHIFT:
        raise unprocessable("punched_at is more than 24 hours after the punch-in")
    fields = {"punch_out": ts, **derive(emp, rec["date"], pin, ts)}
    # the `punch_out: None` condition makes this atomic: of two simultaneous punch-outs only one matches
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None}, {"$set": fields}, return_document=ReturnDocument.AFTER
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return record_out(updated)


def attendance_find_spec(emp_code, date_from, date_to, status, page, page_size) -> dict:
    """Single place that builds the GET /attendance query; the explain endpoint reuses it."""
    df = parse_date(date_from, "date_from") if date_from else None
    dt = parse_date(date_to, "date_to") if date_to else None
    if df and dt and df > dt:
        raise unprocessable("date_from must not be after date_to")
    q: dict = {}
    if emp_code:
        q["emp_code"] = emp_code
    if df or dt:
        q["date"] = {}
        if df:
            q["date"]["$gte"] = date_from
        if dt:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status
    return {
        "filter": q,
        "sort": SON([("date", DESCENDING), ("emp_code", ASCENDING)]),
        "skip": (page - 1) * page_size,
        "limit": page_size,
    }


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    spec = attendance_find_spec(emp_code, date_from, date_to, status, page, page_size)
    coll = db.attendance_logs
    total = coll.count_documents(spec["filter"]) if spec["filter"] else coll.estimated_document_count()
    cur = coll.find(spec["filter"]).sort(list(spec["sort"].items())).skip(spec["skip"]).limit(spec["limit"])
    return {"items": [record_out(d) for d in cur], "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: str, body: RegularizeIn):
    d = parse_date(date)
    emp = get_employee_or_404(emp_code)
    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": d.isoformat()})
    if rec is None:
        raise HTTPException(404, "no attendance record for that date")

    old_status = rec["status"]
    old_pin = aware(rec["punch_in"]) if rec.get("punch_in") else None
    old_pout = aware(rec["punch_out"]) if rec.get("punch_out") else None
    new_status = body.status or old_status
    times_given = body.punch_in is not None or body.punch_out is not None

    if new_status in NON_PRESENCE:
        if times_given:
            raise unprocessable("ABSENT/LEAVE clear the punch times; do not supply punch_in or punch_out")
        new_pin = new_pout = None
    else:
        new_pin = ms_to_dt(body.punch_in) if body.punch_in is not None else old_pin
        new_pout = ms_to_dt(body.punch_out) if body.punch_out is not None else old_pout
        if new_pin is None:
            raise unprocessable("a presence status requires a punch_in")
        if body.punch_in is not None and attendance_date(
            new_pin, emp.get("shift_start", "09:30"), emp.get("shift_end", "18:30")
        ) != d:
            raise unprocessable("punch_in must stay on the record's attendance date")
        if new_pout is not None and times_given:
            if new_pout <= new_pin:
                raise unprocessable("punch_out must be after punch_in")
            if new_pout - new_pin > MAX_SHIFT:
                raise unprocessable("punch_out must be within 24 hours of punch_in")

    new = {"status": new_status, "punch_in": new_pin, "punch_out": new_pout, **derive(emp, rec["date"], new_pin, new_pout)}
    old = {
        "status": old_status,
        "punch_in": old_pin,
        "punch_out": old_pout,
        "work_hours": rec.get("work_hours"),
        "late_minutes": rec.get("late_minutes") or 0,
        "overtime_minutes": rec.get("overtime_minutes") or 0,
        "half_day": bool(rec.get("half_day", False)),
    }
    changes = {f: {"from": old[f], "to": new[f]} for f in new if old[f] != new[f]}
    if not changes:
        raise unprocessable("the request changes nothing")

    entry = {"at": now_utc(), "by": body.regularized_by, "reason": body.reason, "changes": changes}

    # Optimistic concurrency: the update only applies if the record is still exactly as we read it
    # (same status, times and the same number of history entries). Otherwise somebody else changed it
    # meanwhile and we return 409 instead of silently overwriting their change or losing their history entry.
    hist = rec.get("history")
    cond = {
        "_id": rec["_id"],
        "status": old_status,
        "punch_in": rec.get("punch_in"),
        "punch_out": rec.get("punch_out"),
        "history": {"$size": len(hist)} if isinstance(hist, list) else {"$exists": False},
    }
    updated = db.attendance_logs.find_one_and_update(
        cond, {"$set": new, "$push": {"history": entry}}, return_document=ReturnDocument.AFTER
    )
    if updated is None:
        raise HTTPException(409, "record was modified concurrently, retry")
    return record_out(updated)


# --------------------------------------------------------------------------- #
# 5. Aggregation pipelines + analytics endpoints
# --------------------------------------------------------------------------- #
# Small expression fragments shared by the pipelines. They read STORED values only (never recompute).
IS_PRESENT = {"$in": ["$status", PRESENCE]}
LATE_MIN = {"$ifNull": ["$late_minutes", 0]}
IS_WEEKDAY = {  # $dayOfWeek: 1 = Sunday ... 7 = Saturday  -> Mon-Fri = 2..6
    "$in": [{"$dayOfWeek": {"$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}}}, [2, 3, 4, 5, 6]]
}
PRESENT_VALUE = {"$cond": [IS_PRESENT, {"$cond": [{"$ifNull": ["$half_day", False]}, 0.5, 1]}, 0]}  # R5/R6, any day
PRESENT_WEEKDAY_VALUE = {"$cond": [IS_WEEKDAY, PRESENT_VALUE, 0]}  # R7
LATE_FLAG = {"$cond": [{"$gt": [LATE_MIN, 0]}, 1, 0]}
HAS_HOURS = {"$and": [IS_PRESENT, {"$isNumber": "$work_hours"}]}
# work_hours in whole hundredths (exact integers), so sums/averages are exact and can be rounded half-up later
HOURS_CENTS = {"$cond": [HAS_HOURS, {"$round": [{"$multiply": ["$work_hours", 100]}, 0]}, 0]}


def count_if(status: str) -> dict:
    return {"$cond": [{"$eq": ["$status", status]}, 1, 0]}


def working_days(first: date, last: date, joined_on: str) -> int:
    """R7: Mon-Fri days in [first, last], counted from joined_on for a mid-month joiner."""
    start = max(first, date.fromisoformat(joined_on))
    n, d = 0, start
    while d <= last:
        n += d.weekday() < 5
        d += timedelta(days=1)
    return n


def monthly_pipeline(emp_code: str, month: str) -> list:
    first, last = parse_month(month)
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},
        {
            "$group": {
                "_id": None,
                "present_days": {"$sum": PRESENT_WEEKDAY_VALUE},
                "leave_days": {"$sum": count_if("LEAVE")},
                "late_count": {"$sum": LATE_FLAG},
                "total_late_minutes": {"$sum": LATE_MIN},
                "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
            }
        },
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(..., pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$")):
    emp = get_employee_or_404(emp_code)
    first, last = parse_month(month)
    agg = next(db.attendance_logs.aggregate(monthly_pipeline(emp_code, month)), None) or {}
    wd = working_days(first, last, emp["joined_on"])
    present = float(agg.get("present_days", 0))
    pct = None if wd == 0 else round_half_up(Decimal(str(present)) * 100 / wd, 2)
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": wd,
        "present_days": present,
        "leave_days": int(agg.get("leave_days", 0)),
        "late_count": int(agg.get("late_count", 0)),
        "total_late_minutes": int(agg.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(agg.get("total_overtime_minutes", 0)),
        "attendance_pct": pct,
    }


def summary_pipeline(month: str, department: Optional[str]) -> list:
    """Starts from EMPLOYEES so people with zero logs are counted; logs are joined per employee."""
    first, last = parse_month(month)
    match: dict = {"joined_on": {"$lte": last.isoformat()}}  # R9
    if department:
        match["department"] = department
    return [
        {"$match": match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "pipeline": [
                    {"$match": {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},
                    {
                        "$group": {
                            "_id": None,
                            "present_days": {"$sum": PRESENT_WEEKDAY_VALUE},
                            "hours_cents": {"$sum": HOURS_CENTS},
                            "hours_n": {"$sum": {"$cond": [HAS_HOURS, 1, 0]}},
                            "late_count": {"$sum": LATE_FLAG},
                            "total_late_minutes": {"$sum": LATE_MIN},
                            "leave_count": {"$sum": count_if("LEAVE")},
                            "on_duty_count": {"$sum": count_if("ON_DUTY")},
                        }
                    },
                ],
                "as": "a",
            }
        },
        {"$unwind": {"path": "$a", "preserveNullAndEmptyArrays": True}},  # keep employees with no logs
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                **{
                    k: {"$sum": {"$ifNull": [f"$a.{k}", 0]}}
                    for k in (
                        "present_days", "hours_cents", "hours_n", "late_count",
                        "total_late_minutes", "leave_count", "on_duty_count",
                    )
                },
            }
        },
        {"$sort": {"_id": 1}},
    ]


@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(..., pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$"), department: Optional[str] = None
):
    items = []
    for r in db.employees.aggregate(summary_pipeline(month, department), allowDiskUse=True):
        n = int(r["hours_n"])
        avg = None if n == 0 else round_half_up(Decimal(int(r["hours_cents"])) / n / 100, 2)  # mean over RECORDS
        items.append(
            {
                "department": r["_id"],
                "headcount": int(r["headcount"]),
                "present_days": float(r["present_days"]),
                "avg_work_hours": avg,
                "late_count": int(r["late_count"]),
                "total_late_minutes": int(r["total_late_minutes"]),
                "leave_count": int(r["leave_count"]),
                "on_duty_count": int(r["on_duty_count"]),
            }
        )
    return {"month": month, "items": items}


def leaderboard_pipeline(month: str, limit: int, department: Optional[str]) -> list:
    first, last = parse_month(month)
    p: list = [
        {"$match": {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}, "late_minutes": {"$gt": 0}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"}, "late_count": {"$sum": 1}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "e"}},
        {"$unwind": "$e"},  # logs of unknown employees are dropped
    ]
    if department:
        p.append({"$match": {"e.department": department}})  # rank only inside the department
    p += [
        {
            "$project": {
                "_id": 0, "emp_code": "$_id", "name": "$e.name", "department": "$e.department",
                "total_late_minutes": 1, "late_count": 1,
            }
        },
        # $rank = standard competition ranking (1,2,2,4); ties share a rank and the next rank is skipped
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},  # limit applies AFTER ranking, so ties at the cutoff are all kept
        {"$sort": {"total_late_minutes": -1, "emp_code": 1}},
    ]
    return p


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    rows = db.attendance_logs.aggregate(leaderboard_pipeline(month, limit, department), allowDiskUse=True)
    return {
        "month": month,
        "items": [
            {
                "rank": int(r["rank"]), "emp_code": r["emp_code"], "name": r["name"],
                "department": r["department"], "total_late_minutes": int(r["total_late_minutes"]),
                "late_count": int(r["late_count"]),
            }
            for r in rows
        ],
    }


def trend_pipeline(department: str, d_from: date, d_to: date) -> list:
    """One row per calendar day, built inside MongoDB ($range + $map), then a 7-row moving average.

    It starts from EMPLOYEES (index on department) so a department with no logs still yields every day.
    """
    n = (d_to - d_from).days + 1
    f, t = d_from.isoformat(), d_to.isoformat()
    start = datetime(d_from.year, d_from.month, d_from.day, tzinfo=UTC)
    day = {"$dateAdd": {"startDate": start, "unit": "day", "amount": "$$i"}}
    # attendance_rate = present/headcount rounded HALF-UP to 4 decimals, using exact Decimal128 arithmetic
    rate = {
        "$divide": [
            {
                "$floor": {
                    "$add": [
                        {"$multiply": [{"$divide": [{"$toDecimal": "$present_count"}, {"$toDecimal": "$headcount"}]}, 10000]},
                        Decimal128("0.5"),
                    ]
                }
            },
            10000,
        ]
    }
    return [
        {"$match": {"department": department, "joined_on": {"$lte": t}}},
        {"$project": {"_id": 0, "emp_code": 1, "joined_on": 1}},
        {
            "$facet": {
                "joins": [{"$group": {"_id": "$joined_on", "n": {"$sum": 1}}}],
                "activity": [
                    {
                        "$lookup": {
                            "from": "attendance_logs",
                            "localField": "emp_code",
                            "foreignField": "emp_code",
                            "pipeline": [
                                {"$match": {"date": {"$gte": f, "$lte": t}}},
                                {"$group": {"_id": "$date", "present": {"$sum": PRESENT_VALUE}, "late": {"$sum": LATE_FLAG}}},
                            ],
                            "as": "l",
                        }
                    },
                    {"$unwind": "$l"},
                    {"$group": {"_id": "$l._id", "present": {"$sum": "$l.present"}, "late": {"$sum": "$l.late"}}},
                ],
            }
        },
        {
            "$project": {
                "rows": {
                    "$map": {
                        "input": {"$range": [0, n]},
                        "as": "i",
                        "in": {
                            "$let": {
                                "vars": {"dt": day},
                                "in": {
                                    "$let": {
                                        "vars": {"ds": {"$dateToString": {"format": "%Y-%m-%d", "date": "$$dt"}}},
                                        "in": {
                                            "date": "$$ds",
                                            "dow": {"$dayOfWeek": "$$dt"},
                                            "headcount": {  # R9: joined on or before that day
                                                "$sum": {
                                                    "$map": {
                                                        "input": {
                                                            "$filter": {
                                                                "input": "$joins", "as": "j",
                                                                "cond": {"$lte": ["$$j._id", "$$ds"]},
                                                            }
                                                        },
                                                        "as": "j",
                                                        "in": "$$j.n",
                                                    }
                                                }
                                            },
                                            "act": {
                                                "$arrayElemAt": [
                                                    {"$filter": {"input": "$activity", "as": "a", "cond": {"$eq": ["$$a._id", "$$ds"]}}},
                                                    0,
                                                ]
                                            },
                                        },
                                    }
                                },
                            }
                        },
                    }
                }
            }
        },
        {"$unwind": "$rows"},
        {"$replaceRoot": {"newRoot": "$rows"}},
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": {"$and": [{"$gte": ["$dow", 2]}, {"$lte": ["$dow", 6]}]},
                "headcount": 1,
                "present_count": {"$ifNull": ["$act.present", 0]},
                "late_count": {"$ifNull": ["$act.late", 0]},
            }
        },
        {
            "$addFields": {
                "attendance_rate": {"$cond": [{"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]}, rate, None]}
            }
        },
        # mean of the non-null rates in the current row and up to 6 preceding rows (shorter at the start)
        {
            "$setWindowFields": {
                "sortBy": {"date": 1},
                "output": {"moving_avg_7d": {"$avg": "$attendance_rate", "window": {"documents": [-6, 0]}}},
            }
        },
        {"$sort": {"date": 1}},
    ]


def trend_params(department: str, d_from: str, d_to: str) -> tuple[date, date]:
    f, t = parse_date(d_from, "from"), parse_date(d_to, "to")
    if t < f:
        raise unprocessable("'to' must not be before 'from'")
    if (t - f).days + 1 > 92:
        raise unprocessable("range must not exceed 92 days")
    return f, t


def _num(v) -> Optional[Decimal]:
    return None if v is None else (v.to_decimal() if isinstance(v, Decimal128) else Decimal(str(v)))


@app.get("/analytics/departments/{department}/trend")
def department_trend(department: str, from_: str = Query(..., alias="from"), to: str = Query(...)):
    f, t = trend_params(department, from_, to)
    if db.employees.find_one({"department": department}, {"_id": 1}) is None:
        raise HTTPException(404, "unknown department")
    items = []
    for r in db.employees.aggregate(trend_pipeline(department, f, t), allowDiskUse=True):
        rate, avg = _num(r.get("attendance_rate")), _num(r.get("moving_avg_7d"))
        items.append(
            {
                "date": r["date"],
                "is_working_day": bool(r["is_working_day"]),
                "headcount": int(r["headcount"]),
                "present_count": float(r["present_count"]),
                "late_count": int(r["late_count"]),
                "attendance_rate": None if rate is None else float(rate),  # already rounded in the pipeline
                "moving_avg_7d": None if avg is None else round_half_up(avg, 4),
            }
        )
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# 6. Explain endpoint: runs explain on the SAME query/pipeline the endpoint builds
# --------------------------------------------------------------------------- #
@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"],
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(None, pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if endpoint == "attendance_list":
        spec = attendance_find_spec(emp_code, date_from, date_to, status, page, page_size)
        coll, inner = "attendance_logs", {"find": "attendance_logs", **spec}
    else:
        if endpoint == "employee_monthly":
            coll, pipeline = "attendance_logs", monthly_pipeline(require(emp_code, "emp_code"), require(month, "month"))
        elif endpoint == "department_summary":
            coll, pipeline = "employees", summary_pipeline(require(month, "month"), department)
        elif endpoint == "late_leaderboard":
            coll, pipeline = "attendance_logs", leaderboard_pipeline(require(month, "month"), limit, department)
        else:  # department_trend
            dep = require(department, "department")
            f, t = trend_params(dep, require(from_, "from"), require(to, "to"))
            coll, pipeline = "employees", trend_pipeline(dep, f, t)
        inner = {"aggregate": coll, "pipeline": pipeline, "cursor": {}, "allowDiskUse": True}
    raw = db.command(SON([("explain", inner), ("verbosity", "executionStats")]))
    safe = json.loads(json_util.dumps(raw, json_options=json_util.RELAXED_JSON_OPTIONS))
    return {"endpoint": endpoint, "collection": coll, "explain": safe}
