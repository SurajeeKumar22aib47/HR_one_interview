# Employee Attendance & Analytics API

FastAPI + MongoDB. All code is in `app/main.py`.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MONGO_URI="mongodb://localhost:27017" MONGO_DB="attendance_db"   # or put them in a local .env
uvicorn app.main:app --port 8000
```
Indexes are created at startup (idempotent). MongoDB 6.0+ is required (`$setWindowFields`, `$rank`, `$lookup` with `localField` + `pipeline`).

## Tests
`pip install pytest httpx mongomock && pytest tests` covers the business rules (R1-R5) and the employee / punch / list / regularize flows
against an in-memory fake. Analytics need a real MongoDB.

## Status
All endpoints in `openapi.yaml` are implemented, including `/admin/explain/{endpoint}`.

Honest notes:
- <FILL IN what you tested against a real MongoDB and what you did not.>
