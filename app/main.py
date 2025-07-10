import csv
import io
import logging
import re
import secrets
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from io import BytesIO, TextIOWrapper
from math import ceil
from typing import List, Optional

import bcrypt
import chardet
from dateutil.relativedelta import relativedelta
from fastapi import (
    Body,
    Cookie,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.security import HTTPBasic
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, URLSafeSerializer
from passlib.context import CryptContext
from pydantic import BaseModel, ConfigDict, EmailStr, Field, validator
from sqlalchemy import extract, func, inspect
from sqlalchemy.orm import Session, aliased

from app import database, models
from app.database import engine
from app.reports.generator import fetch_user_sessions, generate_pdf_report

from .config import APP_INFO, APP_NAME, BASE_URL, SECRET_KEY, VERSION
from .utils.email import send_email
from .utils.session_csv import convert_csv_row

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title=f"{APP_NAME} API",
    description=f"REST API for {APP_NAME} management",
    version=VERSION,
)
templates = Jinja2Templates(directory="app/templates")
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Mount the static directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Create tables on startup
models.Base.metadata.create_all(bind=engine)

# Serializer for cookie sessions
cookie_serializer = URLSafeSerializer(SECRET_KEY, salt="session")
reset_serializer = URLSafeSerializer(SECRET_KEY, salt="reset")
TOKEN_PATTERN = re.compile(
    r"^[A-Za-z0-9\-_]{20,}$"
)  # URL-safe base64-ish, min length 20


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


# Dependency: DB session
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db_if_empty():
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    if not existing_tables:
        logger.info("Database is empty. Creating all tables...")
        models.Base.metadata.create_all(bind=engine)
        logger.info("Tables created.")
    else:
        logger.info("Database already initialized. Skipping table creation.")


# Automatically call this on module import or app startup
init_db_if_empty()


def make_aware(dt):
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# --- Authentication Helpers ---


def create_auth_cookie(response: Response, username: str):
    token = cookie_serializer.dumps({"username": username})
    response.set_cookie(key="session", value=token, httponly=True)


def get_user_from_cookie(
    session: str = Cookie(default=None), db: Session = Depends(get_db)
):
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        data = cookie_serializer.loads(session)
        user = db.query(models.User).filter_by(username=data["username"]).first()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session")


def get_user_by_api_key(x_api_key: str = Header(...), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.api_key == x_api_key).first()
    if not user:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return user


# --- API Key Auth for device endpoints ---


def get_charger_by_api_key(x_api_key: str = Header(...), db: Session = Depends(get_db)):
    charger = (
        db.query(models.Charger).filter(models.Charger.api_key == x_api_key).first()
    )
    if not charger:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return charger


# --- Schemas ---


class UserRegisterIn(BaseModel):
    username: str
    password: str
    email: EmailStr  # Also validates proper email format

    @validator("username", "email", pre=True)
    def strip_whitespace(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

    @validator("email", pre=True)
    def lowercase_email(cls, v):
        return v.lower() if isinstance(v, str) else v


class ChargerRegisterIn(BaseModel):
    charger_id: str = Field(..., description="Unique identifier for the charger")
    description: str = Field(..., description="Human-readable charger description")
    cost_kwh: Optional[float] = Field(
        None, ge=0.0, description="Cost per kWh in your currency"
    )
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class ChargerPutIn(BaseModel):
    description: str = Field(..., description="Human-readable charger description")
    cost_kwh: Optional[float] = Field(
        None, ge=0.0, description="Cost per kWh in your currency"
    )
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class ChargingEventIn(BaseModel):
    event_type: str
    charger_id: str


class ApikeyOut(BaseModel):
    api_key: str


class ChargerOut(BaseModel):
    id: int
    charger_id: str
    cost_kwh: float
    description: str
    registered_at: datetime
    latitude: Optional[float]
    longitude: Optional[float]
    owner_id: int


class UserOut(BaseModel):
    username: str
    email: str
    api_key: str
    created_at: datetime


class UserInfoOut(BaseModel):
    username: str
    email: str
    api_key: str
    created_at: datetime


class MinimalChargingSessionIn(BaseModel):
    charger_id: str
    cost: Optional[float] = None
    tag: Optional[str] = None
    seconds: int = Field(..., ge=0)
    energy: float = Field(..., ge=0)


class ChargingSessionIn(BaseModel):
    charger_id: str
    cost: Optional[float] = None
    tag: Optional[str] = None
    start_time: datetime
    end_time: datetime
    duration_seconds: int
    energy_charged_kwh: float


class ChargingSessionOut(BaseModel):
    session_uuid: str
    charger_id: str
    cost: float
    tag: Optional[str] = None
    start_time: datetime
    end_time: datetime
    duration_seconds: int
    energy_charged_kwh: float
    model_config = ConfigDict(from_attributes=True)


class ChargingEventOut(BaseModel):
    charger_id: str
    event_type: str
    timestamp: datetime
    model_config = ConfigDict(from_attributes=True)


class ChargingStatusIn(BaseModel):
    charger_id: str
    seconds: int
    energy: float
    status: str
    cost: Optional[float] = None
    phase_count: int


class LastSessionOut(BaseModel):
    session_uuid: str
    charger_id: str
    cost: float
    tag: Optional[str] = None
    start_time: datetime
    end_time: datetime
    duration_seconds: int
    energy_charged_kwh: float
    model_config = ConfigDict(from_attributes=True)


class LastEventOut(BaseModel):
    event_type: str
    timestamp: datetime
    model_config = ConfigDict(from_attributes=True)


class ChargerInfoOut(BaseModel):
    charger_id: str
    cost_kwh: float
    registered_at: datetime
    session_count: int
    last_session: Optional[LastSessionOut]
    last_event: Optional[LastEventOut]
    model_config = ConfigDict(from_attributes=True)


class UserAuthIn(BaseModel):
    username: str
    password: str

    @validator("username", pre=True)
    def strip_username(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

    @classmethod
    def as_form(
        cls,
        username: str = Form(...),
        password: str = Form(...),
    ):
        return cls(username=username, password=password)


# --- User Registration ---


@app.get("/user/register", response_class=HTMLResponse, tags=["User"])
def show_registration_form(request: Request, msg: Optional[str] = None):
    return templates.TemplateResponse(
        "register.html",
        {
            "request": request,
            "msg": msg,
            "now": datetime.utcnow,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
        },
    )


@app.post("/user/register", tags=["User"])
def register_user(info: UserRegisterIn, db: Session = Depends(get_db)):
    if db.query(models.User).filter_by(username=info.username).first():
        raise HTTPException(400, detail="Username already exists")
    if db.query(models.User).filter_by(email=info.email).first():
        raise HTTPException(400, detail="Email already exists")

    hashed_pwd = bcrypt.hashpw(info.password.encode(), bcrypt.gensalt()).decode()
    new_user = models.User(
        username=info.username,
        password=hashed_pwd,
        email=info.email,
        api_key=secrets.token_hex(16),
    )
    db.add(new_user)
    db.commit()

    token = cookie_serializer.dumps({"username": new_user.username})
    response = JSONResponse(
        content={"status": "registered", "username": new_user.username}
    )
    response.set_cookie(key="session", value=token, httponly=True, max_age=3600)
    return response


# --- Password Reset ---


@app.get("/user/reset-password", response_class=HTMLResponse, tags=["User"])
def show_password_reset_form(request: Request, msg: Optional[str] = None):
    return templates.TemplateResponse(
        "reset_password.html",
        {
            "request": request,
            "msg": msg,
            "now": datetime.utcnow,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
        },
    )


@app.post("/user/reset-password", tags=["User"])
def send_password_reset(email: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(email=email).first()
    if user:
        user.pwd_reset_requested_at = datetime.utcnow()
        db.commit()
        token = reset_serializer.dumps(
            {"email": user.email, "ts": user.pwd_reset_requested_at.isoformat()}
        )

        send_email(
            to=user.email,
            subject=f"{APP_NAME} Password Reset",
            template_name="base.html",
            context={
                "subject": f"Reset Your {APP_NAME} Password",
                "heading": "Password Reset Request",
                "message": "Click the button below to reset your password.",
                "action_url": f"{BASE_URL}/user/reset-password/{token}",
                "action_text": "Reset Password",
                "year": datetime.utcnow().year,
            },
        )

    # Always redirect with the message (even if user was not found)
    return RedirectResponse(url="/dashboard/login?msg=sent", status_code=303)


@app.get("/user/reset-password/{token}", response_class=HTMLResponse, tags=["User"])
def show_password_reset_token_form(
    request: Request,
    token: str,
    msg: Optional[str] = None,
    db: Session = Depends(get_db),
):
    try:
        data = reset_serializer.loads(token)
        email = data.get("email")
        user = db.query(models.User).filter_by(email=email).first()
        if not user:
            raise HTTPException(404, detail="User not found")
        token_ts = datetime.fromisoformat(data["ts"])
        if not user.pwd_reset_requested_at or token_ts != user.pwd_reset_requested_at:
            raise HTTPException(400, detail="Invalid or expired token")
        return templates.TemplateResponse(
            "reset_password_token.html",
            {
                "request": request,
                "token": token,
                "msg": msg,
                "email": email,
                "now": datetime.utcnow,
                "appName": APP_NAME,
                "appInfo": APP_INFO,
            },
        )
    except BadSignature:
        raise HTTPException(status_code=400, detail="Invalid or expired token")


@app.post("/user/reset-password/{token}", tags=["User"])
def reset_password_token(
    token: str, password: str = Form(...), db: Session = Depends(get_db)
):
    try:
        data = reset_serializer.loads(token)
        user = db.query(models.User).filter_by(email=data["email"]).first()
        if not user:
            raise HTTPException(404, detail="User not found")
        token_ts = datetime.fromisoformat(data["ts"])
        if not user.pwd_reset_requested_at or token_ts != user.pwd_reset_requested_at:
            raise HTTPException(400, detail="Invalid or expired token")
        hashed_pwd = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        user.password = hashed_pwd
        user.pwd_reset_requested_at = None
        db.commit()
        return RedirectResponse("/dashboard/login?msg=pwdreset", status_code=303)
    except BadSignature:
        raise HTTPException(status_code=400, detail="Invalid or expired token")


@app.get("/user/username-recovery", response_class=HTMLResponse, tags=["User"])
def show_username_recovery_form(request: Request):
    return templates.TemplateResponse(
        "username_recovery.html",
        {
            "request": request,
            "now": datetime.utcnow,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
        },
    )


@app.post("/user/username-recovery", tags=["User"])
def send_username_recovery(email: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(email=email).first()

    if user:
        send_email(
            to=user.email,
            subject=f"{APP_NAME} Password Reset",
            template_name="base.html",
            context={
                "subject": f"{APP_NAME} - Username Recovery",
                "heading": "Username Recovery Request",
                "message": f"<p>Your username is: </p>{user.username}",
                "action_url": f"{BASE_URL}/dashboard/login",
                "action_text": "Login",
                "year": datetime.utcnow().year,
            },
        )

    # Always redirect regardless of match
    return RedirectResponse("/dashboard/login?msg=sentusername", status_code=303)


@app.get("/user/generate_api_key", response_model=ApikeyOut, tags=["User"])
def generate_api_key(db: Session = Depends(get_db), user=Depends(get_user_by_api_key)):
    user.api_key = secrets.token_hex(16)
    db.commit()
    return {"status": "success", "api_key": user.api_key}


@app.delete("/user/{username}", tags=["User"])
def delete_user(
    username: str, db: Session = Depends(get_db), auth_user=Depends(get_user_by_api_key)
):
    # Optionally restrict deletion to self or admins
    if username != auth_user.username:
        raise HTTPException(403, detail="You can only delete your own account.")

    user = db.query(models.User).filter_by(username=username).first()
    if not user:
        raise HTTPException(404, detail="User not found")

    db.delete(user)
    db.commit()

    return {"status": "deleted", "username": username}


@app.get("/users", response_model=List[UserOut], tags=["User"])
def list_users(db: Session = Depends(get_db)):
    users = db.query(models.User).all()
    return users


# --- Charger Registration ---


@app.get("/chargers", response_model=List[ChargerOut], tags=["Charger"])
def list_chargers(db: Session = Depends(get_db), user=Depends(get_user_by_api_key)):
    chargers = db.query(models.Charger).filter_by(owner_id=user.id).all()
    return chargers


@app.post("/charger/register", tags=["Charger"])
def register_charger(
    info: ChargerRegisterIn,
    user=Depends(get_user_by_api_key),
    db: Session = Depends(get_db),
):
    if db.query(models.Charger).filter_by(charger_id=info.charger_id).first():
        raise HTTPException(400, detail="Charger ID already exists")
    new = models.Charger(
        charger_id=info.charger_id,
        description=info.description,
        cost_kwh=info.cost_kwh if info.cost_kwh is not None else 0.0,
        registered_at=datetime.utcnow(),
        latitude=info.latitude,
        longitude=info.longitude,
        owner=user,
    )
    db.add(new)
    db.commit()
    return {"status": "registered", "charger": new}


@app.put("/charger/{charger_id}", tags=["Charger"])
def update_charger(
    charger_id: str,
    data: ChargerPutIn,
    db: Session = Depends(get_db),
    user=Depends(get_user_by_api_key),
):
    charger = (
        db.query(models.Charger)
        .filter_by(charger_id=charger_id, owner_id=user.id)
        .first()
    )
    if not charger:
        raise HTTPException(404, "Charger not found or not owned by user")

    charger.description = data.description
    charger.latitude = data.latitude
    charger.longitude = data.longitude
    charger.cost_kwh = data.cost_kwh if data.cost_kwh is not None else 0.0
    db.commit()
    return {"status": "updated"}


@app.delete("/charger/{charger_id}", tags=["Charger"])
def delete_charger(
    charger_id: str = Path(...),
    db: Session = Depends(get_db),
    user=Depends(get_user_by_api_key),
):
    # Find charger owned by this user
    charger = (
        db.query(models.Charger)
        .filter_by(charger_id=charger_id, owner_id=user.id)
        .first()
    )
    if not charger:
        raise HTTPException(
            status_code=404, detail="Charger not found or not owned by user"
        )

    # Optional: delete related sessions and events
    db.query(models.ChargingSession).filter_by(charger_id=charger.id).delete()
    db.query(models.ChargingEvent).filter_by(charger_id=charger.id).delete()

    db.delete(charger)
    db.commit()

    return {"status": "deleted", "charger_id": charger_id}


# --- Device endpoints (API key auth) ---


@app.post("/charger/status", tags=["Charger"])
def receive_charging_status(
    status: ChargingStatusIn,
    db: Session = Depends(get_db),
    auth_user=Depends(get_user_by_api_key),
):
    # Verify charger ownership
    charger_obj = (
        db.query(models.Charger)
        .filter_by(charger_id=status.charger_id, owner_id=auth_user.id)
        .first()
    )
    # If not found, create it with minimal info
    if not charger_obj:
        charger_obj = models.Charger(
            charger_id=status.charger_id,
            description="Auto-created via /charger/event",
            latitude=50.744617845784745,  # Default Powerdale coordinates
            longitude=4.346385626853893,
            registered_at=datetime.utcnow(),
            owner_id=auth_user.id,
        )
        db.add(charger_obj)
        db.commit()
        db.refresh(charger_obj)

        send_email(
            to=auth_user.email,
            subject=f"{APP_NAME} - New Charger Auto-Created",
            template_name="base.html",
            context={
                "subject": f"{APP_NAME} - New Charger Auto-Created",
                "heading": "New Charger Auto-Created",
                "message": f"""
                        <p>Hello {auth_user.username},</p>
                        <p>A new charger has been automatically registered for your account:</p>
                        <ul>
                        <li><strong>Charger ID:</strong> {charger_obj.charger_id}</li>
                        <li><strong>Location:</strong> {charger_obj.latitude}, {charger_obj.longitude}</li>
                        <li><strong>Description:</strong> {charger_obj.description}</li>
                        </ul>
                    """,
                "action_url": f"{BASE_URL}/dashboard",
                "action_text": "Dasboard",
                "year": datetime.utcnow().year,
            },
        )

    existing = (
        db.query(models.ChargingStatus)
        .filter_by(charger_id=charger_obj.id)
        .order_by(models.ChargingStatus.timestamp.desc())
        .first()
    )

    cost_value = (
        status.cost if status.cost is not None else charger_obj.cost_kwh * status.energy
    )

    if existing:
        existing.seconds = status.seconds
        existing.energy = status.energy
        existing.status = status.status
        existing.cost = cost_value
        existing.phase_count = status.phase_count
        existing.timestamp = datetime.utcnow()
    else:
        existing = models.ChargingStatus(
            charger_id=charger_obj.id,
            seconds=status.seconds,
            energy=status.energy,
            status=status.status,
            cost=cost_value,
            phase_count=status.phase_count,
            timestamp=datetime.utcnow(),
        )
        db.add(existing)

    db.commit()

    return {"status": "logged"}


@app.get("/charger/status/{charger_id}", tags=["Charger"])
def get_latest_charging_status(
    charger_id: str = Path(...),
    db: Session = Depends(get_db),
    user=Depends(get_user_by_api_key),
):
    # Get the charger's DB record
    charger = (
        db.query(models.Charger)
        .filter_by(charger_id=charger_id, owner_id=user.id)
        .first()
    )
    if not charger:
        raise HTTPException(
            status_code=404, detail="Charger not found or not owned by user"
        )

    # Get latest status for this charger
    latest = (
        db.query(models.ChargingStatus)
        .filter_by(charger_id=charger.id)
        .order_by(models.ChargingStatus.timestamp.desc())
        .first()
    )

    if not latest:
        return {
            "status": "not_found",
            "message": "No status logged yet for this charger.",
        }

    return {
        "charger_id": charger.charger_id,
        "timestamp": latest.timestamp,
        "seconds": latest.seconds,
        "energy": latest.energy,
        "status": latest.status,
        "phase_count": latest.phase_count,
    }


@app.post("/charger/event", tags=["Charger Event"])
def charger_event(
    event: ChargingEventIn,
    db: Session = Depends(get_db),
    auth_user=Depends(get_user_by_api_key),  # this gets the User from API key
):
    # Find charger owned by this user with the given charger_id
    charger_obj = (
        db.query(models.Charger)
        .filter(
            models.Charger.charger_id == event.charger_id,
            models.Charger.owner_id == auth_user.id,
        )
        .first()
    )

    # If not found, create it with minimal info
    if not charger_obj:
        charger_obj = models.Charger(
            charger_id=event.charger_id,
            description="Auto-created via /charger/event",
            latitude=50.744617845784745,  # Default Powerdale coordinates
            longitude=4.346385626853893,
            registered_at=datetime.utcnow(),
            owner_id=auth_user.id,
        )
        db.add(charger_obj)
        db.commit()
        db.refresh(charger_obj)

        send_email(
            to=auth_user.email,
            subject=f"{APP_NAME} - New Charger Auto-Created",
            template_name="base.html",
            context={
                "subject": f"{APP_NAME} - New Charger Auto-Created",
                "heading": "New Charger Auto-Created",
                "message": f"""
                        <p>Hello {auth_user.username},</p>
                        <p>A new charger has been automatically registered for your account:</p>
                        <ul>
                        <li><strong>Charger ID:</strong> {charger_obj.charger_id}</li>
                        <li><strong>Location:</strong> {charger_obj.latitude}, {charger_obj.longitude}</li>
                        <li><strong>Description:</strong> {charger_obj.description}</li>
                        </ul>
                    """,
                "action_url": f"{BASE_URL}/dashboard",
                "action_text": "Dasboard",
                "year": datetime.utcnow().year,
            },
        )

    # Get the most recent event for this charger
    last_event = (
        db.query(models.ChargingEvent)
        .filter(models.ChargingEvent.charger_id == charger_obj.id)
        .order_by(models.ChargingEvent.timestamp.desc())
        .first()
    )

    # Skip saving if the event_type is the same as the most recent
    if last_event and last_event.event_type == event.event_type:
        return {"status": "ignored", "reason": "duplicate event_type"}

    # Prepare event data, replace charger_id (string) with internal charger.id (int)
    event_data = event.dict()
    event_data["charger_id"] = charger_obj.id
    event_data["timestamp"] = datetime.now()

    db_session = models.ChargingEvent(**event_data)
    db.add(db_session)
    db.commit()

    return {"status": "saved"}


@app.get(
    "/charger/events", response_model=List[ChargingEventOut], tags=["Charger Event"]
)
def charger_events(
    charger_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user=Depends(get_user_by_api_key),
):
    # Get all chargers owned by this user
    chargers = db.query(models.Charger).filter(models.Charger.owner_id == user.id).all()
    charger_ids = [c.id for c in chargers]  # internal DB ids
    charger_map = {c.id: c.charger_id for c in chargers}  # id -> string map

    # If a specific charger_id is requested, validate ownership
    if charger_id:
        matching = [c for c in chargers if c.charger_id == charger_id]
        if not matching:
            raise HTTPException(
                status_code=404, detail="Charger not found for this user"
            )
        charger_ids = [matching[0].id]

    # Fetch events
    events = (
        db.query(models.ChargingEvent)
        .filter(models.ChargingEvent.charger_id.in_(charger_ids))
        .order_by(models.ChargingEvent.timestamp.desc())
        .all()
    )

    # Convert DB objects to response models using the charger_map for external IDs
    return [
        ChargingEventOut(
            charger_id=charger_map[e.charger_id],
            event_type=e.event_type,
            timestamp=e.timestamp,
        )
        for e in events
    ]


@app.post("/charger/session/minimal", tags=["Charger Session"])
def charger_session_minimal(
    session: MinimalChargingSessionIn,
    db: Session = Depends(get_db),
    auth_user=Depends(get_user_by_api_key),
):
    # Ensure charger belongs to the authenticated user
    charger_obj = (
        db.query(models.Charger)
        .filter(
            models.Charger.charger_id == session.charger_id,
            models.Charger.owner_id == auth_user.id,
        )
        .first()
    )
    if not charger_obj:
        raise HTTPException(
            status_code=404, detail="Charger not found or not owned by user"
        )

    # Compute start and end times
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(seconds=session.seconds)

    # Calculate cost
    cost_value = (
        session.cost
        if session.cost is not None
        else charger_obj.cost_kwh * session.energy
    )

    # Create and store ChargingSession
    db_session = models.ChargingSession(
        charger_id=charger_obj.id,
        tag=session.tag,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=session.seconds,
        energy_charged_kwh=session.energy,
        cost=cost_value,
    )

    db.add(db_session)
    db.commit()

    return {"status": "saved", "session_uuid": db_session.session_uuid}


@app.post("/charger/session", tags=["Charger Session"])
def charger_session(
    session: ChargingSessionIn,
    db: Session = Depends(get_db),
    auth_user=Depends(get_user_by_api_key),  # this gets the User from API key
):
    # Find charger owned by this user with the given charger_id
    charger_obj = (
        db.query(models.Charger)
        .filter(
            models.Charger.charger_id == session.charger_id,
            models.Charger.owner_id == auth_user.id,
        )
        .first()
    )
    if not charger_obj:
        raise HTTPException(
            status_code=404, detail="Charger not found or not owned by user"
        )

    # Prepare session data, replace charger_id (string) with internal charger.id (int)
    session_data = session.dict()
    session_data["charger_id"] = charger_obj.id

    db_session = models.ChargingSession(**session_data)
    db.add(db_session)
    db.commit()

    return {"status": "saved", "session_uuid": db_session.session_uuid}


@app.get("/charger/session/fix-duration", tags=["Charger Session"])
def fix_zero_duration_sessions(
    db: Session = Depends(get_db),
    auth_user=Depends(get_user_by_api_key),
):
    # Find sessions with zero duration that belong to this user's chargers
    sessions_to_fix = (
        db.query(models.ChargingSession)
        .join(models.Charger)
        .filter(
            models.ChargingSession.duration_seconds == 0,
            models.Charger.owner_id == auth_user.id,
        )
        .all()
    )

    fixed_count = 0

    for session in sessions_to_fix:
        if session.start_time and session.end_time:
            duration = int((session.end_time - session.start_time).total_seconds())
            if duration > 0:
                session.duration_seconds = duration
                fixed_count += 1

    db.commit()

    return {"status": "completed", "fixed_sessions": fixed_count}


@app.delete("/charger/sessions/all", tags=["Charger Session"])
def delete_charger_sessions(
    charger_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    auth_user=Depends(get_user_by_api_key),
):

    # Query for chargers owned by the authenticated user
    charger_query = db.query(models.Charger).filter(
        models.Charger.owner_id == auth_user.id
    )

    if charger_id:
        charger_query = charger_query.filter(models.Charger.charger_id == charger_id)

    logger.info(f"Query: {charger_query}")

    charger_ids = [c.id for c in charger_query.all()]

    if not charger_ids:
        raise HTTPException(
            status_code=404, detail="No chargers found for user (or invalid charger_id)"
        )

    # Delete associated sessions
    deleted_count = (
        db.query(models.ChargingSession)
        .filter(models.ChargingSession.charger_id.in_(charger_ids))
        .delete(synchronize_session=False)
    )

    db.commit()

    return {
        "status": "deleted",
        "deleted_sessions": deleted_count,
        "charger_filter_applied": bool(charger_id),
    }


@app.delete("/charger/session/{session_uuid}", tags=["Charger Session"])
def delete_charger_session(
    session_uuid: str,
    db: Session = Depends(get_db),
    auth_user=Depends(get_user_by_api_key),
):
    session_obj = (
        db.query(models.ChargingSession)
        .join(models.Charger, models.ChargingSession.charger_id == models.Charger.id)
        .filter(
            models.ChargingSession.session_uuid == session_uuid,
            models.Charger.owner_id == auth_user.id,
        )
        .first()
    )
    if not session_obj:
        raise HTTPException(
            status_code=404, detail="Charging session not found or not owned by user"
        )

    db.delete(session_obj)
    db.commit()

    return {"status": "deleted"}


@app.get(
    "/charger/sessions",
    response_model=List[ChargingSessionOut],
    tags=["Charger Session"],
)
def charger_sessions(
    charger_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user=Depends(get_user_by_api_key),
):
    # Get all chargers owned by this user
    chargers = db.query(models.Charger).filter(models.Charger.owner_id == user.id).all()
    charger_ids = [c.id for c in chargers]  # real DB ids
    charger_map = {c.id: c.charger_id for c in chargers}  # id -> string map

    if charger_id:
        matching = [c for c in chargers if c.charger_id == charger_id]
        if not matching:
            raise HTTPException(
                status_code=404, detail="Charger not found for this user"
            )
        charger_ids = [matching[0].id]

    sessions = (
        db.query(models.ChargingSession)
        .filter(models.ChargingSession.charger_id.in_(charger_ids))
        .order_by(models.ChargingSession.end_time.desc())
        .all()
    )

    # Convert to schema with correct charger_id string
    result = []
    for s in sessions:
        result.append(
            ChargingSessionOut(
                session_uuid=s.session_uuid,
                charger_id=charger_map[s.charger_id],
                tag=s.tag,
                start_time=s.start_time,
                end_time=s.end_time,
                duration_seconds=s.duration_seconds,
                energy_charged_kwh=s.energy_charged_kwh,
            )
        )

    return result


@app.get("/me", response_model=UserInfoOut, tags=["User"])
def get_my_info(db: Session = Depends(get_db), auth_user=Depends(get_user_by_api_key)):
    return auth_user


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


@app.post("/me", response_model=UserInfoOut, tags=["User"])
def get_my_info_with_username_password(
    auth_data: UserAuthIn = Body(...), db: Session = Depends(get_db)
):
    user = (
        db.query(models.User).filter(models.User.username == auth_data.username).first()
    )
    if not user or not verify_password(auth_data.password, user.password):
        raise HTTPException(status_code=403, detail="Invalid username or password")
    return user


@app.get("/charger/sessions/export", tags=["Charger Sessions"])
def export_charger_sessions(
    format: str = "json",
    db: Session = Depends(get_db),
    auth_charger=Depends(get_charger_by_api_key),
):
    sessions = (
        db.query(models.ChargingSession)
        .filter_by(charger_id=auth_charger.charger_id)
        .all()
    )
    if format == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "session_id",
                "start_time",
                "end_time",
                "tag",
                "energy_kwh",
                "duration_sec",
            ]
        )
        for s in sessions:
            writer.writerow(
                [
                    s.session_id,
                    s.start_time,
                    s.end_time,
                    s.tag,
                    s.energy_charged_kwh,
                    s.duration_seconds,
                ]
            )
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=sessions.csv"},
        )
    elif format == "json":
        return [s.__dict__ for s in sessions]
    raise HTTPException(400, "Unsupported format")


# --- Dashboard UI & Auth with session cookies ---


@app.get("/dashboard/login", response_class=HTMLResponse, tags=["UI"])
def dashboard_login(
    request: Request,
    msg: Optional[str] = None,
    session: str = Cookie(default=None),
    db: Session = Depends(get_db),
):
    # Check if user is already authenticated
    if session:
        try:
            data = cookie_serializer.loads(session)
            user = db.query(models.User).filter_by(username=data["username"]).first()
            if user:
                # User is authenticated, redirect to dashboard
                return RedirectResponse(url="/dashboard", status_code=302)
        except Exception:
            pass  # Ignore invalid cookie/session errors

    # Show login page if not authenticated
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "msg": msg,
            "now": datetime.utcnow,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
        },
    )


@app.post("/dashboard/login", tags=["UI"])
def dashboard_login_post(
    form: UserAuthIn = Depends(UserAuthIn.as_form), db: Session = Depends(get_db)
):
    user = db.query(models.User).filter_by(username=form.username).first()
    if not user or not bcrypt.checkpw(form.password.encode(), user.password.encode()):
        return RedirectResponse(url="/dashboard/login?msg=invalid", status_code=303)

    response = RedirectResponse(url="/dashboard", status_code=303)
    create_auth_cookie(response, username=user.username)
    return response


@app.get("/dashboard/logout", tags=["UI"])
def dashboard_logout(response: Response):
    response = RedirectResponse(url="/dashboard/login", status_code=302)
    response.delete_cookie("session", path="/")
    return response


@app.get("/dashboard", response_class=HTMLResponse, tags=["UI"])
def dashboard(
    request: Request,
    user=Depends(get_user_from_cookie),
    db: Session = Depends(get_db),
    charger: Optional[str] = Query(None),  # allow charger to be None
):
    # Get all chargers owned by user
    chargers = db.query(models.Charger).filter_by(owner_id=user.id).all()
    charger_ids = {int(c.id) for c in chargers}

    # Auto-select if only one charger exists
    if len(charger_ids) == 1:
        charger = str(next(iter(charger_ids)))  # cast to string for consistency
    elif charger is None:
        charger = request.cookies.get("selected_charger", "all")

    # Validate
    if charger != "all" and int(charger) not in charger_ids:
        charger = "all"

    # Filter sessions
    if charger == "all":
        sessions = (
            db.query(models.ChargingSession)
            .join(models.Charger)
            .filter(models.Charger.owner_id == user.id)
            .order_by(models.ChargingSession.start_time.desc())
            .all()
        )
    else:
        sessions = (
            db.query(models.ChargingSession)
            .filter_by(charger_id=int(charger))
            .order_by(models.ChargingSession.start_time.desc())
            .all()
        )

    # Optionally, sort all sessions by start_time descending
    sessions.sort(key=lambda s: s.start_time, reverse=True)

    response = templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "chargers": chargers,
            "selected_charger": charger,
            "sessions": sessions,
            "now": datetime.utcnow,
            "apiKey": user.api_key,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
        },
    )

    # Store selected charger in cookie
    response.set_cookie(
        key="selected_charger",
        value=charger,
        max_age=3600 * 24 * 7,  # 7 days
        httponly=True,
    )
    return response


@app.get("/dashboard/sessions", response_model=dict, tags=["UI"])
def dashboard_charger_sessions(
    page: int = Query(1, ge=1),
    charger_id: Optional[int] = Query(None, description="Filter by charger ID"),
    start_date: Optional[str] = Query(
        None, description="Start date in YYYY-MM-DD format"
    ),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format"),
    user: models.User = Depends(get_user_by_api_key),
    db: Session = Depends(get_db),
):
    per_page = 10

    # Get user's chargers
    chargers = db.query(models.Charger).filter(models.Charger.owner_id == user.id).all()
    charger_ids = [c.id for c in chargers]
    charger_map = {c.id: c.charger_id for c in chargers}

    # Validate charger_id if provided
    if charger_id is not None:
        if charger_id not in charger_ids:
            raise HTTPException(
                status_code=403, detail="Charger does not belong to the user."
            )
        charger_ids = [charger_id]

    # Parse dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Use YYYY-MM-DD."
        )

    # Build query
    query = db.query(models.ChargingSession).filter(
        models.ChargingSession.charger_id.in_(charger_ids)
    )

    if start_dt:
        query = query.filter(models.ChargingSession.start_time >= start_dt)
    if end_dt:
        query = query.filter(models.ChargingSession.start_time <= end_dt)

    query = query.order_by(models.ChargingSession.end_time.desc())

    total = query.count()
    sessions = query.offset((page - 1) * per_page).limit(per_page).all()
    total_pages = ceil(total / per_page)

    session_list = [
        ChargingSessionOut(
            session_uuid=s.session_uuid,
            charger_id=charger_map[s.charger_id],
            tag=s.tag,
            start_time=make_aware(s.start_time),
            end_time=make_aware(s.end_time),
            cost=s.cost,
            duration_seconds=s.duration_seconds,
            energy_charged_kwh=s.energy_charged_kwh,
        )
        for s in sessions
    ]

    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "sessions": session_list,
    }


@app.get("/dashboard/monthly_sessions", tags=["UI"])
def dashboard_monthly_sessions(
    year: Optional[int] = Query(None, ge=2000, le=2100),
    charger_id: Optional[int] = Query(None, description="Filter by charger ID"),
    db: Session = Depends(get_db),
    user=Depends(get_user_from_cookie),
):
    if year is None:
        year = datetime.utcnow().year

    charger_alias = aliased(models.Charger)

    # Get available years filtered by user and optional charger
    all_years_query = (
        db.query(extract("year", models.ChargingSession.start_time).label("year"))
        .join(charger_alias, models.ChargingSession.charger_id == charger_alias.id)
        .filter(charger_alias.owner_id == user.id)
    )
    if charger_id is not None:
        all_years_query = all_years_query.filter(
            models.ChargingSession.charger_id == charger_id
        )

    year_rows = all_years_query.distinct().all()
    available_years = sorted(int(row.year) for row in year_rows if row.year is not None)

    total_sessions_year = 0
    total_energy_year = 0.0
    total_cost_year = 0.0

    monthly_data = []
    for month in range(1, 13):
        month_data_query = (
            db.query(
                func.count(models.ChargingSession.id).label("session_count"),
                func.sum(models.ChargingSession.energy_charged_kwh).label("total_kwh"),
                func.sum(models.ChargingSession.cost).label("total_cost"),
            )
            .join(charger_alias, models.ChargingSession.charger_id == charger_alias.id)
            .filter(
                extract("year", models.ChargingSession.start_time) == year,
                extract("month", models.ChargingSession.start_time) == month,
                charger_alias.owner_id == user.id,
            )
        )
        if charger_id is not None:
            month_data_query = month_data_query.filter(
                models.ChargingSession.charger_id == charger_id
            )

        month_data = month_data_query.one()

        session_count = month_data.session_count or 0
        total_kwh = month_data.total_kwh or 0.0
        total_cost = month_data.total_cost or 0.0

        monthly_data.append(
            {
                "sessions": session_count,
                "kwh": round(total_kwh, 2),
                "cost": round(total_cost, 2),
            }
        )

        total_sessions_year += session_count
        total_energy_year += total_kwh
        total_cost_year += total_cost

    year_summary = {
        "total_sessions": total_sessions_year,
        "total_energy_kwh": round(total_energy_year, 2),
        "total_cost": round(total_cost_year, 2),
    }

    today = datetime.utcnow().replace(day=1)
    last_12_months_summary = defaultdict(
        lambda: {"sessions": 0, "kwh": 0.0, "cost": 0.0}
    )

    for i in range(12):
        month_start = today - relativedelta(months=i)
        y = month_start.year
        m = month_start.month

        last_month_query = (
            db.query(
                func.count(models.ChargingSession.id).label("session_count"),
                func.sum(models.ChargingSession.energy_charged_kwh).label("total_kwh"),
                func.sum(models.ChargingSession.cost).label("total_cost"),
            )
            .join(charger_alias, models.ChargingSession.charger_id == charger_alias.id)
            .filter(
                extract("year", models.ChargingSession.start_time) == y,
                extract("month", models.ChargingSession.start_time) == m,
                charger_alias.owner_id == user.id,
            )
        )
        if charger_id is not None:
            last_month_query = last_month_query.filter(
                models.ChargingSession.charger_id == charger_id
            )

        last_month_data = last_month_query.one()

        key = f"{y}-{m:02d}"
        last_12_months_summary[key]["sessions"] = last_month_data.session_count or 0
        last_12_months_summary[key]["kwh"] = round(last_month_data.total_kwh or 0.0, 2)
        last_12_months_summary[key]["cost"] = round(
            last_month_data.total_cost or 0.0, 2
        )

    last_12_months_summary = dict(sorted(last_12_months_summary.items()))

    return {
        "year": year,
        "available_years": available_years,
        "monthly_data": monthly_data,
        "last_12_months_summary": last_12_months_summary,
        "year_summary": year_summary,
    }


@app.get("/dashboard/daily_sessions", tags=["UI"])
def dashboard_daily_sessions(
    month: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}$"),
    charger_id: Optional[int] = Query(None, description="Filter by charger ID"),
    db: Session = Depends(get_db),
    user=Depends(get_user_from_cookie),
):
    # Parse month or default to current month
    if month is None:
        today = datetime.utcnow()
        year = today.year
        month_num = today.month
    else:
        try:
            year, month_num = map(int, month.split("-"))
        except ValueError:
            return {"error": "Invalid month format. Use YYYY-MM"}

    # Alias Charger for join clarity
    charger_alias = aliased(models.Charger)

    # Query distinct available months for sessions belonging to user's chargers
    all_months_query = (
        db.query(
            extract("year", models.ChargingSession.start_time).label("y"),
            extract("month", models.ChargingSession.start_time).label("m"),
        )
        .join(charger_alias, models.ChargingSession.charger_id == charger_alias.id)
        .filter(charger_alias.owner_id == user.id)
    )

    if charger_id is not None:
        all_months_query = all_months_query.filter(
            models.ChargingSession.charger_id == charger_id
        )

    all_months = all_months_query.distinct().order_by("y", "m").all()
    available_months = [f"{int(y):04d}-{int(m):02d}" for y, m in all_months if y and m]

    # Base query to get total kWh per day for the requested month
    query = (
        db.query(
            func.date(models.ChargingSession.start_time).label("day"),
            func.sum(models.ChargingSession.energy_charged_kwh).label("total_kwh"),
        )
        .join(charger_alias, models.ChargingSession.charger_id == charger_alias.id)
        .filter(
            extract("year", models.ChargingSession.start_time) == year,
            extract("month", models.ChargingSession.start_time) == month_num,
            charger_alias.owner_id == user.id,
        )
    )

    if charger_id is not None:
        query = query.filter(models.ChargingSession.charger_id == charger_id)

    sessions = (
        query.group_by(func.date(models.ChargingSession.start_time))
        .order_by(func.date(models.ChargingSession.start_time))
        .all()
    )

    # Map day -> total kWh
    energy_by_day = {str(row.day): float(row.total_kwh or 0) for row in sessions}

    # Calculate days in month and fill missing days with 0
    first_day = datetime(year, month_num, 1)
    next_month = first_day + relativedelta(months=1)
    days_in_month = (next_month - relativedelta(days=1)).day

    daily_data = []
    for day in range(1, days_in_month + 1):
        date_str = f"{year}-{month_num:02d}-{day:02d}"
        kwh = energy_by_day.get(date_str, 0)
        daily_data.append({"day": day, "kwh": round(kwh, 2)})

    return {
        "month": f"{year}-{month_num:02d}",
        "available_months": available_months,
        "daily_data": daily_data,
    }


@app.get("/dashboard/statuses", tags=["UI"])
def get_all_charger_statuses(
    db: Session = Depends(get_db), user=Depends(get_user_from_cookie)
):
    chargers = db.query(models.Charger).filter_by(owner_id=user.id).all()
    results = {}

    for charger in chargers:
        latest = (
            db.query(models.ChargingStatus)
            .filter_by(charger_id=charger.id)
            .order_by(models.ChargingStatus.timestamp.desc())
            .first()
        )
        if latest:
            results[charger.charger_id] = {
                "status": latest.status,
                "energy": round(latest.energy, 2),
                "seconds": latest.seconds,
                "cost": latest.cost,
                "phase_count": latest.phase_count,
                "timestamp": latest.timestamp.isoformat(),
            }

    return results


@app.get("/dashboard/change-password", response_class=HTMLResponse, tags=["User"])
def show_change_password_form(
    request: Request,
    user: models.User = Depends(get_user_from_cookie),
):
    return templates.TemplateResponse(
        "change_password.html",
        {
            "request": request,
            "user": user,
            "now": datetime.utcnow,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
        },
    )


@app.post("/dashboard/change-password", response_class=HTMLResponse, tags=["User"])
def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_new_password: str = Form(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_user_from_cookie),
):
    # Validate current password
    if not bcrypt.checkpw(current_password.encode(), user.password.encode()):
        return templates.TemplateResponse(
            "change_password.html",
            {
                "request": request,
                "user": user,
                "error": "Current password is incorrect.",
                "now": datetime.utcnow,
                "appName": APP_NAME,
                "appInfo": APP_INFO,
            },
            status_code=400,
        )

    # Validate new password match
    if new_password != confirm_new_password:
        return templates.TemplateResponse(
            "change_password.html",
            {
                "request": request,
                "user": user,
                "error": "New passwords do not match.",
                "now": datetime.utcnow,
                "appName": APP_NAME,
                "appInfo": APP_INFO,
            },
            status_code=400,
        )

    # Hash and update new password
    user.password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    db.commit()

    return templates.TemplateResponse(
        "change_password.html",
        {
            "request": request,
            "user": user,
            "success": "Your password has been updated.",
            "now": datetime.utcnow,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
        },
    )


@app.get("/dashboard/profile", response_class=HTMLResponse, tags=["User"])
def show_profile(
    request: Request,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_user_from_cookie),
):
    return templates.TemplateResponse(
        "profile.html",
        {
            "request": request,
            "user": user,
            # add any other needed context
            "now": datetime.utcnow,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
        },
    )


@app.post("/dashboard/profile", response_class=HTMLResponse, tags=["User"])
def update_profile(
    request: Request,
    email: str = Form(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_user_from_cookie),
):
    if email == user.email:
        return templates.TemplateResponse(
            "profile.html",
            {
                "request": request,
                "user": user,
                "success": "No changes made.",
                "now": datetime.utcnow,
                "appName": APP_NAME,
                "appInfo": APP_INFO,
            },
        )

    if "@" not in email:
        return templates.TemplateResponse(
            "profile.html",
            {
                "request": request,
                "user": user,
                "error": "Invalid email address.",
                "now": datetime.utcnow,
                "appName": APP_NAME,
                "appInfo": APP_INFO,
            },
        )

    # Check if new email is already used
    if db.query(models.User).filter(models.User.email == email).first():
        return templates.TemplateResponse(
            "profile.html",
            {
                "request": request,
                "user": user,
                "error": "That email is already in use.",
                "now": datetime.utcnow,
                "appName": APP_NAME,
                "appInfo": APP_INFO,
            },
        )

    # Create a secure token
    token = secrets.token_urlsafe(32)

    # Store the pending email and token
    user.pending_email = email
    user.email_change_token = token
    db.commit()

    # Send confirmation email
    send_email(
        to=email,
        subject=f"{APP_NAME} Email Change Confirmation",
        template_name="base.html",
        context={
            "subject": f"Confirm Your New {APP_NAME} Email",
            "heading": "Confirm Email Change",
            "message": (
                f"You requested to change your email to <strong>{email}</strong>. "
                "Click the button below to confirm this change."
            ),
            "action_url": f"{BASE_URL}/dashboard/confirm-email-change/{token}",
            "action_text": "Confirm Email Change",
            "year": datetime.utcnow().year,
        },
    )

    return templates.TemplateResponse(
        "profile.html",
        {
            "request": request,
            "user": user,
            "success": f"A confirmation link was sent to {email}. Please check your inbox.",
            "now": datetime.utcnow,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
        },
    )


def send_email_confirmation(to: str, token: str, user_id: int, base_url: str):
    confirm_link = f"{base_url}confirm-email-change?token={token}&user_id={user_id}"
    subject = "Confirm your new email address"
    body = f"Click the link below to confirm your new email:\n\n{confirm_link}"

    # Use your preferred email-sending method here
    send_email(to=to, subject=subject, body=body)


@app.get("/dashboard/confirm-email-change/{token}", tags=["User"])
def confirm_email_change(
    token: str,
    request: Request,
    db: Session = Depends(get_db),
):
    if not TOKEN_PATTERN.match(token):
        raise HTTPException(status_code=400, detail="Invalid token format.")

    user = db.query(models.User).filter_by(email_change_token=token).first()
    if not user:
        raise HTTPException(
            status_code=400, detail="Invalid or expired confirmation link."
        )

    if not user.pending_email:
        raise HTTPException(status_code=400, detail="No pending email to confirm.")

    # Finalize email change
    user.email = user.pending_email
    user.pending_email = None
    user.email_change_token = None
    db.commit()

    return templates.TemplateResponse(
        "profile.html",
        {
            "request": request,
            "user": user,
            "success": "Email address successfully updated.",
            "now": datetime.utcnow,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
        },
    )


@app.get("/dashboard/import-nexxtmove-csv", response_class=HTMLResponse, tags=["UI"])
def show_csv_import_form(
    request: Request,
    db: Session = Depends(get_db),
    user=Depends(get_user_from_cookie),
):
    chargers = db.query(models.Charger).filter_by(owner_id=user.id).all()
    return templates.TemplateResponse(
        "import_nexxtmove_csv.html",
        {
            "request": request,
            "chargers": chargers,
            "now": datetime.utcnow,
            "apiKey": user.api_key,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
        },
    )


@app.get(
    "/dashboard/create-session", response_class=HTMLResponse, tags=["Charger Session"]
)
def show_create_session_form(
    request: Request,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_user_from_cookie),
):
    chargers = (
        db.query(models.Charger)
        .filter_by(owner_id=user.id)
        .order_by(models.Charger.charger_id.asc())
        .all()
    )
    return templates.TemplateResponse(
        "create_session.html",
        {
            "request": request,
            "chargers": chargers,
            "now": datetime.utcnow,
            "apiKey": user.api_key,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
        },
    )


@app.get("/dashboard/create-charger", response_class=HTMLResponse, tags=["UI"])
def show_create_charger_form(
    request: Request,
    user: models.User = Depends(get_user_from_cookie),
):
    return templates.TemplateResponse(
        "create_charger.html",
        {
            "request": request,
            "user": user,
            "apiKey": user.api_key,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
            "now": datetime.utcnow,
        },
    )


@app.get("/dashboard/charger/{charger_id}", response_class=HTMLResponse, tags=["UI"])
def show_manage_charger_form(
    charger_id: str,
    request: Request,
    user: models.User = Depends(get_user_from_cookie),
    db: Session = Depends(get_db),
):
    charger = (
        db.query(models.Charger)
        .filter_by(charger_id=charger_id, owner_id=user.id)
        .first()
    )
    if not charger:
        raise HTTPException(
            status_code=404, detail="Charger not found or not owned by user"
        )

    return templates.TemplateResponse(
        "manage_charger.html",
        {
            "request": request,
            "user": user,
            "apiKey": user.api_key,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
            "now": datetime.utcnow,
            "charger": charger,  # Pass charger model directly to template
        },
    )


@app.post("/import-nexxtmove-csv", tags=["Charger"])
async def import_csv(
    charger_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_user_by_api_key),
):
    # Check if the charger exists and is owned by the user
    charger = (
        db.query(models.Charger).filter_by(id=charger_id, owner_id=user.id).first()
    )
    if not charger:
        raise HTTPException(
            status_code=404,
            detail=f"Charger '{charger_id}' not found or not owned by this user",
        )

    # File format validation
    if not file.filename.endswith(".csv") or file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Detect encoding (optional, requires chardet)
    file.file.seek(0)
    raw_data = file.file.read(10000)
    file.file.seek(0)
    detected = chardet.detect(raw_data)
    encoding = detected["encoding"] or "utf-8"

    # Parse and insert sessions
    content = TextIOWrapper(file.file, encoding=encoding)
    reader = csv.DictReader(content, delimiter=";")

    created = 0
    for row in reader:
        try:
            session_data = convert_csv_row(row, charger_id)
            session = models.ChargingSession(**session_data)
            db.add(session)
            created += 1
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error in row: {e}")

    db.commit()
    return JSONResponse(
        content={
            "message": f"{created} session{'s' if created != 1 else ''} imported for charger '{charger.charger_id}'"
        }
    )


@app.get(
    "/reports/download",
    summary="Download charging session report as PDF",
    tags=["Reports"],
)
def download_report(
    username: str = Query(...),
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    tag: Optional[str] = Query(None, description="Partial tag name"),
    charger_ids: Optional[List[str]] = Query(
        None, description="List of charger_id strings (e.g., garage, home)"
    ),
    group_by_tag: bool = Query(
        False, description="Group sessions by tag instead of charger"
    ),
    db: Session = Depends(get_db),
):
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Use YYYY-MM-DD."
        )

    try:
        grouped = fetch_user_sessions(
            db=db,
            username=username,
            start_date=start_dt,
            end_date=end_dt,
            tag=tag,
            charger_ids=charger_ids,
            group_by_tag=group_by_tag,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not grouped:
        raise HTTPException(
            status_code=404, detail="No sessions found for this user and date range."
        )

    buffer = BytesIO()
    generate_pdf_report(username, grouped, start_dt, end_dt, buffer, group_by_tag)
    buffer.seek(0)

    filename = f"{APP_NAME}_charging_report_{start_date}_to_{end_date}"

    if group_by_tag:
        filename += "_grouped_by_tag"

    if tag:
        filename += f"_tag_{tag}"

    if charger_ids:
        filename += "_chargers_" + "_".join(charger_ids)

    filename = slugify(filename) + ".pdf"

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/dashboard")


# --- Exception handler to redirect 401 to login on dashboard paths ---


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 401 and request.url.path.startswith("/dashboard"):
        return RedirectResponse(url="/dashboard/login")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
