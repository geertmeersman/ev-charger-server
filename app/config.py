import os
import re
import sys
from pathlib import Path

VERSION_FILE = Path(__file__).parent.parent / "VERSION"

try:
    VERSION = VERSION_FILE.read_text().strip()
except FileNotFoundError:
    VERSION = "v0.0.0"  # default fallback version

SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@example.com")
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-change-me")
BASE_URL = os.getenv("BASE_URL")
APP_NAME = os.getenv("APP_NAME", "EV Charger Server")
REPORT_TITLE = os.getenv("REPORT_TITLE", "Charging Sessions Report")
APP_INFO = f"{APP_NAME} {VERSION}"
READONLY_MODE = os.getenv("READONLY", "true").lower() in ("true", "1", "yes")
HEX_COLOR_RE = re.compile(r"^([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")

# Validate required variables
missing_vars = []
if not SMTP_SERVER:
    missing_vars.append("SMTP_SERVER")
if not SMTP_USERNAME:
    missing_vars.append("SMTP_USERNAME")
if not SMTP_PASSWORD:
    missing_vars.append("SMTP_PASSWORD")
if not BASE_URL:
    missing_vars.append("BASE_URL")

if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)
