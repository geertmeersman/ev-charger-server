# ⚡ EV Charger Server

**EV Charger Server** is a full-featured, API-driven and web-based system for managing electric vehicle (EV) charging sessions, stations, and users. It supports both REST APIs and a modern dashboard for users, along with secure device integration via API keys.

---

[![maintainer](https://img.shields.io/badge/maintainer-Geert%20Meersman-green?style=for-the-badge&logo=github)](https://github.com/geertmeersman)
[![buyme_coffee](https://img.shields.io/badge/Buy%20me%20an%20Omer-donate-yellow?style=for-the-badge&logo=buymeacoffee)](https://www.buymeacoffee.com/geertmeersman)
[![MIT License](https://img.shields.io/github/license/geertmeersman/ev-charger-server?style=for-the-badge)](https://github.com/geertmeersman/ev-charger-server/blob/main/LICENSE)

[![GitHub issues](https://img.shields.io/github/issues/geertmeersman/ev-charger-server)](https://github.com/geertmeersman/ev-charger-server/issues)
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/geertmeersman/ev-charger-server.svg)](http://isitmaintained.com/project/geertmeersman/ev-charger-server)
[![Percentage of issues still open](http://isitmaintained.com/badge/open/geertmeersman/ev-charger-server.svg)](http://isitmaintained.com/project/geertmeersman/ev-charger-server)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](https://github.com/geertmeersman/ev-charger-server/pulls)

[![github release](https://img.shields.io/github/v/release/geertmeersman/ev-charger-server?logo=github)](https://github.com/geertmeersman/ev-charger-server/releases)
[![github release date](https://img.shields.io/github/release-date/geertmeersman/ev-charger-server)](https://github.com/geertmeersman/ev-charger-server/releases)
[![github last-commit](https://img.shields.io/github/last-commit/geertmeersman/ev-charger-server)](https://github.com/geertmeersman/ev-charger-server/commits)
[![github contributors](https://img.shields.io/github/contributors/geertmeersman/ev-charger-server)](https://github.com/geertmeersman/ev-charger-server/graphs/contributors)
[![github commit activity](https://img.shields.io/github/commit-activity/y/geertmeersman/ev-charger-server?logo=github)](https://github.com/geertmeersman/ev-charger-server/commits/main)

![Docker Pulls](https://img.shields.io/docker/pulls/geertmeersman/ev-charger-server)
![Docker Image Version](https://img.shields.io/docker/v/geertmeersman/ev-charger-server?label=docker%20image%20version)

---

<!-- TOC -->

- [⚡ EV Charger Server](#-ev-charger-server)
    - [🚀 Features](#-features)
    - [🛠️ Tech Stack](#-tech-stack)
    - [🐳 Docker Support](#-docker-support)
    - [⚙️ Environment Variables](#-environment-variables)
    - [📂 Project Structure](#-project-structure)
    - [📘 API Documentation](#-api-documentation)
    - [🔐 Security Note](#-security-note)
    - [📄 License](#-license)

<!-- /TOC -->

---

## 🚀 Features

- **🔐 User Management**
  - Register/login with secure cookie-based sessions
  - Password reset and username recovery via email
  - Generate and manage API keys
  - Change email and password securely

- **🔌 Charger Management**
  - Register chargers with descriptions and geo-coordinates
  - Track cost per kWh
  - Edit or remove chargers and their data

- **⚡ Charging Sessions**
  - Log charging sessions via minimal or detailed APIs
  - Export sessions as CSV or JSON
  - Fix or remove invalid sessions
  - PDF reports filtered by date, tag, or charger

- **📡 Device Integration**
  - Log charging events and status via API
  - Automatically create chargers from incoming events
  - Prevent duplicate status/events

- **📊 Dashboard**
  - Web UI to monitor sessions, charger status, usage trends
  - Monthly/daily energy graphs
  - Import session data from CSV (e.g. Nexxtmove)
  - Account and profile management
  - Readonly mode to show the charger information on a display

---

## 🛠️ Tech Stack

- **FastAPI** – async Python web framework
- **SQLAlchemy** – ORM for relational DBs
- **Jinja2** – for web dashboard templates
- **Uvicorn** – high-performance ASGI server
- **Passlib (bcrypt)** – secure password storage
- **ItsDangerous** – token-based session/email handling
- **SQLite/PostgreSQL** – database support

---

## 🐳 Docker Support

**Docker compose example:**

```yaml
version: '3.8'

services:
  ev-charger-server:
    image: geertmeersman/ev-charger-server:latest
    container_name: ev-charger-server
    restart: unless-stopped
    environment:
      SMTP_SERVER: ${SMTP_SERVER}
      SMTP_PORT: ${SMTP_PORT}
      SMTP_USERNAME: ${SMTP_USERNAME}
      SMTP_PASSWORD: ${SMTP_PASSWORD}
      FROM_EMAIL: ${FROM_EMAIL}
      SECRET_KEY: ${SECRET_KEY}
      BASE_URL: ${BASE_URL}
    tty: true
    ports:
      - "8000:8000"
    volumes:
      - /volumes/ev-charger-server:/app/data
```

---

**Dockerfile example:**

```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV TZ=Europe/Brussels

COPY scripts/start.sh /start.sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
```

**Build and run:**

```bash
docker build -t ev-charger-server .
docker run -e WEB_PORT=8080 -p 8080:8080 \
  -e SMTP_SERVER=smtp.example.com \
  -e SMTP_PORT=587 \
  -e SMTP_USERNAME=myuser \
  -e SMTP_PASSWORD=mypassword \
  -e FROM_EMAIL=noreply@example.com \
  -e SECRET_KEY=mysecretkey \
  -e BASE_URL=https://myapp.com \
  ev-charger-server
```

---

## ⚙️ Environment Variables

| Variable         | Description                                |
|------------------|--------------------------------------------|
| `SMTP_SERVER`    | SMTP host for sending emails               |
| `SMTP_PORT`      | SMTP port (e.g. `587` for TLS)             |
| `SMTP_USERNAME`  | Username for SMTP authentication           |
| `SMTP_PASSWORD`  | Password for SMTP authentication           |
| `FROM_EMAIL`     | Default "from" address for all emails      |
| `SECRET_KEY`     | Cryptographic key for session/token signing|
| `BASE_URL`       | Public base URL (used in email links, UI)  |
| `WEB_PORT`       | (Optional) HTTP port (default: `8000`)     |
| `READ_ONLY`      | (Optional) Readonly mode, to enable readonly dahsboard views |

---

## 📂 Project Structure

```bashPPP
app/
├── main.py               # FastAPI app entry point
├── database.py           # DB engine and session factory
├── models.py             # SQLAlchemy data models
├── templates/            # Jinja2 templates for the dashboard
├── static/               # Static files (CSS, JS)
├── reports/              # PDF report generation logic
├── utils/                # Email sending, CSV parsing, helpers
└── ...
```

---

## 📘 API Documentation

Once running, the following documentation is available:

- **Swagger (OpenAPI)**: [`/docs`](http://localhost:8000/docs)
- **ReDoc**: [`/redoc`](http://localhost:8000/redoc)

---

## 🔐 Security Note

⚠️ **This server runs in plain HTTP (not HTTPS) by default.** It is strongly recommended to place a reverse proxy (such as **NGINX**, **Caddy**, or **Traefik**) in front of this application to:

- Terminate TLS (serve HTTPS)
- Manage domains and ports
- Enable logging, rate limiting, and request filtering

You can also run this behind a secure load balancer or a platform (e.g. Fly.io, Render, Heroku) that provides HTTPS.

---

## 📄 License

This project is licensed under the MIT License.

---
