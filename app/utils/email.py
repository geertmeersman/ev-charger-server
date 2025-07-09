import smtplib
from datetime import datetime
from email.message import EmailMessage
import time
from jinja2 import Environment, FileSystemLoader

from ..config import (
    APP_INFO,
    APP_NAME,
    FROM_EMAIL,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_SERVER,
    SMTP_USERNAME,
)

# Jinja2 setup
env = Environment(loader=FileSystemLoader("app/templates/email"), autoescape=True)


def send_email(to: str, subject: str, template_name: str, context: dict):
    template = env.get_template(template_name)
    context.update(
        {
            "now": datetime.utcnow,
            "appName": APP_NAME,
            "appInfo": APP_INFO,
        }
    )
    html_content = template.render(**context)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = to
    msg.set_content(
        "This is an HTML email. Please view it in an HTML-compatible client."
    )
    msg.add_alternative(html_content, subtype="html")

    try:
        start = time.time()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            print("connect:", time.time() - start)

            smtp.ehlo()
            print("ehlo:", time.time() - start)

            smtp.starttls()
            print("starttls:", time.time() - start)

            smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            print("login:", time.time() - start)

            smtp.send_message(msg)
            print("sent:", time.time() - start)
    except Exception as e:
        print(f"Email send error: {e}")  # Replace with logging in production
