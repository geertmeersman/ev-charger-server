import re
from datetime import datetime


def parse_csv_duration(duration_str: str) -> int:
    duration_str = re.sub(r"[^0-9h]", "", duration_str.lower())
    hours = 0
    minutes = 0
    if "h" in duration_str:
        parts = duration_str.split("h")
        if parts[0]:
            hours = int(parts[0])
        if len(parts) > 1 and parts[1]:
            minutes = int(parts[1])
    return hours * 3600 + minutes * 60


def convert_csv_row(row: dict, charger_id: str) -> dict:
    start_time = datetime.strptime(row["Session started"], "%d-%m-%Y %H:%M:%S")
    end_time = datetime.strptime(row["Session ended"], "%d-%m-%Y %H:%M:%S")
    tag = str(row["Tag"])
    duration_seconds = int((end_time - start_time).total_seconds())
    energy_charged_kwh = float(row["Consumption"].replace(",", "."))

    return {
        "charger_id": charger_id,
        "tag": tag,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_seconds,
        "energy_charged_kwh": energy_charged_kwh,
    }