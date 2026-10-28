"""
Crop Pulse -- farmer dashboard (Flask)
---------------------------------------
Reads your ThingSpeak channel(s), shows live soil + livestock readings,
the ML irrigation recommendation, and plain-language advice for the farmer.

Run:
  pip install -r requirements.txt
  python app.py
  open http://127.0.0.1:5000   (on your phone: http://<laptop-ip>:5000, same WiFi)

ts.env (same folder as app.py, no quotes):
  THINGSPEAK_CHANNEL_ID=3508766
  THINGSPEAK_READ_API_KEY=your_read_key
  # Optional: only if livestock data lives in a SEPARATE channel
  # (field1 = body temp, field2 = activity index in that channel)
  LIVESTOCK_CHANNEL_ID=
  LIVESTOCK_READ_API_KEY=
"""

import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, send_from_directory

load_dotenv("ts.env")

CHANNEL_ID = os.getenv("THINGSPEAK_CHANNEL_ID")
READ_KEY = os.getenv("THINGSPEAK_READ_API_KEY")
LIVESTOCK_CHANNEL_ID = os.getenv("LIVESTOCK_CHANNEL_ID") or None
LIVESTOCK_READ_KEY = os.getenv("LIVESTOCK_READ_API_KEY") or None

# ---- Thresholds: tune these to your crop / animal / sensor ----------------
MOISTURE_DRY = 30        # below -> irrigate
MOISTURE_OK_LOW = 40     # 30-40 -> watch
MOISTURE_WET = 70        # above -> waterlogging risk
PH_LOW, PH_HIGH = 6.0, 7.5
SOIL_TEMP_HOT, SOIL_TEMP_COLD = 35.0, 10.0
# MLX90614 reads SKIN/surface temp, which is lower than core body temp.
# Calibrate these against a normal animal's reading before trusting them.
ANIMAL_TEMP_HIGH, ANIMAL_TEMP_LOW = 39.5, 30.0
ACTIVITY_LOW = 2.0       # activity index below this -> unusually still
OFFLINE_AFTER_MIN = 5    # no new data for this long -> device offline
# ----------------------------------------------------------------------------

app = Flask(__name__)


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch(channel, key, path, **params):
    url = f"https://api.thingspeak.com/channels/{channel}/{path}"
    if key:
        params["api_key"] = key
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def minutes_ago(ts):
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - t).total_seconds() / 60
    except Exception:
        return None


def build_advice(d):
    """Rule-based advice from latest readings + ML output. Most urgent first."""
    tips = []

    def add(level, area, title, detail):
        tips.append({"level": level, "area": area, "title": title, "detail": detail})

    m, st, ph = d["moisture"], d["soil_temp"], d["ph"]
    lt, act = d["livestock_temp"], d["activity_index"]

    # --- Irrigation (sensor rule + ML) ---
    if m is not None:
        if m < MOISTURE_DRY:
            add("urgent", "Crop", "Soil is dry - irrigate now",
                f"Moisture is {m:.0f}% (below {MOISTURE_DRY}%). Water early morning or "
                "evening to cut evaporation loss.")
        elif m < MOISTURE_OK_LOW:
            add("watch", "Crop", "Moisture getting low",
                f"{m:.0f}% - not critical yet. Plan irrigation within the day, "
                "especially if it is hot.")
        elif m > MOISTURE_WET:
            add("watch", "Crop", "Soil may be waterlogged",
                f"{m:.0f}% is very wet. Stop irrigating and check drainage - roots "
                "can rot in standing water.")
        else:
            add("good", "Crop", "Soil moisture is healthy", f"{m:.0f}% - no watering needed.")
    if d["recommendation"] == "IRRIGATE" and (m is None or m >= MOISTURE_DRY):
        add("watch", "Crop", "ML model suggests irrigating",
            "The model also weighs temperature, pH and time since last watering, "
            "so it can flag dryness before the moisture alone looks low.")

    # --- Soil pH ---
    if ph is not None:
        if ph < 5.5:
            add("urgent", "Soil", "Soil is too acidic",
                f"pH {ph:.1f}. Add agricultural lime or wood ash; recheck in 2-3 weeks.")
        elif ph < PH_LOW:
            add("watch", "Soil", "Slightly acidic soil",
                f"pH {ph:.1f}. Light lime or compost helps most crops absorb nutrients.")
        elif ph > 8.0:
            add("urgent", "Soil", "Soil is too alkaline",
                f"pH {ph:.1f}. Add gypsum or elemental sulfur and organic matter.")
        elif ph > PH_HIGH:
            add("watch", "Soil", "Slightly alkaline soil",
                f"pH {ph:.1f}. Add compost/organic matter to bring it down gradually.")
        else:
            add("good", "Soil", "pH is in the ideal range", f"pH {ph:.1f} suits most crops.")

    # --- Soil temperature ---
    if st is not None and st != 0:
        if st > SOIL_TEMP_HOT:
            add("watch", "Soil", "Soil is very hot",
                f"{st:.1f} C. Mulch the surface and irrigate in the evening to protect roots.")
        elif st < SOIL_TEMP_COLD:
            add("watch", "Soil", "Soil is cold",
                f"{st:.1f} C. Seeds germinate poorly - delay sowing if possible.")

    # --- Livestock ---
    if lt is not None:
        if lt > ANIMAL_TEMP_HIGH:
            add("urgent", "Livestock", "Possible fever / heat stress",
                f"Reading {lt:.1f} C. Move the animal to shade, give cool water, and "
                "call a vet if it stays high or the animal is off its feed.")
        elif lt < ANIMAL_TEMP_LOW:
            add("watch", "Livestock", "Body-temp reading unusually low",
                f"{lt:.1f} C. Check the sensor is aimed at the animal (not air or ground) "
                "before drawing conclusions.")
        else:
            add("good", "Livestock", "Body temperature normal", f"{lt:.1f} C.")
    if act is not None:
        if act < ACTIVITY_LOW and lt is not None and lt > ANIMAL_TEMP_HIGH - 1:
            add("urgent", "Livestock", "Warm and unusually still",
                "High temperature combined with low movement can signal illness. "
                "Inspect the animal today.")
        elif act < ACTIVITY_LOW:
            add("watch", "Livestock", "Low activity",
                "Animal is very still. Normal if resting - check again in an hour.")

    order = {"urgent": 0, "watch": 1, "good": 2}
    return sorted(tips, key=lambda t: order[t["level"]])


def latest():
    # The ESP32 writes fields 1-6 and live_inference.py writes fields 7-8 as
    # SEPARATE entries, so the very last entry is often missing half the
    # fields. Take the newest non-empty value of each field instead.
    feeds = fetch(CHANNEL_ID, READ_KEY, "feeds.json", results=50).get("feeds", [])

    def newest(field):
        for f in reversed(feeds):
            v = to_float(f.get(field))
            if v is not None:
                return v, f.get("created_at"), f.get("entry_id")
        return None, None, None

    moisture, m_time, m_id = newest("field1")
    soil_temp, _, _ = newest("field2")
    ph, _, _ = newest("field3")
    pump_on, _, _ = newest("field4")
    animal_t, _, _ = newest("field5")
    activity, _, _ = newest("field6")
    rec, _, _ = newest("field7")

    out = {
        "moisture": moisture,
        "soil_temp": soil_temp,
        "ph": ph,
        "pump_on": pump_on,
        "livestock_temp": animal_t,
        "activity_index": activity,
        "recommendation": "IRRIGATE" if rec == 1 else "HOLD" if rec == 0 else None,
        # "device online" is judged by when SENSOR data last arrived
        "created_at": m_time or (feeds[-1].get("created_at") if feeds else None),
        "entry_id": m_id,
    }
    # Optional separate livestock channel overrides field5/6
    if LIVESTOCK_CHANNEL_ID:
        try:
            L = fetch(LIVESTOCK_CHANNEL_ID, LIVESTOCK_READ_KEY, "feeds/last.json")
            out["livestock_temp"] = to_float(L.get("field1"))
            out["activity_index"] = to_float(L.get("field2"))
        except Exception:
            pass
    age = minutes_ago(out["created_at"])
    out["offline"] = age is None or age > OFFLINE_AFTER_MIN
    out["age_min"] = None if age is None else round(age, 1)
    out["advice"] = build_advice(out)
    return out


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/sw.js")
def service_worker():
    # Must be served from the site root so it can control the whole app
    resp = send_from_directory(app.static_folder, "sw.js", mimetype="application/javascript")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/healthz")
def healthz():
    return "ok"


@app.route("/api/data")
def api_data():
    try:
        return jsonify(latest())
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/history")
def api_history():
    try:
        d = fetch(CHANNEL_ID, READ_KEY, "feeds.json", results=60)
        feeds = d.get("feeds", [])
        return jsonify({
            "t": [f["created_at"] for f in feeds],
            "moisture": [to_float(f.get("field1")) for f in feeds],
            "ph": [to_float(f.get("field3")) for f in feeds],
            "livestock_temp": [to_float(f.get("field5")) for f in feeds],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    # host 0.0.0.0 lets a phone on the same WiFi open the dashboard
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
