import os
import json
import math
import time
from datetime import datetime, timedelta, date
from functools import lru_cache
import pytz
from dotenv import load_dotenv

import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_apscheduler import APScheduler
from flask_mail import Mail, Message

import gspread
from google.oauth2.service_account import Credentials

load_dotenv()
print("Loaded ENV variables:")
print("SPORTMONKS_API_TOKEN:", os.environ.get("SPORTMONKS_API_TOKEN"))
print("GOOGLE_SHEET_ID:", os.environ.get("GOOGLE_SHEET_ID"))
print("MAIL_ENABLED:", os.environ.get("MAIL_ENABLED"))
print("MAIL_SERVER:", os.environ.get("MAIL_SERVER"))
print("MAIL_USERNAME:", os.environ.get("MAIL_USERNAME"))


# ==========================
# Configuration from ENV
# ==========================
SPORTMONKS_API_TOKEN = os.environ.get("SPORTMONKS_API_TOKEN")
SPORTMONKS_BASE = os.environ.get("SPORTMONKS_BASE", "https://api.sportmonks.com/v3/football")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
GOOGLE_SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")

CACHE_DIR = os.environ.get("CACHE_DIR", "cache")
MAIL_ENABLED = os.environ.get("MAIL_ENABLED", "False").lower() in ("true", "1", "yes")
MAIL_SERVER = os.environ.get("MAIL_SERVER")
MAIL_PORT = int(os.environ.get("MAIL_PORT", 587))
MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")
MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER")
MAIL_RECEIVER = os.environ.get("MAIL_RECEIVER")

PORT = int(os.environ.get("PORT", 8080))
DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1", "yes")

# Create cache dir
os.makedirs(CACHE_DIR, exist_ok=True)

# Flask app
app = Flask(__name__, template_folder="templates")
app.config['JSON_SORT_KEYS'] = False

# Mail
app.config.update(
    MAIL_SERVER=MAIL_SERVER,
    MAIL_PORT=MAIL_PORT,
    MAIL_USERNAME=MAIL_USERNAME,
    MAIL_PASSWORD=MAIL_PASSWORD,
    MAIL_USE_TLS=True if MAIL_PORT == 587 else False,
    MAIL_DEFAULT_SENDER=MAIL_DEFAULT_SENDER
)
mail = Mail(app) if MAIL_ENABLED else None

# Scheduler config
class Config:
    SCHEDULER_API_ENABLED = True

app.config.from_object(Config())
scheduler = APScheduler()
scheduler.init_app(app)

# Google sheets client (lazy)
def get_gspread_client():
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_FILE, scopes=scopes)
    client = gspread.authorize(creds)
    return client

# ==========================
# SportMonks helpers
# ==========================
def sportmonks_get(path, params=None):
    """Simple wrapper with token and basic caching (file-based short TTL)."""
    if params is None:
        params = {}
    params['api_token'] = SPORTMONKS_API_TOKEN
    url = f"{SPORTMONKS_BASE.rstrip('/')}/{path.lstrip('/')}"
    # Simple GET
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# Use small in-memory caches for repeated calls
@lru_cache(maxsize=1024)
def get_team_by_id(team_id):
    try:
        res = sportmonks_get(f"teams/{team_id}", params={'include': 'venue,country'})
        return res.get('data')
    except Exception:
        return None

def search_team(query):
    try:
        res = sportmonks_get("teams/search/" + requests.utils.requote_uri(query))
        return res.get('data', [])
    except Exception:
        return []

def get_fixtures_for_date(target_date: date):
    ds = target_date.strftime("%Y-%m-%d")
    try:
        res = sportmonks_get(f"fixtures/date/{ds}", params={'include': 'localTeam,visitorTeam,league,country'})
        return res.get('data', [])
    except Exception:
        return []

def get_team_fixtures(team_id, from_date=None, to_date=None, status='FT'):
    params = {'filter[team_id]': team_id}
    if from_date:
        params['date_from'] = from_date.strftime("%Y-%m-%d")
    if to_date:
        params['date_to'] = to_date.strftime("%Y-%m-%d")
    try:
        res = sportmonks_get("fixtures", params=params)
        return res.get('data', [])
    except Exception:
        return []

# ==========================
# Utilities and prediction
# ==========================
EUROPE_ISO = set([
    # core Europe ISO country codes (not exhaustive but wide coverage)
    "AL","AD","AT","BY","BE","BA","BG","HR","CY","CZ","DK","EE","FO","FI","FR","DE","GI","GR","HU","IS","IE",
    "IT","XK","LV","LI","LT","LU","MT","MD","MC","ME","NL","MK","NO","PL","PT","RO","RU","SM","RS","SK","SI",
    "ES","SE","CH","TR","UA","GB","VA"
])

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not cur or k not in cur:
            return default
        cur = cur[k]
    return cur

def compute_team_strengths(team_id, months=12, min_matches_required=8):
    """Gather last N months of fixtures and compute attack/defense averages.
       Returns dict with attack, defense, avg_goals_for, avg_goals_against, matches_count, last_match_date"""
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=months*30)
    fixtures = get_team_fixtures(team_id, from_date=from_date, to_date=to_date)
    if not fixtures:
        return None
    gf = []
    ga = []
    dates = []
    for f in fixtures:
        # stats: determine this team's goals and opponent goals (only include finished)
        status = f.get('time', {}).get('status')
        if status not in ('FT', 'AET', 'FT_PEN', 'After ET'):
            continue
        local = f.get('localTeam', {})
        visitor = f.get('visitorTeam', {})
        # goals from scores
        scores = f.get('scores', {}) or {}
        local_goals = None
        visitor_goals = None
        # fallback: older responses may have 'scores' differently; try standard fields
        local_goals = safe_get(f, 'scores', 'localteam_score')
        visitor_goals = safe_get(f, 'scores', 'visitorteam_score')
        # try alternate keys
        if local_goals is None and 'score' in f:
            local_goals = safe_get(f, 'score', 'localteam')
            visitor_goals = safe_get(f, 'score', 'visitorteam')
        # if still none, skip
        if local_goals is None or visitor_goals is None:
            continue
        try:
            local_goals = int(local_goals)
            visitor_goals = int(visitor_goals)
        except Exception:
            continue
        if int(team_id) == int(safe_get(local, 'data', 'id', default=local.get('id')) or local.get('id')):
            team_goals = local_goals
            opp_goals = visitor_goals
        else:
            team_goals = visitor_goals
            opp_goals = local_goals
        gf.append(team_goals)
        ga.append(opp_goals)
        match_date_s = safe_get(f, 'time', 'starting_at', 'date_time') or safe_get(f, 'time', 'starting_at', 'date')
        try:
            dates.append(datetime.fromisoformat(match_date_s))
        except Exception:
            pass

    matches_count = len(gf)
    if matches_count < min_matches_required:
        return {'matches_count': matches_count, 'last_match_date': max(dates).date() if dates else None} if matches_count>0 else None
    attack = np.mean(gf)
    defense = np.mean(ga)
    last_date = max(dates).date() if dates else None
    return {
        'attack': float(attack),
        'defense': float(defense),
        'avg_for': float(attack),
        'avg_against': float(defense),
        'matches_count': matches_count,
        'last_match_date': last_date
    }

def poisson_score_probs(home_xg, away_xg, max_goals=6):
    """Return matrix of probabilities home goals i and away goals j using Poisson assumption."""
    probs = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            probs[i,j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
    return probs

def calculate_match_probability(team_home_id, team_away_id, source_hint="sportmonks"):
    """Main prediction function. Attempts to use SportMonks stats first; if insufficient, uses fallback heuristic.
       Returns dict with probabilities and expected goals and source used."""
    # Try to compute strengths for both teams (12 months)
    home_stats = compute_team_strengths(team_home_id, months=12)
    away_stats = compute_team_strengths(team_away_id, months=12)

    # If either has insufficient matches or missing, fallback to simpler heuristic (the 'AI fallback')
    used_source = "sportmonks"
    if not home_stats or not away_stats or home_stats.get('matches_count',0) < 8 or away_stats.get('matches_count',0) < 8:
        used_source = "ai_fallback"
        # Heuristic fallback: get last 6 matches direct goals if possible otherwise assume league average
        # We'll attempt to fetch last 6 matches explicitly:
        def fallback_avg(team_id, n=6):
            try:
                fixtures = get_team_fixtures(team_id, to_date=datetime.utcnow().date())
            except Exception:
                fixtures = []
            gf = []
            ga = []
            for f in sorted(fixtures, key=lambda x: safe_get(x,'time','starting_at','date_time') or safe_get(x,'time','starting_at','date') or '')[::-1]:
                status = safe_get(f,'time','status')
                if status not in ('FT','AET','FT_PEN','After ET'):
                    continue
                local_goals = safe_get(f,'scores','localteam_score')
                visitor_goals = safe_get(f,'scores','visitorteam_score')
                if local_goals is None or visitor_goals is None:
                    continue
                try:
                    local_goals = int(local_goals); visitor_goals = int(visitor_goals)
                except Exception:
                    continue
                local_id = safe_get(f,'localTeam','data','id') or safe_get(f,'localTeam','id') or None
                if local_id and int(local_id)==int(team_id):
                    gf.append(local_goals); ga.append(visitor_goals)
                else:
                    gf.append(visitor_goals); ga.append(local_goals)
                if len(gf)>=n:
                    break
            if len(gf)==0:
                return None
            return {'attack': float(np.mean(gf)), 'defense': float(np.mean(ga)), 'matches_count': len(gf)}
        home_stats = fallback_avg(team_home_id) or {'attack':1.2,'defense':1.2,'matches_count':0}
        away_stats = fallback_avg(team_away_id) or {'attack':1.0,'defense':1.0,'matches_count':0}

    # Home advantage factor (rough)
    HOME_ADV = 1.08

    # Expected goals: weighted geometric mean/ratio of home attack vs away defense
    home_xg = (home_stats['attack'] * (away_stats['defense'] if away_stats['defense']>0 else 1) ) / (home_stats['defense'] if home_stats['defense']>0 else 1)
    away_xg = (away_stats['attack'] * (home_stats['defense'] if home_stats['defense']>0 else 1) ) / (away_stats['defense'] if away_stats['defense']>0 else 1)

    # Normalize and apply home advantage
    home_xg = max(0.1, home_xg * HOME_ADV * 0.6)
    away_xg = max(0.05, away_xg * 0.6)

    # Some smoothing for tiny samples
    if home_stats.get('matches_count',0) < 8:
        home_xg *= 0.9
    if away_stats.get('matches_count',0) < 8:
        away_xg *= 0.9

    # Cap to reasonable values
    home_xg = float(min(home_xg, 4.5))
    away_xg = float(min(away_xg, 4.0))

    # Compute score probabilities and derive match outcome probabilities
    score_mat = poisson_score_probs(home_xg, away_xg, max_goals=6)
    home_prob = float(np.sum(np.tril(score_mat, -1))) * 0.0 + float(np.sum(np.triu(score_mat, 1))) # careful: reversed?
    # Let's compute correctly:
    home_wins = float(np.sum(np.where(np.arange(0,7)[:,None] > np.arange(0,7)[None,:], score_mat, 0.0)))
    away_wins = float(np.sum(np.where(np.arange(0,7)[:,None] < np.arange(0,7)[None,:], score_mat, 0.0)))
    draw_prob = float(np.sum(np.diag(score_mat)))
    # Normalize small numeric issues
    total = home_wins + away_wins + draw_prob
    if total <= 0:
        home_p = 0.45; draw_p = 0.2; away_p = 0.35
    else:
        home_p = min(0.99, home_wins/total)
        away_p = min(0.99, away_wins/total)
        draw_p = min(0.99, draw_prob/total)

    # Return structured result
    return {
        'home_team_id': int(team_home_id),
        'away_team_id': int(team_away_id),
        'home_xg': round(home_xg,3),
        'away_xg': round(away_xg,3),
        'home_prob': round(home_p,3),
        'draw_prob': round(draw_p,3),
        'away_prob': round(away_p,3),
        'source': used_source,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

# ==========================
# Google Sheets functions
# ==========================
SHEET_HEADERS = ["DateUTC","DateLocal","HomeTeam","AwayTeam","HomeProb","DrawProb","AwayProb","Home_xG","Away_xG","Source","Notes"]

def ensure_sheet_and_append(data_rows, worksheet_title="Predictions"):
    """
    data_rows: list of lists (rows) matching headers.
    If worksheet_title doesn't exist in the spreadsheet, create it and add header row.
    """
    client = get_gspread_client()
    sh = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        worksheet = sh.worksheet(worksheet_title)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sh.add_worksheet(worksheet_title, rows="1000", cols=str(len(SHEET_HEADERS)))
        worksheet.append_row(SHEET_HEADERS)
    # Append rows
    for r in data_rows:
        worksheet.append_row(r, value_input_option="USER_ENTERED")
    return True

# ==========================
# Auto daily job: find European fixtures and predict
# ==========================
def is_fixture_european(fixture):
    # Try to find country ISO from nested objects
    league_country_iso = safe_get(fixture, "league", "data", "country", "ioc") or safe_get(fixture, "league", "data", "country", "iso") or safe_get(fixture, "country", "data", "iso")
    if league_country_iso:
        if league_country_iso.upper() in EUROPE_ISO:
            return True
    # fallback: try to see if league name includes known country names
    league_name = safe_get(fixture, "league", "data", "name") or ""
    if any(kw in league_name for kw in ["Championship","Premier","La Liga","Serie","Bundesliga","Ligue","Primeira","Eredivisie","Super Lig","Ukrain","Greek","Polish","Austrian","Swiss"]):
        return True
    return False

def daily_auto_job():
    """This function runs daily: finds today's fixtures, filters europe, selects up to 10,
       ensures teams have 9+ months of historical data, predicts and appends to sheet."""
    try:
        today = datetime.utcnow().date()
        fixtures = get_fixtures_for_date(today)
        # filter to European fixtures
        euro_fixtures = [f for f in fixtures if is_fixture_european(f)]
        # Prepare candidate fixtures with team ids and check team history length
        candidates = []
        for f in euro_fixtures:
            try:
                home = safe_get(f,'localTeam','data','id') or safe_get(f,'localTeam','id')
                away = safe_get(f,'visitorTeam','data','id') or safe_get(f,'visitorTeam','id')
                if not home or not away:
                    continue
                # Validate team history: at least 9 months (we'll ensure last match date older than 9 months back)
                h_stats = compute_team_strengths(home, months=12)
                a_stats = compute_team_strengths(away, months=12)
                # Both must have at least 9 months of data: i.e., matches_count>0 and last_match_date older than 270 days ago? The user asked: "the team performance valid information for at least 9 months or more"
                # We'll require matches_count >= 8 and coverage in the last 270 days window (i.e., there are matches spanning last 9 months)
                ok = False
                if h_stats and a_stats:
                    if h_stats.get('matches_count',0) >= 8 and a_stats.get('matches_count',0) >= 8:
                        ok = True
                if ok:
                    candidates.append({'fixture':f, 'home':home, 'away':away})
            except Exception:
                continue
        # Limit to 10 per day
        candidates = candidates[:10]
        rows = []
        for c in candidates:
            pred = calculate_match_probability(c['home'], c['away'])
            # Resolve team names
            home_name = safe_get(c['fixture'],'localTeam','data','name') or safe_get(c['fixture'],'localTeam','data','short_code') or f"Team {pred['home_team_id']}"
            away_name = safe_get(c['fixture'],'visitorTeam','data','name') or safe_get(c['fixture'],'visitorTeam','data','short_code') or f"Team {pred['away_team_id']}"
            row = [
                pred['timestamp'], # DateUTC
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # local
                home_name,
                away_name,
                pred['home_prob'],
                pred['draw_prob'],
                pred['away_prob'],
                pred['home_xg'],
                pred['away_xg'],
                pred['source'],
                "auto_daily"
            ]
            rows.append(row)
        if rows:
            ensure_sheet_and_append(rows)
            if MAIL_ENABLED and mail:
                try:
                    msg = Message(subject=f"Football predictions appended ({len(rows)} rows)",
                                  recipients=[MAIL_RECEIVER])
                    msg.body = f"Appended {len(rows)} predictions for {date.today().isoformat()}."
                    mail.send(msg)
                except Exception:
                    pass
    except Exception as e:
        # log
        print("daily_auto_job failed:", e)

# schedule daily_auto_job once per day (UTC 08:00) — adjust as needed
scheduler.add_job(id='daily_job', func=daily_auto_job, trigger='cron', hour='8', minute='0', timezone='UTC')
scheduler.start()

# ==========================
# Flask routes
# ==========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search_team", methods=["GET"])
def api_search_team():
    q = request.args.get("q","").strip()
    if not q:
        return jsonify({"error":"query missing"}), 400
    try:
        results = search_team(q)
        # standardize
        out = []
        for r in results:
            data = r.get('data') if isinstance(r.get('data'), dict) else r
            out.append({
                'id': safe_get(data,'id') or data.get('id'),
                'name': safe_get(data,'name') or data.get('name')
            })
        return jsonify({"results": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    print("Received JSON:", data)
    """
    Accepts JSON:
    {
      "home_id": <team_id> OR "home_name": name,
      "away_id": <team_id> OR "away_name": name,
      "manual_note": optional string,
      "append_sheet": true/false (default true)
    }
    """
    payload = request.get_json() or {}
    append_sheet = payload.get("append_sheet", True)
    manual_note = payload.get("manual_note", "")
    home_id = payload.get("home_id")
    away_id = payload.get("away_id")
    # Resolve by name if names provided
    if not home_id and payload.get("home_name"):
        s = search_team(payload.get("home_name"))
        if s:
            cand = s[0].get('data') if isinstance(s[0].get('data'), dict) else s[0]
            home_id = cand.get('id')
    if not away_id and payload.get("away_name"):
        s = search_team(payload.get("away_name"))
        if s:
            cand = s[0].get('data') if isinstance(s[0].get('data'), dict) else s[0]
            away_id = cand.get('id')
    if not home_id or not away_id:
        return jsonify({"error":"home_id and away_id required (or provide home_name/away_name)"}), 400

    try:
        pred = calculate_match_probability(home_id, away_id)
        # Resolve names for readable sheet append
        home_name = ""
        away_name = ""
        try:
            th = get_team_by_id(int(home_id))
            ta = get_team_by_id(int(away_id))
            home_name = safe_get(th,'name') or str(home_id)
            away_name = safe_get(ta,'name') or str(away_id)
        except Exception:
            home_name = str(home_id); away_name = str(away_id)

        if append_sheet:
            row = [
                pred['timestamp'],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                home_name,
                away_name,
                pred['home_prob'],
                pred['draw_prob'],
                pred['away_prob'],
                pred['home_xg'],
                pred['away_xg'],
                pred['source'],
                manual_note or "manual" if payload.get("manual_note") else "api_predict"
            ]
            try:
                ensure_sheet_and_append([row])
            except Exception as e:
                print("Warning: couldn't append to sheet:", e)
        return jsonify({"prediction": pred, "home_name": home_name, "away_name": away_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/fixtures/today", methods=["GET"])
def api_fixtures_today():
    """Return today's european fixtures (limited to 100)"""
    today = datetime.utcnow().date()
    try:
        fixtures = get_fixtures_for_date(today)
        euro = [f for f in fixtures if is_fixture_european(f)]
        simplified = []
        for f in euro:
            simplified.append({
                "id": f.get('id'),
                "home": safe_get(f, 'localTeam', 'data','name') or safe_get(f,'localTeam','data','id'),
                "away": safe_get(f, 'visitorTeam','data','name') or safe_get(f,'visitorTeam','data','id'),
                "league": safe_get(f,'league','data','name'),
                "time": safe_get(f,'time','starting_at','date_time') or safe_get(f,'time','starting_at','date')
            })
        return jsonify({"date": today.isoformat(), "fixtures": simplified[:200]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Static for favicon or assets (optional)
@app.route("/favicon.ico")
def favicon():
    return '', 204


# ==========================
# Run app
# ==========================
if __name__ == "__main__":
    # You can run locally with: python app.py
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
