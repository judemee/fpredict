#!/usr/bin/env python3
import os
import json
import math
import time
from datetime import datetime, timedelta, timezone, date
from functools import lru_cache
import pytz
from dotenv import load_dotenv
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson
import traceback
import schedule
import threading
import time



from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_apscheduler import APScheduler
from flask_mail import Mail, Message

import gspread
from google.oauth2.service_account import Credentials

# -------------------------
# Load env (once) & prints
# -------------------------
load_dotenv(override=True)

SPORTMONKS_API_TOKEN = os.getenv("SPORTMONKS_API_TOKEN")
if SPORTMONKS_API_TOKEN:
    SPORTMONKS_API_TOKEN = SPORTMONKS_API_TOKEN.strip()
else:
    print("⚠️ SPORTMONKS_API_TOKEN not found in environment!")

SPORTMONKS_BASE = os.environ.get("SPORTMONKS_BASE", "https://api.sportmonks.com/v3/football")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
GOOGLE_SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")

print("Loaded ENV variables:")
print("SPORTMONKS_API_TOKEN:", SPORTMONKS_API_TOKEN)
print("GOOGLE_SHEET_ID:", GOOGLE_SHEET_ID)
print("MAIL_ENABLED:", os.environ.get("MAIL_ENABLED"))
print("MAIL_SERVER:", os.environ.get("MAIL_SERVER"))
print("MAIL_USERNAME:", os.environ.get("MAIL_USERNAME"))

# -------------------------
# Config
# -------------------------
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

os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------
# Flask + Mail + Scheduler
# -------------------------
app = Flask(__name__, template_folder="templates")
app.config['JSON_SORT_KEYS'] = False

app.config.update(
    MAIL_SERVER=MAIL_SERVER,
    MAIL_PORT=MAIL_PORT,
    MAIL_USERNAME=MAIL_USERNAME,
    MAIL_PASSWORD=MAIL_PASSWORD,
    MAIL_USE_TLS=True if MAIL_PORT == 587 else False,
    MAIL_DEFAULT_SENDER=MAIL_DEFAULT_SENDER
)
mail = Mail(app) if MAIL_ENABLED else None

class Config:
    SCHEDULER_API_ENABLED = True

app.config.from_object(Config())
scheduler = APScheduler()
scheduler.init_app(app)

# -------------------------
# Google sheets client (lazy)
# -------------------------
def get_gspread_client():
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_FILE, scopes=scopes)
    client = gspread.authorize(creds)
    return client

# -------------------------
# SportMonks helpers
# -------------------------
def sportmonks_get(path, params=None):
    """Simple wrapper with token and basic caching (file-based short TTL)."""
    if params is None:
        params = {}
    params['api_token'] = SPORTMONKS_API_TOKEN
    url = f"{SPORTMONKS_BASE.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

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
        # debug-friendly snippet (no huge prints)
        if DEBUG:
            print("SportMonks search_team query:", query)
            print("SportMonks raw response (snippet):", json.dumps(res, indent=2)[:500])
        return res.get('data', [])
    except Exception as e:
        if DEBUG:
            print("SportMonks search_team ERROR:", e)
        return []

def resolve_team(input_value):
    """Resolve id or text -> full team dict or None"""
    if not input_value:
        return None
    # numeric id?
    try:
        maybe_id = int(str(input_value).strip())
        team = get_team_by_id(maybe_id)
        if team:
            return team
    except Exception:
        pass
    query = str(input_value).strip()
    if not query:
        return None
    results = search_team(query)
    if not results:
        return None
    qnorm = query.lower()
    def unwrap(r):
        return r.get('data') if isinstance(r.get('data'), dict) else r
    # exact match
    for r in results:
        t = unwrap(r)
        if not isinstance(t, dict):
            continue
        name = (t.get('name') or "").lower()
        short = (t.get('short_code') or t.get('code') or "").lower()
        if qnorm == name or qnorm == short:
            try:
                tid = int(t.get('id'))
                full = get_team_by_id(tid)
                return full or t
            except Exception:
                return t
    # substring match
    for r in results:
        t = unwrap(r)
        if not isinstance(t, dict):
            continue
        name = (t.get('name') or "").lower()
        short = (t.get('short_code') or t.get('code') or "").lower()
        if qnorm in name or qnorm in short or name in qnorm:
            try:
                tid = int(t.get('id'))
                full = get_team_by_id(tid)
                return full or t
            except Exception:
                return t
    # fallback first
    first = unwrap(results[0])
    if isinstance(first, dict):
        try:
            tid = int(first.get('id'))
            full = get_team_by_id(tid)
            return full or first
        except Exception:
            return first
    return None

# -------------------------
# Fixtures helpers
# -------------------------
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

# -------------------------
# Utilities and prediction
# -------------------------
EUROPE_ISO = set([
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
    from datetime import datetime, timezone
    to_date = datetime.now(timezone.utc).date()
    from_date = to_date - timedelta(days=months*30)
    fixtures = get_team_fixtures(team_id, from_date=from_date, to_date=to_date)
    print(f"🔍 Fetching fixtures for team_id={team_id}")
    print(f"📊 Fetched {len(fixtures) if fixtures else 0} fixtures from SportMonks")
    if not fixtures:
        print("⚠️ No fixtures returned — possible name mismatch or API limitation.")

    if not fixtures:
        return None
    gf = []; ga = []; dates = []
    for f in fixtures:
        status = f.get('time', {}).get('status')
        if status not in ('FT', 'AET', 'FT_PEN', 'After ET'): continue
        local = f.get('localTeam', {}); visitor = f.get('visitorTeam', {})
        local_goals = safe_get(f, 'scores', 'localteam_score')
        visitor_goals = safe_get(f, 'scores', 'visitorteam_score')
        if local_goals is None and 'score' in f:
            local_goals = safe_get(f, 'score', 'localteam')
            visitor_goals = safe_get(f, 'score', 'visitorteam')
        if local_goals is None or visitor_goals is None: continue
        try:
            local_goals = int(local_goals); visitor_goals = int(visitor_goals)
        except Exception:
            continue
        local_id = safe_get(local, 'data', 'id') or local.get('id')
        try:
            if local_id and int(local_id) == int(team_id):
                team_goals = local_goals; opp_goals = visitor_goals
            else:
                team_goals = visitor_goals; opp_goals = local_goals
        except Exception:
            if str(local_id) == str(team_id):
                team_goals = local_goals; opp_goals = visitor_goals
            else:
                team_goals = visitor_goals; opp_goals = local_goals
        gf.append(team_goals); ga.append(opp_goals)
        match_date_s = safe_get(f, 'time', 'starting_at', 'date_time') or safe_get(f, 'time', 'starting_at', 'date')
        try:
            dates.append(datetime.fromisoformat(match_date_s))
        except Exception:
            pass
    matches_count = len(gf)
    if matches_count < min_matches_required:
        return {'matches_count': matches_count, 'last_match_date': max(dates).date() if dates else None} if matches_count>0 else None
    attack = np.mean(gf); defense = np.mean(ga)
    last_date = max(dates).date() if dates else None
    return {'attack': float(attack), 'defense': float(defense), 'avg_for': float(attack), 'avg_against': float(defense), 'matches_count': matches_count, 'last_match_date': last_date}

def poisson_score_probs(home_xg, away_xg, max_goals=6):
    probs = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            probs[i,j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
    return probs

def calculate_match_probability(team_home_id, team_away_id, source_hint="sportmonks"):
    home_stats = compute_team_strengths(team_home_id, months=12)
    away_stats = compute_team_strengths(team_away_id, months=12)
    used_source = "sportmonks"
    if not home_stats or not away_stats or home_stats.get('matches_count',0) < 8 or away_stats.get('matches_count',0) < 8:
        used_source = "ai_fallback"
        def fallback_avg(team_id, n=6):
            try:
                fixtures = get_team_fixtures(team_id, to_date=datetime.now(timezone.utc).date())
            except Exception:
                fixtures = []
            gf=[]; ga=[]
            for f in sorted(fixtures, key=lambda x: safe_get(x,'time','starting_at','date_time') or safe_get(x,'time','starting_at','date') or '')[::-1]:
                status = safe_get(f,'time','status')
                if status not in ('FT','AET','FT_PEN','After ET'): continue
                local_goals = safe_get(f,'scores','localteam_score'); visitor_goals = safe_get(f,'scores','visitorteam_score')
                if local_goals is None or visitor_goals is None: continue
                try:
                    local_goals = int(local_goals); visitor_goals = int(visitor_goals)
                except Exception:
                    continue
                local_id = safe_get(f,'localTeam','data','id') or safe_get(f,'localTeam','id') or None
                try:
                    if local_id and int(local_id)==int(team_id):
                        gf.append(local_goals); ga.append(visitor_goals)
                    else:
                        gf.append(visitor_goals); ga.append(local_goals)
                except Exception:
                    if str(local_id) == str(team_id):
                        gf.append(local_goals); ga.append(visitor_goals)
                    else:
                        gf.append(visitor_goals); ga.append(local_goals)
                if len(gf)>=n: break
            if len(gf)==0: return None
            return {'attack': float(np.mean(gf)), 'defense': float(np.mean(ga)), 'matches_count': len(gf)}
        home_stats = fallback_avg(team_home_id) or {'attack':1.2,'defense':1.2,'matches_count':0}
        away_stats = fallback_avg(team_away_id) or {'attack':1.0,'defense':1.0,'matches_count':0}

    HOME_ADV = 1.08
    home_xg = (home_stats['attack'] * (away_stats['defense'] if away_stats['defense']>0 else 1) ) / (home_stats['defense'] if home_stats['defense']>0 else 1)
    away_xg = (away_stats['attack'] * (home_stats['defense'] if home_stats['defense']>0 else 1) ) / (away_stats['defense'] if away_stats['defense']>0 else 1)
    home_xg = max(0.1, home_xg * HOME_ADV * 0.6)
    away_xg = max(0.05, away_xg * 0.6)
    if home_stats.get('matches_count',0) < 8: home_xg *= 0.9
    if away_stats.get('matches_count',0) < 8: away_xg *= 0.9
    home_xg = float(min(home_xg, 4.5))
    away_xg = float(min(away_xg, 4.0))

    score_mat = poisson_score_probs(home_xg, away_xg, max_goals=6)
    home_wins = float(np.sum(np.where(np.arange(0,7)[:,None] > np.arange(0,7)[None,:], score_mat, 0.0)))
    away_wins = float(np.sum(np.where(np.arange(0,7)[:,None] < np.arange(0,7)[None,:], score_mat, 0.0)))
    draw_prob = float(np.sum(np.diag(score_mat)))
    total = home_wins + away_wins + draw_prob
    if total <= 0:
        home_p = 0.45; draw_p = 0.2; away_p = 0.35
    else:
        home_p = min(0.99, home_wins/total)
        away_p = min(0.99, away_wins/total)
        draw_p = min(0.99, draw_prob/total)

    return {
        'home_team_id': int(team_home_id),
        'away_team_id': int(team_away_id),
        'home_xg': round(home_xg,3),
        'away_xg': round(away_xg,3),
        'home_prob': round(home_p,3),
        'draw_prob': round(draw_p,3),
        'away_prob': round(away_p,3),
        'source': used_source,
        'timestamp': datetime.now(timezone.utc).isoformat()

    }

# -------------------------
# Google Sheets functions
# -------------------------
SHEET_HEADERS = ["DateUTC","DateLocal","HomeTeam","AwayTeam","HomeProb","DrawProb","AwayProb","Home_xG","Away_xG","Source","Notes"]

def ensure_sheet_and_append(data_rows, worksheet_title="Predictions"):
    if not GOOGLE_SHEET_ID:
        raise RuntimeError("GOOGLE_SHEET_ID not configured")
    client = get_gspread_client()
    sh = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        worksheet = sh.worksheet(worksheet_title)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sh.add_worksheet(worksheet_title, rows="1000", cols=str(len(SHEET_HEADERS)))
        worksheet.append_row(SHEET_HEADERS)
    for r in data_rows:
        worksheet.append_row(r, value_input_option="USER_ENTERED")
    return True

# -------------------------
# Daily auto job (kept as-is)
# -------------------------
def is_fixture_european(fixture):
    league_country_iso = safe_get(fixture, "league", "data", "country", "ioc") or safe_get(fixture, "league", "data", "country", "iso") or safe_get(fixture, "country", "data", "iso")
    if league_country_iso:
        try:
            if league_country_iso.upper() in EUROPE_ISO:
                return True
        except Exception:
            pass
    league_name = safe_get(fixture, "league", "data", "name") or ""
    if any(kw in league_name for kw in ["Championship","Premier","La Liga","Serie","Bundesliga","Ligue","Primeira","Eredivisie","Super Lig","Ukrain","Greek","Polish","Austrian","Swiss"]):
        return True
    return False

def get_recent_fixtures(from_date=None, to_date=None):
    """
    Fetch fixtures between from_date and to_date from SportMonks.
    Filters to European fixtures only.
    """
    if from_date is None:
        from_date = datetime.date.today() - datetime.timedelta(days=2)
    if to_date is None:
        to_date = datetime.date.today() + datetime.timedelta(days=1)

    all_fixtures = []
    current = from_date
    while current <= to_date:
        daily = get_fixtures_for_date(current)
        for f in daily:
            if is_fixture_european(f):
                all_fixtures.append(f)
        current += datetime.timedelta(days=1)

    return all_fixtures


def predict_match(home_name, away_name, append_sheet=True):
    """
    Wrapper around /api/predict logic for automated daily job.
    Uses the same SportMonks or local fallback prediction logic.
    """
    try:
        # Try to resolve via SportMonks
        resolved_home = resolve_team(home_name)
        resolved_away = resolve_team(away_name)

        if not resolved_home or not resolved_away:
            # fallback to local model if team lookup fails
            pred = local_model_predict(home_name, away_name)
        else:
            pred = calculate_match_probability(int(resolved_home['id']), int(resolved_away['id']))

        if append_sheet:
            row = [
                pred['timestamp'],
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            try:
                ensure_sheet_and_append([row])
            except Exception as e:
                print(f"⚠️ Could not append to Google Sheet: {e}")

        return pred

    except Exception as e:
        print(f"⚠️ predict_match error for {home_name} vs {away_name}: {e}")
        return None


def daily_auto_job():
    """
    Automatically fetches recent fixtures and appends predictions to Google Sheets.
    Includes SportMonks + fallback handling.
    """
    print("\n[AutoJob] Starting daily auto job at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    appended_count = 0
    skipped_count = 0

    try:
        # Example: get upcoming fixtures (last 2 days + next 1 day)
        from_date = (datetime.date.today() - datetime.timedelta(days=2))
        to_date = (datetime.date.today() + datetime.timedelta(days=1))
        fixtures = get_recent_fixtures(from_date=from_date, to_date=to_date)
        print(f"[AutoJob] Retrieved {len(fixtures)} fixtures to process.")

        for f in fixtures:
            try:
                home = safe_get(f, "home", "name") or safe_get(f, "localTeam", "data", "name")
                away = safe_get(f, "away", "name") or safe_get(f, "visitorTeam", "data", "name")
                if not home or not away:
                    skipped_count += 1
                    continue

                # Run prediction (will auto fallback if SportMonks fails)
                result = predict_match(home, away, append_sheet=True)

                if result:
                    appended_count += 1
                    print(f"✅ Appended: {home} vs {away} → {result.get('source', 'unknown')}")
                else:
                    print(f"⚠️ Skipped {home} vs {away} (no result)")
                    skipped_count += 1

            except Exception as e:
                print(f"❌ Error processing fixture: {e}")
                traceback.print_exc()
                skipped_count += 1

        print(f"\n[AutoJob Completed @ {datetime.datetime.now().strftime('%H:%M:%S')}]")
        print(f"🟩 Appended: {appended_count}")
        print(f"🟨 Skipped/Failed: {skipped_count}\n")

    except Exception as e:
        print(f"🚨 [AutoJob] Fatal error: {e}")
        traceback.print_exc()
        

# -------------------------
# Flask routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search_team", methods=["GET"])
def api_search_team_route():
    q = request.args.get("q","").strip()
    if not q:
        return jsonify({"error":"query missing"}), 400
    try:
        results = search_team(q)
        out = []
        for r in results:
            data = r.get('data') if isinstance(r.get('data'), dict) else r
            out.append({'id': safe_get(data,'id') or data.get('id'), 'name': safe_get(data,'name') or data.get('name')})
        return jsonify({"results": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# Local model fallback helpers
# -------------------------
def local_model_predict(home_name, away_name):
    """
    Try to call external local model module if present:
      - expects predict_with_local_model.predict_match(home_name, away_name) -> dict
    Otherwise fallback to built-in elo_fallback().
    """
    # Try to import user-provided model
    try:
        import predict_with_local_model as pwm
        if hasattr(pwm, "predict_match"):
            res = pwm.predict_match(home_name, away_name)
            if isinstance(res, dict):
                return res
    except Exception as e:
        if DEBUG:
            print("local_model_predict: no external model or error:", e)

    # Built-in simple Elo fallback
    return elo_fallback(home_name, away_name)

def elo_fallback(home_name, away_name):
    """
    Advanced local model fallback using recent fixtures (if available)
    and an Elo-like logic when data is incomplete.
    """

    # Helper to get last N match stats from SportMonks if team name is resolvable
    def recent_form(team_name, n=10):
        try:
            team = resolve_team(team_name)
            if not team or not team.get("id"):
                return None
            team_id = int(team["id"])
            fixtures = get_team_fixtures(team_id, to_date=datetime.now(timezone.utc).date())
            results = []
            for f in sorted(
                fixtures,
                key=lambda x: safe_get(x, "time", "starting_at", "date_time")
                or safe_get(x, "time", "starting_at", "date")
                or "",
                reverse=True,
            ):
                if safe_get(f, "time", "status") not in ("FT", "AET", "FT_PEN", "After ET"):
                    continue
                lg = safe_get(f, "scores", "localteam_score")
                vg = safe_get(f, "scores", "visitorteam_score")
                if lg is None or vg is None:
                    continue
                try:
                    lg, vg = int(lg), int(vg)
                except Exception:
                    continue
                lid = safe_get(f, "localTeam", "data", "id") or safe_get(f, "localTeam", "id")
                if str(lid) == str(team_id):
                    results.append((lg, vg))  # scored, conceded
                else:
                    results.append((vg, lg))
                if len(results) >= n:
                    break
            if not results:
                return None
            # Weighted average (recent matches weighted more)
            weights = np.linspace(0.5, 1.0, num=len(results))
            weights /= weights.sum()
            gf = np.dot([r[0] for r in results], weights)
            ga = np.dot([r[1] for r in results], weights)
            return {"gf": gf, "ga": ga, "count": len(results)}
        except Exception:
            return None

    home_form = recent_form(home_name, 10)
    away_form = recent_form(away_name, 10)

    # Fallback to pseudo-Elo if we don't get enough data
    def pseudo_elo(name):
        s = sum(ord(c) for c in (name or ""))
        return 1400 + (s % 400)

    home_elo = pseudo_elo(home_name)
    away_elo = pseudo_elo(away_name)

    # Use form if available to slightly adjust Elo baseline
    if home_form:
        home_elo += (home_form["gf"] - home_form["ga"]) * 20
    if away_form:
        away_elo += (away_form["gf"] - away_form["ga"]) * 20

    # Compute win expectancy (Elo-style)
    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    expected_away = 1 - expected_home
    draw_prob = 0.25 * (1 - abs(expected_home - 0.5) * 2)

    # Normalize
    home_p = expected_home * (1 - draw_prob)
    away_p = expected_away * (1 - draw_prob)
    total = home_p + away_p + draw_prob
    home_p, draw_prob, away_p = [x / total for x in (home_p, draw_prob, away_p)]

    # Estimate xG: based on form or derived from probabilities
    if home_form:
        home_xg = min(3.0, 1.0 + 0.4 * home_form["gf"])
    else:
        home_xg = 1.2 + (home_elo - 1500) / 1000
    if away_form:
        away_xg = min(2.5, 0.9 + 0.4 * away_form["gf"])
    else:
        away_xg = 1.0 + (away_elo - 1500) / 1200

    return {
        "home_team_id": None,
        "away_team_id": None,
        "home_xg": round(home_xg, 3),
        "away_xg": round(away_xg, 3),
        "home_prob": round(home_p, 3),
        "draw_prob": round(draw_prob, 3),
        "away_prob": round(away_p, 3),
        "source": "elo_hybrid_local",
        "confidence": round(0.5 + 0.1 * ((home_form is not None) + (away_form is not None)), 2),
        'timestamp': datetime.now(timezone.utc).isoformat()

    }


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts JSON:
    {
      "home_id": <team_id> OR "home_name": name,
      "away_id": <team_id> OR "away_name": name,
      "manual_note": optional string,
      "append_sheet": true/false (default true)
    }
    """
    data = request.get_json(force=True) or {}
    print("Received JSON:", data)

    append_sheet = data.get("append_sheet", True)
    manual_note = data.get("manual_note", "")

    home_id_in = data.get("home_id")
    away_id_in = data.get("away_id")
    home_name_in = data.get("home_name")
    away_name_in = data.get("away_name")

    # normalize strings
    if isinstance(home_name_in, str): home_name_in = home_name_in.strip()
    if isinstance(away_name_in, str): away_name_in = away_name_in.strip()

    # Attempt to resolve via sportmonks first
    resolved_home = None
    resolved_away = None

    if home_id_in is not None:
        try:
            resolved_home = get_team_by_id(int(home_id_in))
            if not resolved_home:
                resolved_home = {'id': int(home_id_in), 'name': str(home_id_in)}
        except Exception:
            resolved_home = None

    if away_id_in is not None:
        try:
            resolved_away = get_team_by_id(int(away_id_in))
            if not resolved_away:
                resolved_away = {'id': int(away_id_in), 'name': str(away_id_in)}
        except Exception:
            resolved_away = None

    if home_name_in and not resolved_home:
        try:
            resolved_home = resolve_team(home_name_in)
        except Exception as e:
            if DEBUG: print("resolve home error:", e)
            resolved_home = None

    if away_name_in and not resolved_away:
        try:
            resolved_away = resolve_team(away_name_in)
        except Exception as e:
            if DEBUG: print("resolve away error:", e)
            resolved_away = None

    # If we didn't get both teams from API, use local model fallback
    use_local_model = False
    if not resolved_home or not resolved_away:
        print("Warning: SportMonks resolution incomplete -> falling back to local model")
        use_local_model = True

    if not use_local_model:
        try:
            final_home_id = int(resolved_home.get('id'))
            final_away_id = int(resolved_away.get('id'))
        except Exception:
            use_local_model = True

    if use_local_model:
        # prefer using names passed in, else resolved objects if any
        hn = home_name_in or safe_get(resolved_home, 'name') or "Home"
        an = away_name_in or safe_get(resolved_away, 'name') or "Away"
        local_pred = local_model_predict(hn, an)
        # attach names for clarity
        home_name_out = hn
        away_name_out = an
        response = {
            "prediction": local_pred,
            "home_name": home_name_out,
            "away_name": away_name_out,
            "source": local_pred.get("source", "local_model")
        }
        # optionally append to sheet
        if append_sheet:
            try:
                row = [
                    local_pred.get('timestamp'),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    home_name_out,
                    away_name_out,
                    local_pred.get('home_prob'),
                    local_pred.get('draw_prob'),
                    local_pred.get('away_prob'),
                    local_pred.get('home_xg'),
                    local_pred.get('away_xg'),
                    local_pred.get('source'),
                    manual_note or "local_fallback"
                ]
                try:
                    ensure_sheet_and_append([row])
                except Exception as e:
                    print("Warning: couldn't append fallback to sheet:", e)
            except Exception:
                pass
        return jsonify(response)

    # Otherwise proceed with sportmonks-based prediction
    try:
        pred = calculate_match_probability(final_home_id, final_away_id)
        th = resolved_home
        ta = resolved_away
        home_name_out = safe_get(th, 'name') or str(final_home_id)
        away_name_out = safe_get(ta, 'name') or str(final_away_id)

        if append_sheet:
            row = [
                pred['timestamp'],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                home_name_out,
                away_name_out,
                pred['home_prob'],
                pred['draw_prob'],
                pred['away_prob'],
                pred['home_xg'],
                pred['away_xg'],
                pred['source'],
                manual_note or "api_predict"
            ]
            try:
                ensure_sheet_and_append([row])
            except Exception as e:
                print("Warning: couldn't append to sheet:", e)

        return jsonify({
            "prediction": pred,
            "home_name": home_name_out,
            "away_name": away_name_out,
            "source": pred.get("source", "sportmonks")
        })
    except Exception as e:
        print("Error computing sportmonks prediction:", e)
        # as a last resort, run local fallback
        hn = home_name_in or safe_get(resolved_home, 'name') or "Home"
        an = away_name_in or safe_get(resolved_away, 'name') or "Away"
        local_pred = local_model_predict(hn, an)
        return jsonify({
            "prediction": local_pred,
            "home_name": hn,
            "away_name": an,
            "source": local_pred.get("source", "local_model"),
            "note": "sportmonks prediction failed, returned local fallback"
        }), 200

@app.route("/api/fixtures/today", methods=["GET"])
def api_fixtures_today():
    today = datetime.now(timezone.utc).date()
    try:
        fixtures = get_fixtures_for_date(today)
        euro = [f for f in fixtures if is_fixture_european(f)]
        simplified = []
        for f in euro:
            simplified.append({
                "id": f.get('id'),
                "home": safe_get(f, 'localTeam', 'data','name') or safe_get(f,'localTeam','data','id'),
                "away": safe_get(f, 'visitorTeam', 'data','name') or safe_get(f,'visitorTeam','data','id'),
                "league": safe_get(f,'league','data','name'),
                "time": safe_get(f,'time','starting_at','date_time') or safe_get(f,'time','starting_at','date')
            })
        return jsonify({"date": today.isoformat(), "fixtures": simplified[:200]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/favicon.ico")
def favicon():
    return '', 204

# -------------------------
# Run + Scheduler
# -------------------------
def schedule_auto_job():
    """
    Background scheduler that runs daily_auto_job() automatically at a set time.
    """
    # 🕘 Set the time you want the job to run daily (24h format)
    schedule.every().day.at("09:00").do(daily_auto_job)
    print("[Scheduler] AutoJob daily scheduler started — will run every day at 09:00.")
    while True:
        schedule.run_pending()
        time.sleep(60)  # check every 60 seconds


if __name__ == "__main__":
    # Start the scheduler in a background thread
    threading.Thread(target=schedule_auto_job, daemon=True).start()

    print("[App] Flask server starting on http://0.0.0.0:8080 ...")
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
