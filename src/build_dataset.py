# src/build_dataset.py
import os
import pandas as pd
from tqdm import tqdm
from utils import load_match_file

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "cricsheet")
OUT_CSV = os.path.join(BASE_DIR, "data", "ball_by_ball.csv")

def extract_winner(match):
    if not isinstance(match, dict):
        return None
    # try common locations
    outcome = match.get("outcome")
    if isinstance(outcome, dict):
        w = outcome.get("winner")
        if w:
            return w
    info = match.get("info")
    if isinstance(info, dict):
        oc = info.get("outcome")
        if isinstance(oc, dict):
            w = oc.get("winner")
            if w:
                return w
        w2 = info.get("winner")
        if w2:
            return w2
    # fallback: any top-level winner-like fields
    for k in ["winner", "result", "match_winner"]:
        if k in match and isinstance(match[k], str):
            return match[k]
    return None

def extract_match_id_and_date(match):
    mid = None
    date = None
    info = match.get("info")
    if isinstance(info, dict):
        mid = info.get("match_id") or info.get("id")
        dates = info.get("dates") or info.get("date") or info.get("start_date")
        if isinstance(dates, list) and dates:
            date = dates[0]
        elif isinstance(dates, str):
            date = dates
    return mid, date

def process_match(match):
    rows = []
    winner = extract_winner(match)
    match_id, match_date = extract_match_id_and_date(match)
    innings = match.get("innings", [])
    # first innings total
    first_inn_total = None
    if len(innings) >= 1:
        try:
            k0 = list(innings[0].keys())[0]
            d0 = innings[0][k0]
            s = 0
            for deliv in d0.get("deliveries", []):
                for _, info in deliv.items():
                    s += info.get("runs", {}).get("total", 0)
            first_inn_total = s
        except Exception:
            first_inn_total = None

    for inn_idx, inn in enumerate(innings):
        k = list(inn.keys())[0]
        inn_data = inn[k]
        team = inn_data.get("team")
        deliveries = inn_data.get("deliveries", [])
        cum_score = 0
        cum_wkts = 0
        ball_count = 0

        # detect overs limit
        overs_limit = 20
        top_info = match.get("info", {})
        if isinstance(top_info, dict) and top_info.get("overs"):
            try:
                overs_limit = int(top_info.get("overs"))
            except:
                pass

        for deliv in deliveries:
            for ball_str, info_ball in deliv.items():
                ball_count += 1
                runs_total = info_ball.get("runs", {}).get("total", 0)
                is_wicket = 1 if info_ball.get("wicket") else 0
                batsman = info_ball.get("batsman")
                bowler = info_ball.get("bowler")

                balls_elapsed = ball_count - 1
                overs_completed = balls_elapsed // 6 + (balls_elapsed % 6) / 6.0

                target = None
                runs_required = None
                balls_remaining = None
                req_run_rate = None
                current_run_rate = (cum_score) / (balls_elapsed / 6.0) if balls_elapsed > 0 else 0.0

                if inn_idx == 1 and first_inn_total is not None:
                    target = first_inn_total + 1
                    runs_required = max(0, target - cum_score)
                    balls_remaining = max(0, overs_limit * 6 - balls_elapsed)
                    req_run_rate = runs_required / (balls_remaining / 6.0) if balls_remaining > 0 else 999

                row = {
                    "match_id": match_id,
                    "date": match_date,
                    "innings": inn_idx + 1,
                    "team": team,
                    "over_ball": ball_str,
                    "overs_completed": overs_completed,
                    "balls_elapsed": balls_elapsed,
                    "score_before": cum_score,
                    "wickets_before": cum_wkts,
                    "runs_in_ball": runs_total,
                    "is_wicket": is_wicket,
                    "target": target,
                    "runs_required": runs_required,
                    "balls_remaining": balls_remaining,
                    "req_run_rate": req_run_rate,
                    "current_run_rate": current_run_rate,
                    "batsman": batsman,
                    "bowler": bowler,
                    "winner": winner,
                }
                rows.append(row)

                cum_score += runs_total
                if is_wicket == 1:
                    cum_wkts += 1

    return rows

def build_all():
    all_rows = []
    print("Searching YAML files in:", DATA_DIR)
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(('.yaml', '.yml', '.json'))]
    print("Found", len(files), "files")
    for f in tqdm(files):
        try:
            m = load_match_file(f)
            rows = process_match(m)
            all_rows.extend(rows)
        except Exception as e:
            print("Failed:", f, e)

    df = pd.DataFrame(all_rows)
    df = df[df["target"].notnull()].copy()
    df = df[df["winner"].notnull()].copy()
    df["label"] = df.apply(lambda r: 1 if (r["winner"] == r["team"]) else 0, axis=1)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("Saved", OUT_CSV, "rows:", len(df))

if __name__ == "__main__":
    build_all()
