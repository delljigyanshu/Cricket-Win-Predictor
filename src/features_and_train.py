# src/features_and_train.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib

CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'ball_by_ball.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading:", CSV)
df = pd.read_csv(CSV)

# keep only valid
df = df[df['target'].notnull()].copy()
df = df[df['winner'].notnull()].copy()
df['label'] = df.apply(lambda r: 1 if (r['winner'] == r['team']) else 0, axis=1)

print("Rows:", len(df), "Label counts:", df['label'].value_counts().to_dict())

# Build simple player-form:
#  - batsman_form: rolling avg runs per match (last 5 matches)
#  - bowler_form: rolling avg wickets per match (last 5 matches)
if 'batsman' in df.columns and 'bowler' in df.columns and 'match_id' in df.columns:
    # aggregate per match
    bat_match = df.groupby(['match_id', 'batsman'])['runs_in_ball'].sum().reset_index()
    bat_match = bat_match.sort_values(['batsman', 'match_id'])
    bat_forms = {}
    for batsman, g in bat_match.groupby('batsman'):
        g = g.reset_index(drop=True)
        g['rolling'] = g['runs_in_ball'].rolling(5, min_periods=1).mean()
        # store last rolling keyed by match
        bat_forms[batsman] = g.set_index('match_id')['rolling'].to_dict()

    bowl_match = df.groupby(['match_id', 'bowler'])['is_wicket'].sum().reset_index()
    bowl_match = bowl_match.sort_values(['bowler', 'match_id'])
    bowl_forms = {}
    for bowler, g in bowl_match.groupby('bowler'):
        g = g.reset_index(drop=True)
        g['rolling'] = g['is_wicket'].rolling(5, min_periods=1).mean()
        bowl_forms[bowler] = g.set_index('match_id')['rolling'].to_dict()

    # default medians
    global_bat = np.nanmedian([v for d in bat_forms.values() for v in d.values()]) if bat_forms else 0.0
    global_bowl = np.nanmedian([v for d in bowl_forms.values() for v in d.values()]) if bowl_forms else 0.0

    def map_bat_form(row):
        try:
            mid = row['match_id']
            b = row['batsman']
            return bat_forms.get(b, {}).get(mid, global_bat)
        except:
            return global_bat

    def map_bowl_form(row):
        try:
            mid = row['match_id']
            b = row['bowler']
            return bowl_forms.get(b, {}).get(mid, global_bowl)
        except:
            return global_bowl

    df['batsman_form'] = df.apply(map_bat_form, axis=1)
    df['bowler_form'] = df.apply(map_bowl_form, axis=1)
else:
    df['batsman_form'] = 0.0
    df['bowler_form'] = 0.0

# features
def add_features(d):
    d = d.copy()
    d['wickets_in_hand'] = 10 - d['wickets_before']
    d['frac_innings_complete'] = d['balls_elapsed'] / 120.0
    d['runs_required_norm'] = d['runs_required'] / d['target']
    d['rr_diff'] = d['current_run_rate'] - d['req_run_rate']
    d['pressure'] = d['runs_required'] / (d['balls_remaining'] + 1) * (1.0 / (d['wickets_in_hand'] + 1))
    d['over_int'] = d['over_ball'].apply(lambda x: int(str(x).split('.')[0]) if pd.notnull(x) else 0)
    return d

df = add_features(df)

FEATURE_COLS = [
    'score_before', 'wickets_before', 'overs_completed', 'balls_remaining',
    'runs_required', 'req_run_rate', 'current_run_rate', 'wickets_in_hand',
    'frac_innings_complete', 'runs_required_norm', 'rr_diff',
    'pressure', 'over_int', 'batsman_form', 'bowler_form'
]

train_df = df.dropna(subset=FEATURE_COLS + ['label']).copy()
X = train_df[FEATURE_COLS].values
y = train_df['label'].values

# quick random split (for real eval use chronological split by date)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_s, y_train)
p_lr = lr.predict_proba(X_test_s)[:,1]
print("LR AUC:", roc_auc_score(y_test, p_lr), "LogLoss:", log_loss(y_test, p_lr))

# GB
try:
    import xgboost as xgb
    gb = xgb.XGBClassifier(n_estimators=200, eval_metric='logloss', use_label_encoder=False, n_jobs=-1)
    gb.fit(X_train, y_train)
except Exception:
    gb = GradientBoostingClassifier(n_estimators=200)
    gb.fit(X_train, y_train)

p_gb = gb.predict_proba(X_test)[:,1]
print("GB AUC (before calib):", roc_auc_score(y_test, p_gb))

# Calibrate using holdout (isotonic)
calib = CalibratedClassifierCV(gb, method='isotonic', cv='prefit')
calib.fit(X_test, y_test)
p_gb_cal = calib.predict_proba(X_test)[:,1]
print("GB AUC (calibrated):", roc_auc_score(y_test, p_gb_cal), "LogLoss:", log_loss(y_test, p_gb_cal))

# save models
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
joblib.dump(lr, os.path.join(MODEL_DIR, 'logistic.joblib'))
joblib.dump(gb, os.path.join(MODEL_DIR, 'gb.joblib'))
joblib.dump(calib, os.path.join(MODEL_DIR, 'gb_calibrated.joblib'))
print("Saved models to:", MODEL_DIR)
