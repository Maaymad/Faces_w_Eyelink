"""
Face familiarity and fixational eye movements -- analysis modeled on
Kadosh & Bonneh (2022), Sci Rep 12:20178, adapted to this task, plus the
link to duration reproduction found in the previous experiment.

Paper (free viewing, 4 s per face): fixation-related measures triggered by
the landing of saccades > 1 dva (here > SACC_MIN_DEG, see SETTINGS) --
  * msRT: latency of the first microsaccade (< 0.6 dva; here < MSRT_MAX_DEG) 150-600 ms after
    fixation onset (corrective microsaccades right after landing ignored);
    familiar faces -> SHORTER msRT (less oculomotor inhibition).
  * fixation duration (microsaccade-free fixations only); familiar shorter,
    not significant.
  * saccade size: no familiarity difference.
  * across successive saccades: adaptation for familiar faces.

Here the face is shown for 0.8 or 1.6 s after a forced fixation on a cross
below it, so the measures are computed while the face is on screen
(FACE_ONSET -> FACE_OFFSET; the display changes at offset, so later data is
not used). Added because of the design: the latency of the first saccade
from the cross to the face, and the number of saccades on the face.

Saccades and microsaccades are detected from the raw samples with Engbert &
Kliegl's velocity algorithm, on the AVERAGE of both eyes (see gaze_signals();
where one eye is missing, the other eye is used, corrected for the typical
offset between the eyes). The papers used one eye.

Familiarity: the participant's own rating (0-100) of each face; "familiar"
= rating >= FAMILIAR_CUTOFF (ratings are bimodal), also tested continuously.

Previous experiment: more familiar faces -> smaller reproduction error
(less over-reproduction), beta = -0.057, p = 0.053 (participant 40); the
number of saccades did not moderate it. Both are re-tested here.

Microsaccades are also detected a second way, IN ADDITION to the main
detection (see BINO_LAMBDA): both eyes, lenient threshold, binocular
criterion. Measures from it end in "_bino" (msrt_bino_ms, fix_dur_bino_ms,
ms_rate_bino_hz); all other measures are unchanged.

Run with venv311:
    venv311/bin/python familiarity_oculomotor_analysis.py
All eye-movement analyses are in this one file, run in this order by main():
  1. fixation-related measures (Kadosh & Bonneh, 2022) -- figures 1-8
  2. microsaccade amplitude histograms and rate per trial phase -- figure 9
  3. stimulus-locked analyses (Rosenzweig & Bonneh, 2019) -- figures 10-12

Outputs: data/analysis_familiarity/{figures,stats}
"""

import re
import warnings
from pathlib import Path

import eyelinkio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

####################################################
# SETTINGS
####################################################

DATA_DIR = Path(__file__).resolve().parent / "data"
# Participants: every EDF directly in DATA_DIR ("<id>_<session>.edf") that has
# a matching behavioral and ratings CSV ("<id>_sess<session>_*"). Others are
# listed and skipped. Subfolders (e.g. data/pilot, older design) are not used.


def find_subjects():
    found, skipped = [], []
    for edf in sorted(DATA_DIR.glob("*_*.edf")):
        sid, sess = edf.stem.rsplit("_", 1)
        beh = sorted(DATA_DIR.glob(f"{sid}_sess{sess}_*_behavioral.csv"))
        rat = sorted(DATA_DIR.glob(f"{sid}_sess{sess}_*_ratings.csv"))
        if beh and rat:
            found.append(dict(id=sid, edf=edf, behavioral=beh[-1], ratings=rat[-1]))
        else:
            skipped.append(f"{edf.name} (missing {'behavioral' if not beh else 'ratings'} CSV)")
    return found, skipped
OUT_DIR = DATA_DIR / "analysis_familiarity"
# Trial and participant exclusions decided on the behavioral data by
# behavioral_familiarity_lmm.R (reproduction > 4 s, > 3 MAD per participant x
# duration, flat / reversed / too steep timing slope). The same
# trials and participants are dropped here, so run the R script first.
EXCLUSIONS_CSV = OUT_DIR / "behavioral" / "trial_exclusions.csv"
FIG_DIR, STATS_DIR = OUT_DIR / "figures", OUT_DIR / "stats"

# Display geometry (eyelink_FamiliarityTime.py): 1920 px = 40.6 cm at 60 cm
PX_PER_CM = 1920 / 40.6
PX_PER_DEG = PX_PER_CM * 2 * 60 * np.tan(np.radians(0.5))

# Gaze = average of both eyes (gaze_signals()); one eye where the other is missing.
LAMBDA = 8                  # Engbert & Kliegl velocity threshold (x median-based SD)
# Additional microsaccade detection (alongside the main one, not instead of it):
# both eyes, a more lenient threshold, and a microsaccade counts only if the
# other eye moves at the same time (Engbert & Kliegl's binocular criterion).
# Recovers microsaccades hidden by noise in one eye (e.g. participant 003)
# without counting noise. Measures with this detection end in "_bino".
BINO_LAMBDA = 8
MIN_DUR_S = 0.009          # minimum event duration (paper: 9 ms)
MERGE_GAP_S = 0.010        # merge events closer than this
# Microsaccades up to 2 deg, as in Rosenzweig & Bonneh (2019); Kadosh & Bonneh
# (2022) used 0.08-1 deg (msRT < 0.6 deg) with > 1 deg fixation-inducing
# saccades. With a 2 deg microsaccade limit, fixation-inducing saccades start
# at 2 deg too, so no movement counts as both.
SACC_MIN_DEG = 2.0         # fixation-inducing saccade
MS_MIN_DEG, MS_MAX_DEG = 0.08, 2.0     # microsaccade (Rosenzweig & Bonneh 2019: 0.08-2 deg)
MSRT_MAX_DEG = 2.0         # microsaccades counted for msRT (all microsaccades)
MSRT_WIN_S = (0.00, 0.600)            # msRT window after fixation onset
BLINK_PAD_S = 0.100        # ignore events this close to a blink
FAMILIAR_CUTOFF = 50       # familiarity rating >= cutoff -> "familiar"
ANTICIPATION_MS = 80       # first saccades starting earlier are anticipatory

COLORS = {"unfamiliar": "#4C72B0", "familiar": "#DD8452"}
# group mean across participants (drawn on top of the per-participant means)
GROUP_COLORS = {"unfamiliar": "#08306B", "familiar": "#B22222"}

####################################################
# EYE-MOVEMENT DETECTION
####################################################

def velocity(x, y, dt):
    """Engbert & Kliegl 5-sample velocity (deg/s)."""
    vx = np.zeros_like(x); vy = np.zeros_like(y)
    vx[2:-2] = (x[4:] + x[3:-1] - x[1:-3] - x[:-4]) / (6 * dt)
    vy[2:-2] = (y[4:] + y[3:-1] - y[1:-3] - y[:-4]) / (6 * dt)
    return vx, vy


def detect_events(t, x, y, lam=None):
    """Saccades/microsaccades in one segment. x, y in deg (NaN = missing).
    lam = velocity threshold (default LAMBDA).
    Returns a DataFrame with onset, offset, amplitude (deg), peak velocity."""
    lam = LAMBDA if lam is None else lam
    dt = np.median(np.diff(t))
    vx, vy = velocity(x, y, dt)
    ok = np.isfinite(vx) & np.isfinite(vy)
    if ok.sum() < 50:
        return pd.DataFrame(columns=["onset", "offset", "amp", "peak_v"])
    sx = np.sqrt(np.nanmedian(vx[ok] ** 2) - np.nanmedian(vx[ok]) ** 2)
    sy = np.sqrt(np.nanmedian(vy[ok] ** 2) - np.nanmedian(vy[ok]) ** 2)
    above = ok & (((vx / (lam * sx)) ** 2 + (vy / (lam * sy)) ** 2) > 1)
    # runs of supra-threshold samples
    edges = np.diff(above.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(len(above))
    runs = []
    for s, e in zip(starts, ends):
        if runs and (t[s] - t[runs[-1][1] - 1]) < MERGE_GAP_S:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    rows = []
    for s, e in runs:
        if t[e - 1] - t[s] < MIN_DUR_S:
            continue
        seg_x, seg_y = x[s:e], y[s:e]
        if not (np.all(np.isfinite(seg_x)) and np.all(np.isfinite(seg_y))):
            continue
        amp = np.hypot(seg_x[-1] - seg_x[0], seg_y[-1] - seg_y[0])
        peak = np.nanmax(np.hypot(vx[s:e], vy[s:e]))
        rows.append(dict(onset=t[s], offset=t[e - 1], amp=amp, peak_v=peak,
                         x_end=seg_x[-1], y_end=seg_y[-1]))
    return pd.DataFrame(rows)

####################################################
# PER-SUBJECT EXTRACTION
####################################################

def image_key(value):
    return re.sub(r"\.\w+$", "", str(value).replace("\\", "/").rsplit("/", 1)[-1])


def read_trials(edf):
    msgs = [(t, m.decode(errors="replace").strip()) for t, m in edf["discrete"]["messages"]]
    trials, cur = [], None
    for t, m in msgs:
        if m.startswith("TRIAL_START"):
            parts = m.split(maxsplit=1)
            cur = dict(practice=False, timeout=False, label=parts[1] if len(parts) > 1 else "")
            trials.append(cur)
        elif cur is None:
            continue
        elif m == "PRACTICE_TRIAL":
            cur["practice"] = True
        elif m == "REPRODUCTION_TIMEOUT":
            # no response within 10 s: the experiment saves no behavioral row
            cur["timeout"] = True
        elif m.startswith("FACE_IMAGE"):
            cur["face_image"] = image_key(m.split(maxsplit=1)[1])
        elif m in ("FACE_ONSET", "FACE_OFFSET", "PRE_FACE_FIX_GAZE_OK"):
            cur[m] = t
    return [tr for tr in trials if not tr["practice"] and tr["label"].isdigit()
            and "FACE_ONSET" in tr and "FACE_OFFSET" in tr]


def gaze_signals(edf):
    """Gaze in degrees from both eyes.

    Returns (x, y) = average of the two eyes; where only one eye has valid
    data, that eye is used, shifted by half the median left-right offset
    (vergence) so the signal does not jump when it switches between the
    average and a single eye (a jump would be detected as a saccade).
    Also returns each eye separately (for the binocular detection), all
    blinks (either eye) and the share of samples by source."""
    names = edf["info"]["sample_fields"]
    S = edf["samples"]

    def one(eye):
        xp, yp = S[names.index(f"xpos_{eye}")], S[names.index(f"ypos_{eye}")]
        bad = ~np.isfinite(xp) | ~np.isfinite(yp) | (np.abs(xp) > 1e5) | (np.abs(yp) > 1e5)
        return np.where(bad, np.nan, xp) / PX_PER_DEG, np.where(bad, np.nan, yp) / PX_PER_DEG

    (xl, yl), (xr, yr) = one("left"), one("right")
    ok_l, ok_r = np.isfinite(xl), np.isfinite(xr)
    both = ok_l & ok_r
    dx = np.nanmedian((xl - xr)[both]) if both.any() else 0.0     # left - right offset
    dy = np.nanmedian((yl - yr)[both]) if both.any() else 0.0
    x = np.where(both, (xl + xr) / 2, np.where(ok_l, xl - dx / 2, np.where(ok_r, xr + dx / 2, np.nan)))
    y = np.where(both, (yl + yr) / 2, np.where(ok_l, yl - dy / 2, np.where(ok_r, yr + dy / 2, np.nan)))
    share = dict(both=both.mean(), left_only=(ok_l & ~ok_r).mean(), right_only=(ok_r & ~ok_l).mean(),
                 none=(~ok_l & ~ok_r).mean())
    return x, y, (xl, yl), (xr, yr), edf["discrete"]["blinks"], share


def extract_subject(subject, edf_path):
    edf = eyelinkio.read_edf(str(edf_path))
    T = edf["times"]
    # main signal = average of both eyes; each eye separately for the binocular detection
    x, y, (xl, yl), (xr, yr), blinks_all, eye_share = gaze_signals(edf)
    blinks = blinks_all                         # a blink in either eye distorts the average

    def clean(ev, on, off, blink_set):
        if not len(ev):
            return ev
        near = np.zeros(len(ev), bool)
        for b in blink_set:
            near |= (ev["offset"] > b["stime"] - BLINK_PAD_S) & (ev["onset"] < b["etime"] + BLINK_PAD_S)
        ev = ev[~near]
        return ev[(ev["onset"] >= on - 0.02) & (ev["onset"] < off)].reset_index(drop=True)

    trial_rows, epoch_rows, event_rows, dropped = [], [], [], []
    for tr in read_trials(edf):
        i = int(tr["label"])                      # trial_num, as in the behavioral CSV
        if tr["timeout"]:
            dropped.append((i, "reproduction timeout"))
            continue
        on, off = tr["FACE_ONSET"], tr["FACE_OFFSET"]
        seg = (T >= on - 0.3) & (T <= off + 0.05)       # pad for velocity edges
        ev = detect_events(T[seg], x[seg], y[seg])
        if len(ev):
            near_blink = np.zeros(len(ev), bool)
            for b in blinks:
                near_blink |= (ev["offset"] > b["stime"] - BLINK_PAD_S) & \
                              (ev["onset"] < b["etime"] + BLINK_PAD_S)
            ev = ev[~near_blink]
            ev = ev[(ev["onset"] >= on - 0.02) & (ev["onset"] < off)].reset_index(drop=True)
        blink_in_face = any((b["etime"] > on) and (b["stime"] < off) for b in blinks)
        sacc = ev[ev["amp"] >= SACC_MIN_DEG] if len(ev) else ev
        ms = ev[(ev["amp"] >= MS_MIN_DEG) & (ev["amp"] < MS_MAX_DEG)] if len(ev) else ev

        # binocular microsaccades (lenient threshold, both eyes must move together)
        # each eye separately; a microsaccade counts only if both eyes move at
        # the same time; its amplitude = mean of the two eyes' amplitudes
        ev_b = clean(detect_events(T[seg], xl[seg], yl[seg], lam=BINO_LAMBDA), on, off, blinks_all)
        ev_o = clean(detect_events(T[seg], xr[seg], yr[seg], lam=BINO_LAMBDA), on, off, blinks_all)
        ms_bino = ev_b.iloc[0:0] if len(ev_b) else ev_b
        if len(ev_b) and len(ev_o):
            keep, amps = [], []
            for m in ev_b.itertuples():
                o = ev_o[(ev_o["onset"] <= m.offset) & (ev_o["offset"] >= m.onset)]
                if len(o):
                    keep.append(m.Index); amps.append((m.amp + o["amp"].iloc[0]) / 2)
            ms_bino = ev_b.loc[keep].assign(amp=amps)
            ms_bino = ms_bino[(ms_bino["amp"] >= MS_MIN_DEG) & (ms_bino["amp"] < MS_MAX_DEG)]
        for _, e in ev.iterrows():
            event_rows.append(dict(subject=subject, trial=i, onset_rel=e.onset - on,
                                   amp=e.amp, peak_v=e.peak_v))

        first = sacc.iloc[0] if len(sacc) else None
        anticipatory = first is not None and (first.onset - on) * 1000 < ANTICIPATION_MS
        trial_rows.append(dict(
            subject=subject, trial=i, face_image=tr["face_image"],
            face_dur_s=off - on, blink_in_face=blink_in_face,
            first_sacc_latency_ms=(first.onset - on) * 1000 if first is not None and not anticipatory else np.nan,
            anticipatory_first_sacc=anticipatory,
            first_sacc_amp=first.amp if first is not None else np.nan,
            n_saccades=len(sacc), sacc_rate_hz=len(sacc) / (off - on),
            mean_sacc_amp=sacc["amp"].mean() if len(sacc) else np.nan,
            n_ms=len(ms), ms_rate_hz=len(ms) / (off - on),
            n_ms_bino=len(ms_bino), ms_rate_bino_hz=len(ms_bino) / (off - on),
            gaze_both_eyes=eye_share["both"]))

        # Fixation-related epochs: landing of each > SACC_MIN_DEG saccade on the face
        sacc = sacc.reset_index(drop=True)
        for k, s in sacc.iterrows():
            land = s.offset
            if land >= off:
                continue
            next_on = sacc.iloc[k + 1].onset if k + 1 < len(sacc) else np.inf
            win_end = min(land + MSRT_WIN_S[1], next_on, off)
            after = ms[(ms["onset"] > land) & (ms["onset"] < min(next_on, off))]
            cand = after[(after["onset"] >= land + MSRT_WIN_S[0]) & (after["onset"] < win_end)
                         & (after["amp"] < MSRT_MAX_DEG)]
            msrt = (cand.iloc[0].onset - land) * 1000 if len(cand) else np.nan
            # censored if the window was cut by face offset / next saccade before
            # a microsaccade appeared
            censored = np.isnan(msrt) and win_end < land + MSRT_WIN_S[1]
            ms_free = len(after) == 0
            fix_dur = (next_on - land) * 1000 if np.isfinite(next_on) and next_on < off else np.nan
            # the same with the binocular microsaccades
            after_b = ms_bino[(ms_bino["onset"] > land) & (ms_bino["onset"] < min(next_on, off))] \
                if len(ms_bino) else ms_bino
            cand_b = after_b[(after_b["onset"] >= land + MSRT_WIN_S[0]) & (after_b["onset"] < win_end)
                             & (after_b["amp"] < MSRT_MAX_DEG)] if len(after_b) else after_b
            msrt_b = (cand_b.iloc[0].onset - land) * 1000 if len(cand_b) else np.nan
            epoch_rows.append(dict(
                subject=subject, trial=i, face_image=tr["face_image"], serial=k + 1,
                landing_rel_ms=(land - on) * 1000, sacc_amp=s.amp,
                msrt_ms=msrt, msrt_censored=censored,
                fix_dur_ms=fix_dur if ms_free else np.nan, ms_free=ms_free,
                msrt_bino_ms=msrt_b,
                msrt_bino_censored=np.isnan(msrt_b) and win_end < land + MSRT_WIN_S[1],
                fix_dur_bino_ms=fix_dur if len(after_b) == 0 else np.nan))
    return pd.DataFrame(trial_rows), pd.DataFrame(epoch_rows), pd.DataFrame(event_rows), dropped


def load_exclusions():
    """{subject: {trial: reason}} for discarded trials and {subject: reason}
    for discarded participants, from EXCLUSIONS_CSV."""
    if not EXCLUSIONS_CSV.exists():
        raise SystemExit(f"{EXCLUSIONS_CSV} not found -- run behavioral_familiarity_lmm.R first "
                         "(it decides which trials and participants are excluded).")
    ex = pd.read_csv(EXCLUSIONS_CSV, dtype={"participant": str})
    ex["participant"] = ex["participant"].str.zfill(3)
    ex = ex[ex["excluded"].astype(str).str.upper() == "TRUE"]
    bad_subjects = ex[ex["subject_excluded"].astype(str).str.upper() == "TRUE"] \
        .groupby("participant")["reason"].first().to_dict()
    bad_trials = {s: dict(zip(g["trial_num"].astype(int), g["reason"]))
                  for s, g in ex[~ex["participant"].isin(bad_subjects)].groupby("participant")}
    return bad_trials, bad_subjects


def all_ratings(subjects):
    """Every face's rating from each participant's ratings CSV -- including
    faces whose trial was discarded, since the rating does not depend on the
    reproduction."""
    nat = {"IL": "israeli", "UK": "uk", "N": "neutral"}
    rows = []
    for sub in subjects:
        rat = pd.read_csv(sub["ratings"])
        rat["face_image"] = rat["face_image"].map(image_key)
        rat["subject"] = sub["id"]
        rows.append(rat)
    R = pd.concat(rows, ignore_index=True)
    R["face_nationality"] = R["face_image"].str.split("_").str[0].map(nat)
    R["familiar"] = np.where(R["familiarity_rating"] >= FAMILIAR_CUTOFF, "familiar", "unfamiliar")
    return R[["subject", "face_image", "face_nationality", "familiarity_rating", "familiar"]]


def merge_behavior(subject, trials, behavioral_path, ratings_path):
    beh = pd.read_csv(behavioral_path)
    rat = pd.read_csv(ratings_path)
    beh = beh[beh["is_practice"].astype(str).str.lower() != "true"].reset_index(drop=True)
    beh["face_image"] = beh["face_image"].map(image_key)
    beh["trial"] = beh["trial_num"].astype(int)
    rat["face_image"] = rat["face_image"].map(image_key)
    # Match EDF and behavioral trials by trial number. EDF trials without a
    # behavioral row (e.g. no response) are dropped together with their eye data.
    no_beh = sorted(set(trials["trial"]) - set(beh["trial"]))
    no_edf = sorted(set(beh["trial"]) - set(trials["trial"]))
    if no_edf:
        print(f"  {subject}: behavioral trials missing from the EDF (left out): {no_edf}")
    df = trials.merge(beh[["trial", "face_image", "face_nationality", "face_duration_planned_s",
                           "reproduced_duration_s", "reproduction_error_s"]],
                      on="trial", how="inner", suffixes=("", "_beh"))
    mismatch = df.loc[df["face_image"] != df["face_image_beh"], "trial"].tolist()
    assert not mismatch, f"{subject}: trials {mismatch} show a different face in the EDF and CSV"
    df = df.drop(columns="face_image_beh")
    df = df.merge(rat[["face_image", "familiarity_rating", "attractiveness_rating"]],
                  on="face_image", how="left")
    df["familiar"] = np.where(df["familiarity_rating"] >= FAMILIAR_CUTOFF, "familiar", "unfamiliar")
    df["rel_error"] = df["reproduction_error_s"] / df["face_duration_planned_s"]
    return df, no_beh

####################################################
# STATISTICS
####################################################

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def familiarity_tests(df, measure, label, extra=""):
    """Familiar vs unfamiliar per subject (Welch t, Mann-Whitney) and pooled
    regression with subject as a fixed effect (2 subjects: too few for a
    random effect), binary and continuous familiarity."""
    rows = []
    d = df.dropna(subset=[measure])
    for subj, g in list(d.groupby("subject")) + [("both", d)]:
        f = g.loc[g["familiar"] == "familiar", measure]
        u = g.loc[g["familiar"] == "unfamiliar", measure]
        if len(f) < 3 or len(u) < 3:
            continue
        row = dict(measure=label, subject=subj, n_familiar=len(f), n_unfamiliar=len(u),
                   mean_familiar=f.mean(), mean_unfamiliar=u.mean(),
                   diff=f.mean() - u.mean(), cohens_d=cohens_d(f, u))
        if subj != "both":
            # tests are undefined when a participant's values don't vary
            # (e.g. exactly one saccade on every trial) -> left empty
            if f.std() > 0 or u.std() > 0:
                # Welch's test is valid when one group is constant, but scipy
                # warns about "precision loss" for that group -- silenced here
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    row["welch_p"] = stats.ttest_ind(f, u, equal_var=False).pvalue
                row["mannwhitney_p"] = stats.mannwhitneyu(f, u).pvalue
            if g[measure].nunique() > 1:
                row["spearman_rho_rating"], row["spearman_p_rating"] = \
                    stats.spearmanr(g["familiarity_rating"], g[measure])
        else:
            m1 = smf.ols(f"{measure} ~ C(familiar, Treatment('unfamiliar')) + C(subject){extra}", g).fit()
            term = [k for k in m1.params.index if k.startswith("C(familiar")][0]
            row["model_beta_familiar"], row["model_p_familiar"] = m1.params[term], m1.pvalues[term]
            m2 = smf.ols(f"{measure} ~ familiarity_z + C(subject){extra}", g).fit()
            row["model_beta_rating_z"], row["model_p_rating_z"] = m2.params["familiarity_z"], m2.pvalues["familiarity_z"]
            # familiarity is nearly confounded with face category (Israeli faces
            # familiar, neutral unfamiliar): same model controlling for category
            m3 = smf.ols(f"{measure} ~ familiarity_z + C(face_nationality) + C(subject){extra}", g).fit()
            row["model_beta_rating_z_ctrl_category"] = m3.params["familiarity_z"]
            row["model_p_rating_z_ctrl_category"] = m3.pvalues["familiarity_z"]
        rows.append(row)
    return rows

####################################################
# FIGURES
####################################################

def paired_plot(ax, df, measure, ylabel, title):
    """Paper-style: per-subject means, familiar vs unfamiliar, with trial dots."""
    subjects = sorted(df["subject"].unique())
    spread = np.linspace(-0.3, 0.3, len(subjects)) if len(subjects) > 1 else [0.0]
    for j, subj in enumerate(subjects):
        g = df[df["subject"] == subj].dropna(subset=[measure])
        means = []
        for k, cond in enumerate(["unfamiliar", "familiar"]):
            v = g.loc[g["familiar"] == cond, measure]
            xpos = k + spread[j]
            ax.scatter(np.full(len(v), xpos) + np.random.uniform(-0.05, 0.05, len(v)), v,
                       s=8, alpha=0.25, color=COLORS[cond])
            m, se = v.mean(), v.std(ddof=1) / np.sqrt(max(len(v), 1))
            ax.errorbar(xpos, m, yerr=se, fmt="o", color="k", ms=6, capsize=3)
            means.append((xpos, m))
        ax.plot(*zip(*means), "k-", lw=1)
        ax.text(means[-1][0] + 0.05, means[-1][1], f"S{subj}", fontsize=8, va="center")
    add_group_means(ax, df, measure)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["unfamiliar", "familiar"])
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def add_group_means(ax, df, measure, x_offset=0.45):
    """Mean across participants for each group (participant means averaged,
    each participant counted once), +- SE across participants, as large
    diamonds in GROUP_COLORS, just right of each group's participants."""
    pm = df.dropna(subset=[measure]).groupby(["subject", "familiar"])[measure].mean().unstack()
    pts = []
    for k, cond in enumerate(["unfamiliar", "familiar"]):
        if cond not in pm:
            continue
        v = pm[cond].dropna()
        if not len(v):
            continue
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
        ax.errorbar(k + x_offset, v.mean(), yerr=se, fmt="D", ms=11, color=GROUP_COLORS[cond],
                    mec="white", mew=1.2, capsize=4, lw=2, zorder=5,
                    label=f"{cond} mean (N = {len(v)})")
        pts.append((k + x_offset, v.mean()))
    if len(pts) == 2:
        ax.plot(*zip(*pts), "-", color="grey", lw=2, zorder=4)
    ax.set_xlim(-0.55, 1.75)
    ax.legend(frameon=False, fontsize=7, loc="best")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True); STATS_DIR.mkdir(parents=True, exist_ok=True)
    trials, epochs, events = [], [], []
    dropped_rows = []
    subjects, skipped = find_subjects()
    bad_trials, bad_subjects = load_exclusions()
    print("Participants:", ", ".join(s["id"] for s in subjects))
    if skipped:
        print("Skipped:", "; ".join(skipped))
    for s, why in bad_subjects.items():
        print(f"  {s}: {why} -- not analyzed")
        dropped_rows.append(dict(subject=s, trial="all", reason=why))
    subjects = [sub for sub in subjects if sub["id"] not in bad_subjects]
    for sub in subjects:
        s = sub["id"]
        t, e, ev, dropped = extract_subject(s, sub["edf"])
        t, no_beh = merge_behavior(s, t, sub["behavioral"], sub["ratings"])
        dropped += [(n, "no behavioral row") for n in no_beh]
        excl = bad_trials.get(s, {})
        dropped += [(n, f"behavioral exclusion: {excl[n]}") for n in sorted(excl) if n in set(t["trial"])]
        t = t[~t["trial"].isin(excl)].reset_index(drop=True)
        for n, why in dropped:
            print(f"  {s}: trial {n} dropped ({why}) -- its eye data is not used")
            dropped_rows.append(dict(subject=s, trial=n, reason=why))
        e = e.merge(t[["trial", "familiar", "familiarity_rating", "face_duration_planned_s",
                       "face_nationality"]], on="trial")          # drops epochs of dropped trials
        e["subject"] = s
        ev = ev[ev["trial"].isin(t["trial"])]
        trials.append(t); epochs.append(e); events.append(ev)
    T = pd.concat(trials, ignore_index=True)
    E = pd.concat(epochs, ignore_index=True)
    EV = pd.concat(events, ignore_index=True)
    T["familiarity_z"] = T.groupby("subject")["familiarity_rating"].transform(lambda v: (v - v.mean()) / v.std())
    E["familiarity_z"] = E.groupby("subject")["familiarity_rating"].transform(lambda v: (v - v.mean()) / v.std())
    T["duration_c"] = T["face_duration_planned_s"] - T["face_duration_planned_s"].mean()
    E["duration_c"] = E["face_duration_planned_s"] - E["face_duration_planned_s"].mean()
    E["first_fix"] = E["serial"] == 1
    pd.DataFrame(dropped_rows, columns=["subject", "trial", "reason"]).to_csv(
        STATS_DIR / "dropped_trials.csv", index=False)
    T.to_csv(STATS_DIR / "trials.csv", index=False)
    E.to_csv(STATS_DIR / "fixation_epochs.csv", index=False)

    # ---------- data summary ----------
    summ = T.groupby("subject").agg(
        trials=("trial", "size"), familiar_trials=("familiar", lambda v: (v == "familiar").sum()),
        blink_trials=("blink_in_face", "sum"),
        first_sacc_found=("first_sacc_latency_ms", lambda v: v.notna().sum()),
        anticipatory=("anticipatory_first_sacc", "sum"),
        mean_first_sacc_amp=("first_sacc_amp", "mean"),
        mean_saccades=("n_saccades", "mean"), mean_ms=("n_ms", "mean"),
        mean_ms_bino=("n_ms_bino", "mean"),
        share_both_eyes=("gaze_both_eyes", "first"),
        mean_error_s=("reproduction_error_s", "mean"))
    ep = E.groupby("subject").agg(epochs=("trial", "size"),
                                  msrt_epochs=("msrt_ms", lambda v: v.notna().sum()),
                                  censored=("msrt_censored", "sum"),
                                  msrt_bino_epochs=("msrt_bino_ms", lambda v: v.notna().sum()),
                                  msfree_fix_dur=("fix_dur_ms", lambda v: v.notna().sum()))
    summ = summ.join(ep)
    summ.to_csv(STATS_DIR / "data_summary.csv")
    xt = pd.crosstab([T["subject"], T["face_nationality"]], T["familiar"])
    xt.to_csv(STATS_DIR / "familiar_by_nationality.csv")

    # ---------- familiarity tests (paper measures + task-specific) ----------
    ext = " + duration_c"
    rows = []
    rows += familiarity_tests(E, "msrt_ms", f"msRT, {MSRT_WIN_S[0]*1000:g}-{MSRT_WIN_S[1]*1000:g} ms after >{SACC_MIN_DEG:g} deg landing (ms)", ext)
    rows += familiarity_tests(E[~E["first_fix"]], "msrt_ms", "msRT, later fixations only (ms)", ext)
    rows += familiarity_tests(E, "fix_dur_ms", "MS-free fixation duration (ms)", ext)
    rows += familiarity_tests(E, "sacc_amp", f"Saccade size, >{SACC_MIN_DEG:g} deg (deg)", ext)
    rows += familiarity_tests(E[~E["first_fix"]], "sacc_amp",
                              "Saccade size on the face, 2nd saccade on (deg)", ext)
    rows += familiarity_tests(T, "first_sacc_latency_ms", "First-saccade latency to face (ms)", ext)
    rows += familiarity_tests(T, "n_saccades", f"Saccades (>{SACC_MIN_DEG:g} deg) per trial", ext)
    rows += familiarity_tests(T, "sacc_rate_hz", "Saccade rate on face (Hz)", ext)
    rows += familiarity_tests(T, "ms_rate_hz", "Microsaccade rate on face (Hz)", ext)
    # additional binocular microsaccade detection
    rows += familiarity_tests(E, "msrt_bino_ms", "msRT, binocular detection (ms)", ext)
    rows += familiarity_tests(E, "fix_dur_bino_ms", "MS-free fixation duration, binocular (ms)", ext)
    rows += familiarity_tests(T, "ms_rate_bino_hz", "Microsaccade rate on face, binocular (Hz)", ext)
    fam = pd.DataFrame(rows)
    fam.to_csv(STATS_DIR / "familiarity_effects.csv", index=False)

    # ---------- adaptation across successive saccades (paper Fig. 4) ----------
    adapt = []
    for meas in ("msrt_ms", "sacc_amp"):
        d = E.dropna(subset=[meas])
        m = smf.ols(f"{meas} ~ serial * C(familiar, Treatment('unfamiliar')) + C(subject) + duration_c", d).fit()
        for k in m.params.index:
            if "serial" in k:
                adapt.append(dict(measure=meas, term=k, beta=m.params[k], p=m.pvalues[k]))
    pd.DataFrame(adapt).to_csv(STATS_DIR / "serial_saccade_adaptation.csv", index=False)

    # ---------- mean saccade size per saccade in the trial, familiar vs unfamiliar ----------
    # serial 1 = the saccade from the fixation cross to the face (~5-6 deg);
    # 2+ = saccades made on the face. Saccades >= SACC_MAX_SERIAL are pooled.
    SACC_MAX_SERIAL = 4
    E["serial_bin"] = np.minimum(E["serial"], SACC_MAX_SERIAL)
    size_rows = []
    for sb, g in E.groupby("serial_bin"):
        label = f"{int(sb)}" + ("+" if sb == SACC_MAX_SERIAL else "")
        for subj, gs in list(g.groupby("subject")) + [("all", g)]:
            row = dict(saccade=label, subject=subj)
            for cond in ("unfamiliar", "familiar"):
                v = gs.loc[gs["familiar"] == cond, "sacc_amp"]
                row.update({f"n_{cond}": len(v), f"mean_{cond}_deg": v.mean(),
                            f"se_{cond}_deg": v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan})
            row["diff_deg"] = row["mean_familiar_deg"] - row["mean_unfamiliar_deg"]
            if subj == "all" and row["n_familiar"] >= 3 and row["n_unfamiliar"] >= 3:
                form = "sacc_amp ~ C(familiar, Treatment('unfamiliar')) + duration_c"
                if gs["subject"].nunique() > 1:
                    form += " + C(subject)"
                m = smf.ols(form, gs).fit()
                term = [k for k in m.params.index if k.startswith("C(familiar")][0]
                row["model_diff_deg"], row["model_p"] = m.params[term], m.pvalues[term]
            size_rows.append(row)
    sacc_size = pd.DataFrame(size_rows)
    sacc_size.to_csv(STATS_DIR / "saccade_size_by_serial.csv", index=False)

    # ---------- familiarity -> reproduction error (previous experiment) ----------
    beh = []
    for subj, g in list(T.groupby("subject")) + [("both", T)]:
        r, p = stats.pearsonr(g["familiarity_rating"], g["reproduction_error_s"])
        rho, ps = stats.spearmanr(g["familiarity_rating"], g["reproduction_error_s"])
        row = dict(subject=subj, n=len(g), pearson_r=r, pearson_p=p, spearman_rho=rho, spearman_p=ps)
        form = "reproduction_error_s ~ familiarity_z + duration_c" + (" + C(subject)" if subj == "both" else "")
        m = smf.ols(form, g).fit()
        row.update(beta_familiarity=m.params["familiarity_z"], p_familiarity=m.pvalues["familiarity_z"],
                   beta_duration=m.params["duration_c"], p_duration=m.pvalues["duration_c"])
        m2 = smf.ols(form + " + attractiveness_rating", g).fit()
        row.update(beta_familiarity_ctrl_attract=m2.params["familiarity_z"],
                   p_familiarity_ctrl_attract=m2.pvalues["familiarity_z"])
        m3 = smf.ols(form.replace("familiarity_z", "familiarity_z * duration_c"), g).fit()
        row.update(beta_fam_x_duration=m3.params["familiarity_z:duration_c"],
                   p_fam_x_duration=m3.pvalues["familiarity_z:duration_c"])
        beh.append(row)
    pd.DataFrame(beh).to_csv(STATS_DIR / "familiarity_vs_error.csv", index=False)

    # Do eye movements carry / moderate the familiarity effect on error?
    # z within participant; a participant whose values never vary (e.g. exactly
    # one saccade on every trial) can't be standardized and is left out of
    # that model (dropping the rows BEFORE fitting, so no empty participant
    # column is left in the design matrix).
    zscore = lambda v: (v - v.mean()) / v.std() if v.std() > 0 else v * np.nan
    T["sacc_z"] = T.groupby("subject")["n_saccades"].transform(zscore)
    T["lat_z"] = T.groupby("subject")["first_sacc_latency_ms"].transform(zscore)
    mod = []
    for name, form, z in [
        ("moderation: saccades", "reproduction_error_s ~ familiarity_z * sacc_z + duration_c + C(subject)", "sacc_z"),
        ("moderation: first-saccade latency", "reproduction_error_s ~ familiarity_z * lat_z + duration_c + C(subject)", "lat_z"),
    ]:
        d = T.dropna(subset=["reproduction_error_s", "familiarity_z", "duration_c", z])
        left_out = sorted(set(T["subject"]) - set(d["subject"]))
        m = smf.ols(form, d).fit()
        for k in m.params.index:
            if "familiarity" in k or k in ("sacc_z", "lat_z"):
                mod.append(dict(model=name, term=k, beta=m.params[k], p=m.pvalues[k],
                                n_participants=d["subject"].nunique(),
                                left_out_no_variance=", ".join(f"S{s}" for s in left_out)))
    pd.DataFrame(mod).to_csv(STATS_DIR / "eye_movements_and_error.csv", index=False)

    # ---------- the same, separately for each face duration ----------
    dur_rows, dur_err = [], []
    for dur in sorted(T["face_duration_planned_s"].unique()):
        Td, Ed = T[T["face_duration_planned_s"] == dur], E[E["face_duration_planned_s"] == dur]
        part = []
        part += familiarity_tests(Ed, "msrt_ms", f"msRT, {MSRT_WIN_S[0]*1000:g}-{MSRT_WIN_S[1]*1000:g} ms after >{SACC_MIN_DEG:g} deg landing (ms)")
        part += familiarity_tests(Ed, "fix_dur_ms", "MS-free fixation duration (ms)")
        part += familiarity_tests(Ed, "sacc_amp", f"Saccade size, >{SACC_MIN_DEG:g} deg (deg)")
        part += familiarity_tests(Td, "first_sacc_latency_ms", "First-saccade latency to face (ms)")
        part += familiarity_tests(Td, "n_saccades", f"Saccades (>{SACC_MIN_DEG:g} deg) per trial")
        part += familiarity_tests(Td, "ms_rate_hz", "Microsaccade rate on face (Hz)")
        part += familiarity_tests(Ed, "msrt_bino_ms", "msRT, binocular detection (ms)")
        part += familiarity_tests(Td, "ms_rate_bino_hz", "Microsaccade rate on face, binocular (Hz)")
        part += familiarity_tests(Td, "reproduction_error_s", "Reproduction error (s)")
        for r in part:
            r["duration_s"] = dur
        dur_rows += part
        for subj, g in list(Td.groupby("subject")) + [("both", Td)]:
            r, p = stats.pearsonr(g["familiarity_rating"], g["reproduction_error_s"])
            form = "reproduction_error_s ~ familiarity_z" + (" + C(subject)" if subj == "both" else "")
            m = smf.ols(form, g).fit()
            m2 = smf.ols(form + " + C(face_nationality)", g).fit()
            dur_err.append(dict(duration_s=dur, subject=subj, n=len(g),
                                mean_error_s=g["reproduction_error_s"].mean(),
                                pearson_r=r, pearson_p=p,
                                beta_familiarity=m.params["familiarity_z"], p_familiarity=m.pvalues["familiarity_z"],
                                p_familiarity_ctrl_category=m2.pvalues["familiarity_z"]))
    by_dur = pd.DataFrame(dur_rows)
    by_dur = by_dur[["duration_s"] + [c for c in by_dur.columns if c != "duration_s"]]
    by_dur.to_csv(STATS_DIR / "familiarity_effects_by_duration.csv", index=False)
    pd.DataFrame(dur_err).to_csv(STATS_DIR / "familiarity_vs_error_by_duration.csv", index=False)

    # ---------- how many faces are rated familiar ----------
    # all ratings of the analyzed participants, not only faces with a kept trial
    R = all_ratings(subjects)
    rating_summary = R.groupby("subject").agg(
        faces=("face_image", "size"),
        familiar=("familiar", lambda v: (v == "familiar").sum()),
        rated_0_10=("familiarity_rating", lambda v: (v <= 10).sum()),
        rated_11_89=("familiarity_rating", lambda v: ((v > 10) & (v < 90)).sum()),
        rated_90_100=("familiarity_rating", lambda v: (v >= 90).sum()),
        median_rating=("familiarity_rating", "median"))
    by_cat = pd.crosstab(R["subject"], [R["face_nationality"], R["familiar"]])
    per_face = R.groupby(["face_image", "face_nationality"]).agg(
        participants=("subject", "size"),
        rated_familiar=("familiar", lambda v: (v == "familiar").sum()),
        mean_rating=("familiarity_rating", "mean")).reset_index()
    per_face["prop_familiar"] = per_face["rated_familiar"] / per_face["participants"]
    rating_summary.to_csv(STATS_DIR / "familiarity_rating_summary.csv")
    by_cat.to_csv(STATS_DIR / "familiarity_rating_by_category.csv")
    per_face.sort_values(["face_nationality", "face_image"]).to_csv(
        STATS_DIR / "familiarity_per_face.csv", index=False)

    subjects_sorted = sorted(R["subject"].unique())
    cats = ["israeli", "uk", "neutral"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    bins = np.arange(0, 105, 5)
    for k, subj in enumerate(subjects_sorted):
        v = R.loc[R["subject"] == subj, "familiarity_rating"]
        axes[0].hist(v, bins=bins, histtype="step", lw=2, label=f"S{subj}")
    axes[0].axvline(FAMILIAR_CUTOFF, color="k", ls="--", lw=1)
    axes[0].set_xlabel("familiarity rating"); axes[0].set_ylabel("number of faces")
    axes[0].set_title("Distribution of familiarity ratings", fontsize=10); axes[0].legend(frameon=False, fontsize=8)
    w = 0.8 / len(subjects_sorted)
    for k, subj in enumerate(subjects_sorted):
        g = R[R["subject"] == subj]
        n = [((g["face_nationality"] == c) & (g["familiar"] == "familiar")).sum() for c in cats]
        tot = [(g["face_nationality"] == c).sum() for c in cats]
        bars = axes[1].bar(np.arange(3) + (k - (len(subjects_sorted) - 1) / 2) * w, n, w, label=f"S{subj}")
        for bar, nn, tt in zip(bars, n, tot):
            axes[1].text(bar.get_x() + bar.get_width() / 2, nn + 0.5, f"{nn}/{tt}", ha="center", fontsize=8)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(cats)
    axes[1].set_ylabel(f"faces rated familiar (≥{FAMILIAR_CUTOFF})")
    axes[1].set_title("Familiar faces per category", fontsize=10); axes[1].legend(frameon=False, fontsize=8)
    for c, col in zip(cats, ["#DD8452", "#8172B3", "#4C72B0"]):
        g = per_face[per_face["face_nationality"] == c].sort_values("mean_rating")
        axes[2].scatter(np.arange(len(g)) + cats.index(c) * (len(g) + 3), g["mean_rating"],
                        s=12 + 25 * g["rated_familiar"], color=col, label=c, alpha=0.8)
    axes[2].axhline(FAMILIAR_CUTOFF, color="k", ls="--", lw=1)
    axes[2].set_xticks([]); axes[2].set_ylabel("mean rating across participants")
    axes[2].set_title("Each face, sorted within category\n(dot size = participants rating it familiar)", fontsize=10)
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "6_familiarity_ratings.png", dpi=150); plt.close(fig)

    # ---------- figures ----------
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    paired_plot(axes[0, 0], E, "msrt_ms", "msRT (ms)", f"msRT after >{SACC_MIN_DEG:g}° saccade landing\n(paper: familiar shorter)")
    paired_plot(axes[0, 1], E, "fix_dur_ms", "fixation duration (ms)", "MS-free fixation duration\n(paper: familiar shorter, n.s.)")
    paired_plot(axes[0, 2], E, "sacc_amp", "saccade size (deg)", f"Saccade size >{SACC_MIN_DEG:g}°\n(paper: no difference)")
    paired_plot(axes[1, 0], T, "first_sacc_latency_ms", "latency (ms)", "First saccade from cross to face")
    paired_plot(axes[1, 1], T, "n_saccades", "saccades per trial", f"Saccades (>{SACC_MIN_DEG:g}°) while the face is shown")
    paired_plot(axes[1, 2], T, "ms_rate_hz", "microsaccades / s", "Microsaccade rate on the face")
    fig.suptitle("Familiar vs unfamiliar faces (familiarity rating ≥ 50) — mean ± SE per participant", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG_DIR / "1_oculomotor_familiarity.png", dpi=150); plt.close(fig)

    subjects_all = sorted(T["subject"].unique())
    n_pan = len(subjects_all) + 1                                  # one per participant + group
    n_col = min(n_pan, 4)
    fig, axes = plt.subplots(int(np.ceil(n_pan / n_col)), n_col, figsize=(4.7 * n_col, 4.2 * np.ceil(n_pan / n_col)),
                             squeeze=False)
    axes = axes.ravel()
    for ax in axes[n_pan:]:
        ax.set_visible(False)
    for j, subj in enumerate(subjects_all):
        g = T[T["subject"] == subj]
        axes[j].scatter(g["familiarity_rating"], g["reproduction_error_s"],
                        c=np.where(g["face_duration_planned_s"] > 1, "#55A868", "#C44E52"), s=14, alpha=0.7)
        b = np.polyfit(g["familiarity_rating"], g["reproduction_error_s"], 1)
        xs = np.linspace(0, 100, 50); axes[j].plot(xs, np.polyval(b, xs), "k-")
        r, p = stats.pearsonr(g["familiarity_rating"], g["reproduction_error_s"])
        axes[j].axhline(0, color="gray", lw=0.6)
        axes[j].set_title(f"S{subj}: r = {r:.2f}, p = {p:.3f}", fontsize=10)
        axes[j].set_xlabel("familiarity rating"); axes[j].set_ylabel("reproduction error (s)")
    paired_plot(axes[n_pan - 1], T, "reproduction_error_s", "reproduction error (s)",
                "Error by familiarity\n(previous experiment: familiar smaller)")
    axes[0].scatter([], [], c="#C44E52", label="0.8 s"); axes[0].scatter([], [], c="#55A868", label="1.6 s")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "2_familiarity_vs_error.png", dpi=150); plt.close(fig)

    durs = sorted(T["face_duration_planned_s"].unique())
    panels = [(E, "msrt_ms", "msRT (ms)"), (T, "first_sacc_latency_ms", "first-saccade latency (ms)"),
              (T, "n_saccades", "saccades per trial"), (T, "reproduction_error_s", "reproduction error (s)")]
    fig, axes = plt.subplots(len(durs), len(panels), figsize=(4.2 * len(panels), 3.8 * len(durs)), squeeze=False)
    for i, dur in enumerate(durs):
        for j, (D, meas, lab) in enumerate(panels):
            paired_plot(axes[i, j], D[D["face_duration_planned_s"] == dur], meas, lab, f"{lab} — {dur:g} s faces")
    fig.suptitle("Familiar vs unfamiliar, separately for each face duration — mean ± SE per participant", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG_DIR / "5_by_duration.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, meas, lab in ((axes[0], "msrt_ms", "msRT (ms)"), (axes[1], "sacc_amp", "saccade size (deg)")):
        for cond in ("unfamiliar", "familiar"):
            g = E[(E["familiar"] == cond) & (E["serial"] <= 4)].groupby("serial")[meas]
            ax.errorbar(g.mean().index + (0.05 if cond == "familiar" else -0.05), g.mean(),
                        yerr=g.std() / np.sqrt(g.count()), marker="o", color=COLORS[cond], label=cond, capsize=3)
        ax.set_xlabel("serial saccade number in trial"); ax.set_ylabel(lab)
        ax.set_xticks([1, 2, 3, 4]); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Across successive saccades (paper Fig. 4: adaptation for familiar)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "3_serial_saccades.png", dpi=150); plt.close(fig)

    # mean saccade size per saccade, familiar vs unfamiliar (all participants)
    allp = sacc_size[sacc_size["subject"] == "all"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.4, 1]})
    xs = np.arange(len(allp)); w = 0.36
    for k, cond in enumerate(("unfamiliar", "familiar")):
        xpos = xs + (k - 0.5) * w
        axes[0].bar(xpos, allp[f"mean_{cond}_deg"], w, yerr=allp[f"se_{cond}_deg"],
                    color=COLORS[cond], alpha=0.85, capsize=3, label=cond)
        for x, n, m in zip(xpos, allp[f"n_{cond}"], allp[f"mean_{cond}_deg"]):
            axes[0].text(x, (m if np.isfinite(m) else 0) + 0.25, f"n={n}", ha="center", fontsize=7)
    for x, p in zip(xs, allp.get("model_p", pd.Series([np.nan] * len(allp)))):
        if np.isfinite(p):
            axes[0].text(x, axes[0].get_ylim()[1] * 0.97, f"p = {p:.2f}", ha="center", fontsize=8)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([("1st\n(cross → face)" if s == "1" else f"{s}") for s in allp["saccade"]])
    axes[0].set_xlabel("saccade in the trial"); axes[0].set_ylabel("mean saccade size (deg) ± SE")
    axes[0].set_title("Mean saccade size per saccade, all participants", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    later = E[~E["first_fix"]]
    bins = np.linspace(SACC_MIN_DEG, max(later["sacc_amp"].max(), SACC_MIN_DEG + 1), 25)
    for cond in ("unfamiliar", "familiar"):
        v = later.loc[later["familiar"] == cond, "sacc_amp"]
        axes[1].hist(v, bins=bins, histtype="step", lw=2, color=COLORS[cond], density=True,
                     label=f"{cond} (n={len(v)}, mean {v.mean():.2f}°)")
    axes[1].set_xlabel("saccade size (deg)"); axes[1].set_ylabel("density")
    axes[1].set_title("Saccades on the face (2nd saccade on)", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "7_saccade_size_by_serial.png", dpi=150); plt.close(fig)

    # main (left eye) vs additional binocular microsaccade detection
    subj_sorted = sorted(T["subject"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    rates = T.groupby("subject")[["ms_rate_hz", "ms_rate_bino_hz"]].mean().reindex(subj_sorted)
    counts = E.groupby("subject")[["msrt_ms", "msrt_bino_ms"]].count().reindex(subj_sorted)
    xs = np.arange(len(subj_sorted)); w = 0.38
    for ax, df_, cols, ylab, title in (
            (axes[0], rates, ["ms_rate_hz", "ms_rate_bino_hz"], "microsaccades / s", "Microsaccade rate on the face"),
            (axes[1], counts, ["msrt_ms", "msrt_bino_ms"], "fixations with an msRT value", "msRT values per participant")):
        ax.bar(xs - w / 2, df_[cols[0]], w, color="gray", label=f"main (both-eye average, λ={LAMBDA})")
        ax.bar(xs + w / 2, df_[cols[1]], w, color="#8172B3", label=f"binocular (λ={BINO_LAMBDA})")
        ax.set_xticks(xs); ax.set_xticklabels([f"S{s}" for s in subj_sorted])
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10); ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    paired_plot(axes[2], E, "msrt_bino_ms", "msRT (ms)", "msRT, binocular detection\n(familiar vs unfamiliar)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "8_binocular_microsaccades.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(EV["amp"].clip(upper=12), bins=np.linspace(0, 12, 61), color="gray")
    axes[0].axvline(SACC_MIN_DEG, color="k", ls="--"); axes[0].set_yscale("log")
    axes[0].set_xlabel("event amplitude (deg)"); axes[0].set_ylabel("count (log)")
    axes[0].set_title("Detected (micro)saccades while the face is shown", fontsize=10)
    axes[1].scatter(EV["amp"], EV["peak_v"], s=3, alpha=0.3, color="k")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("amplitude (deg)"); axes[1].set_ylabel("peak velocity (deg/s)")
    axes[1].set_title("Main sequence (detection check)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "4_detection_check.png", dpi=150); plt.close(fig)

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(summ.round(2).to_string()); print(); print(xt); print()
    cols = ["measure", "subject", "n_familiar", "n_unfamiliar", "mean_familiar", "mean_unfamiliar",
            "cohens_d", "welch_p", "spearman_rho_rating", "model_p_familiar", "model_p_rating_z",
            "model_p_rating_z_ctrl_category"]
    print(fam[cols].round(3).to_string(index=False)); print()
    print(pd.DataFrame(beh).round(3).to_string(index=False)); print()
    print(pd.DataFrame(mod).round(3).to_string(index=False)); print()
    print(pd.DataFrame(adapt).round(3).to_string(index=False)); print()
    print("SACCADE SIZE PER SACCADE (all participants)")
    print(sacc_size[sacc_size["subject"] == "all"].drop(columns="subject").round(3).to_string(index=False)); print()
    print("FAMILIARITY RATINGS"); print(rating_summary.round(1).to_string()); print(); print(by_cat); print()
    agree = per_face.groupby("face_nationality")["prop_familiar"].describe()[["count", "mean", "min", "max"]]
    print(agree.round(2)); print()
    print("BY DURATION")
    print(by_dur[["duration_s"] + cols].round(3).to_string(index=False)); print()
    print(pd.DataFrame(dur_err).round(3).to_string(index=False))

    run_microsaccade_histograms(T)
    run_stimulus_locked(T)


####################################################
# MICROSACCADE HISTOGRAMS AND RATE BY TRIAL PHASE
# Amplitude of every detected eye movement, per participant and familiarity
# condition, in two time frames -- the face window (FACE_ONSET -> FACE_OFFSET,
# what the analyses above use) and the full trial (TRIAL_START -> TRIAL_END:
# pre-face cross, face, post-face cross, reproduction, ITI) -- plus the
# microsaccade rate in each phase. Same detection as above.
####################################################

PHASES = [("pre-face cross", "PRE_FACE_FIX_ONSET", "FACE_ONSET"),
          ("face", "FACE_ONSET", "FACE_OFFSET"),
          ("post-face cross", "POST_FACE_FIX_ONSET", "REPRODUCTION_ONSET"),
          ("reproduction", "REPRODUCTION_ONSET", "REPRODUCTION_RESPONSE"),
          ("ITI", "ITI_ONSET", "ITI_OFFSET")]
FRAMES = [("face window", "FACE_ONSET", "FACE_OFFSET"),
          ("full trial", "TRIAL_START", "TRIAL_END")]
BINS = np.logspace(np.log10(0.03), np.log10(15), 40)


def trial_times(edf):
    """Message times per main trial: {trial_num: {message: time}}."""
    out, cur = {}, None
    for t, m in edf["discrete"]["messages"]:
        m = m.decode(errors="replace").strip()
        key = m.split()[0] if m else ""
        if key == "TRIAL_START":
            label = m.split(maxsplit=1)[1] if len(m.split()) > 1 else ""
            cur = {"TRIAL_START": t} if label.isdigit() else None
            if cur is not None:
                out[int(label)] = cur
        elif cur is not None:
            if key == "PRACTICE_TRIAL":
                cur["practice"] = True
            elif key in ("PRE_FACE_FIX_ONSET", "FACE_ONSET", "FACE_OFFSET", "POST_FACE_FIX_ONSET",
                         "REPRODUCTION_ONSET", "REPRODUCTION_RESPONSE", "ITI_ONSET", "ITI_OFFSET",
                         "TRIAL_END"):
                cur[key] = t
    return {k: v for k, v in out.items() if not v.get("practice")}


def events_between(T, x, y, blinks, start, end):
    seg = (T >= start - 0.05) & (T <= end + 0.05)
    ev = detect_events(T[seg], x[seg], y[seg])
    if not len(ev) or "amp" not in ev:
        return pd.DataFrame(columns=["onset", "offset", "amp", "peak_v"])
    near = np.zeros(len(ev), bool)
    for b in blinks:
        near |= (ev["offset"] > b["stime"] - BLINK_PAD_S) & (ev["onset"] < b["etime"] + BLINK_PAD_S)
    ev = ev[~near]
    return ev[(ev["onset"] >= start) & (ev["onset"] < end)]


def run_microsaccade_histograms(trials):
    """Amplitude histograms of every detected eye movement (face window and
    full trial) and the microsaccade rate in each phase of the trial."""
    fam = trials[["subject", "trial", "familiar"]]
    subjects, _ = find_subjects()

    amp_rows, rate_rows = [], []
    for sub in subjects:
        s = sub["id"]
        cond = fam[fam["subject"] == s].set_index("trial")["familiar"]
        if cond.empty:                       # excluded participant
            continue
        edf = eyelinkio.read_edf(str(sub["edf"]))
        T = edf["times"]
        x, y, _, _, blinks, _ = gaze_signals(edf)          # both-eye average, all blinks

        for n, tt in trial_times(edf).items():
            if n not in cond.index:          # dropped trials (e.g. no response)
                continue
            c = cond[n]
            for frame, a, b in FRAMES:
                if a in tt and b in tt:
                    ev = events_between(T, x, y, blinks, tt[a], tt[b])
                    for amp in ev["amp"]:
                        amp_rows.append(dict(subject=s, familiar=c, frame=frame, amp=amp))
                    rate_rows.append(dict(subject=s, familiar=c, phase=frame, dur=tt[b] - tt[a],
                                          n_ms=int(((ev["amp"] >= MS_MIN_DEG) & (ev["amp"] < MS_MAX_DEG)).sum())
                                          if len(ev) else 0))
            for phase, a, b in PHASES:
                if a in tt and b in tt and tt[b] > tt[a]:
                    ev = events_between(T, x, y, blinks, tt[a], tt[b])
                    rate_rows.append(dict(subject=s, familiar=c, phase=phase, dur=tt[b] - tt[a],
                                          n_ms=int(((ev["amp"] >= MS_MIN_DEG) & (ev["amp"] < MS_MAX_DEG)).sum())
                                          if len(ev) else 0))

    A = pd.DataFrame(amp_rows)
    R = pd.DataFrame(rate_rows)
    rate = (R.groupby(["subject", "phase", "familiar"])[["n_ms", "dur"]].sum()
              .assign(ms_per_s=lambda d: d["n_ms"] / d["dur"]).reset_index())
    rate.to_csv(STATS_DIR / "microsaccade_rate_by_phase.csv", index=False)

    subs = sorted(A["subject"].unique())
    dur_tot = R[R["phase"].isin([f[0] for f in FRAMES])].groupby(["subject", "phase", "familiar"])["dur"].sum()
    fig, axes = plt.subplots(len(subs), 2, figsize=(12, 2.6 * len(subs)), sharex=True, squeeze=False)
    for i, s in enumerate(subs):
        for j, (frame, _, _) in enumerate(FRAMES):
            ax = axes[i, j]
            ax.axvspan(MS_MIN_DEG, MS_MAX_DEG, color="#EEEEEE", zorder=0)
            text = []
            for c in ("unfamiliar", "familiar"):
                v = A[(A["subject"] == s) & (A["frame"] == frame) & (A["familiar"] == c)]["amp"]
                secs = dur_tot.get((s, frame, c), np.nan)
                h, _ = np.histogram(v, BINS)
                ax.step(BINS[:-1], h / secs, where="post", color=COLORS[c], lw=1.6, label=c)
                ms_rate = ((v >= MS_MIN_DEG) & (v < MS_MAX_DEG)).sum() / secs
                text.append(f"{c}: {ms_rate:.2f} MS/s")
            ax.set_xscale("log")
            ax.text(0.98, 0.95, "\n".join(text), transform=ax.transAxes, ha="right", va="top", fontsize=8)
            if i == 0:
                ax.set_title(f"{frame}" + (" (face on screen)" if j == 0 else " (all phases)"), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"S{s}\nevents / s per bin")
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("amplitude (deg, log scale)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="center right")
    fig.suptitle("Amplitude of every detected eye movement (grey band = microsaccade range 0.08–1°)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "9_microsaccade_histograms.png"
    fig.savefig(out, dpi=150); plt.close(fig)

    wide = rate.pivot_table(index="subject", columns="phase", values="ms_per_s",
                            aggfunc=lambda v: v.mean())
    order = ["face window", "full trial"] + [p[0] for p in PHASES]
    print("\nMICROSACCADES PER SECOND BY TRIAL PHASE (both conditions averaged):")
    print(wide[[c for c in order if c in wide.columns]].round(2).to_string())
    print("\nSaved", out)


####################################################
# STIMULUS-LOCKED ANALYSES, as in Rosenzweig & Bonneh (2019), Sci Rep 9:3029
# Faces flashed during central fixation (1 familiar vs 7 unfamiliar, N = 19):
#   1. microsaccade / blink RATE MODULATION after stimulus onset, familiar vs
#      unfamiliar, cluster-based permutation test (familiar: prolonged
#      inhibition);
#   2. msRT early 50-250 ms (LAST microsaccade, onset of inhibition) and late
#      250-800 ms (FIRST microsaccade, release); familiar late +73 ms, early
#      ~30 ms earlier; blink RT late +78 ms;
#   3. per participant: permutation test on trial labels (effect = difference /
#      pooled SD) and a familiar-vs-unfamiliar scatter (15 of 19 longer);
#   4. late msRT across the session (decreased for unfamiliar only).
# Here everything is locked to FACE_ONSET, epoch -200..800 ms. Unlike the paper
# the eyes jump from the cross to the face ~150-200 ms after onset, so the
# early window mostly holds that saccade and the late window holds
# microsaccades after landing; large-saccade rate is plotted alongside.
####################################################

SL_EPOCH_S = (-0.2, 0.8)            # relative to FACE_ONSET
EARLY_S, LATE_S = (0.05, 0.25), (0.25, 0.8)
KERNEL_ALPHA = 1 / 0.02          # causal alpha kernel (1/s), peak at 20 ms
SL_BLINK_PAD_S = 0.1                # paper: missing data discarded with a 100 ms margin
N_PERM = 2000
SESSION_BINS = 5
SL_RNG = np.random.default_rng(2019)

####################################################
# EXTRACTION
####################################################


def face_onsets(edf):
    """{trial_num: FACE_ONSET time} for main (non-practice) trials."""
    out, cur = {}, None
    for t, m in edf["discrete"]["messages"]:
        m = m.decode(errors="replace").strip()
        if m.startswith("TRIAL_START"):
            lab = m.split(maxsplit=1)[1] if len(m.split()) > 1 else ""
            cur = int(lab) if lab.isdigit() else None
        elif cur is not None and m == "FACE_ONSET":
            out[cur] = t
    return out


def stimulus_locked_extract(sub, cond):
    edf = eyelinkio.read_edf(str(sub["edf"]))
    T = edf["times"]
    x, y, _, _, blinks, _ = gaze_signals(edf)              # both-eye average, all blinks

    rows = []
    for n, on in face_onsets(edf).items():
        if n not in cond.index:                       # dropped trials (no response) / practice
            continue
        seg = (T >= on + SL_EPOCH_S[0] - 0.2) & (T <= on + SL_EPOCH_S[1] + 0.1)
        ev = detect_events(T[seg], x[seg], y[seg])
        if not len(ev) or "amp" not in ev:
            ev = pd.DataFrame(columns=["onset", "offset", "amp"])
        bl = [(b["stime"], b["etime"]) for b in blinks
              if b["etime"] > on + SL_EPOCH_S[0] and b["stime"] < on + SL_EPOCH_S[1]]
        if len(ev):
            near = np.zeros(len(ev), bool)
            for s_, e_ in bl:
                near |= (ev["offset"] > s_ - SL_BLINK_PAD_S) & (ev["onset"] < e_ + SL_BLINK_PAD_S)
            ev = ev[~near]
        rel = lambda v: np.asarray(v, float) - on
        ms = ev[(ev["amp"] >= MS_MIN_DEG) & (ev["amp"] < MS_MAX_DEG)]
        sac = ev[ev["amp"] >= SACC_MIN_DEG]
        rows.append(dict(subject=sub["id"], trial=n, familiar=cond[n],
                         ms_on=rel(ms["onset"]), sacc_on=rel(sac["onset"]),
                         blink_on=rel([s_ for s_, _ in bl]),
                         blink_in_epoch=len(bl) > 0))
    return rows

####################################################
# MEASURES
####################################################


def rate_curve(onsets_per_trial, t_axis):
    """Event rate (Hz): onsets as 1 ms impulses convolved with a causal alpha
    kernel, averaged over trials."""
    dt = t_axis[1] - t_axis[0]
    k_t = np.arange(0, 0.2, dt)
    kernel = KERNEL_ALPHA ** 2 * k_t * np.exp(-KERNEL_ALPHA * k_t)      # integrates to 1
    pad = int(0.2 / dt)
    curves = []
    for on in onsets_per_trial:
        train = np.zeros(len(t_axis) + pad)
        idx = np.round((np.asarray(on) - t_axis[0]) / dt).astype(int) + pad
        idx = idx[(idx >= 0) & (idx < len(train))]
        np.add.at(train, idx, 1 / dt)
        curves.append(np.convolve(train, kernel * dt)[pad:pad + len(t_axis)])
    return np.mean(curves, axis=0) if curves else np.full(len(t_axis), np.nan)


def window_rt(onsets, win, which):
    """First (release) or last (onset of inhibition) event in a window, in ms."""
    v = np.asarray(onsets, float)
    v = v[(v >= win[0]) & (v < win[1])]
    if not len(v):
        return np.nan
    return (v.min() if which == "first" else v.max()) * 1000


def cluster_perm(diff, n_perm=N_PERM):
    """Paired cluster-based permutation test over time (sign flips across
    participants). diff: participants x time. Returns clusters with p."""
    n = diff.shape[0]
    tcrit = stats.t.ppf(0.975, n - 1)

    def tvals(d):
        sd = d.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sd > 0, d.mean(axis=0) / (sd / np.sqrt(n)), 0.0)

    def clusters(t):
        out, start = [], None
        for i, v in enumerate(np.append(np.abs(t) > tcrit, False)):
            if v and start is None:
                start = i
            elif not v and start is not None:
                out.append((start, i, t[start:i].sum()))
                start = None
        return out

    t_obs = tvals(diff)
    obs = clusters(t_obs)
    null = []
    for _ in range(n_perm):
        flip = SL_RNG.choice([-1, 1], size=(n, 1))
        cl = clusters(tvals(diff * flip))
        null.append(max([abs(c[2]) for c in cl], default=0))
    null = np.array(null)
    return t_obs, [dict(start=s, end=e, mass=m, p=(null >= abs(m)).mean()) for s, e, m in obs]


def within_subject_perm(values, labels, n_perm=1000):
    """Paper's per-participant test: shuffle condition labels across trials;
    effect = (familiar - unfamiliar) / pooled SD."""
    v, lab = np.asarray(values, float), np.asarray(labels)
    ok = np.isfinite(v)
    v, lab = v[ok], lab[ok]
    if (lab == "familiar").sum() < 2 or (lab == "unfamiliar").sum() < 2:
        return np.nan, np.nan

    def es(l):
        a, b = v[l == "familiar"], v[l == "unfamiliar"]
        return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)

    obs = es(lab)
    perm = np.array([es(SL_RNG.permutation(lab)) for _ in range(n_perm)])
    return obs, (np.abs(perm) >= abs(obs)).mean()

####################################################
# MAIN
####################################################


def run_stimulus_locked(trials):
    """Rosenzweig & Bonneh (2019)-style analyses; trials = this script's
    trial table (subject, trial, familiar), so dropped trials stay out."""
    subjects, _ = find_subjects()
    R = []
    for sub in subjects:
        cond = trials[trials["subject"] == sub["id"]].set_index("trial")["familiar"]
        if not cond.empty:                   # excluded participants are skipped
            R += stimulus_locked_extract(sub, cond)
    R = pd.DataFrame(R)
    R["session_pos"] = R.groupby("subject")["trial"].rank(pct=True)

    # --- msRT / bkRT per trial ---
    R["msrt_early_ms"] = [window_rt(v, EARLY_S, "last") for v in R["ms_on"]]
    R["msrt_late_ms"] = [window_rt(v, LATE_S, "first") for v in R["ms_on"]]
    R["bkrt_late_ms"] = [window_rt(v, LATE_S, "first") for v in R["blink_on"]]
    R["first_sacc_ms"] = [window_rt(v, (0.05, 0.8), "first") for v in R["sacc_on"]]
    R.drop(columns=["ms_on", "sacc_on", "blink_on"]).to_csv(STATS_DIR / "stimulus_locked_trials.csv", index=False)

    measures = [("msrt_late_ms", "msRT late 250-800 ms (first MS)  [paper: familiar +73 ms]"),
                ("msrt_early_ms", "msRT early 50-250 ms (last MS)  [paper: familiar ~-30 ms]"),
                ("bkrt_late_ms", "blink RT late 250-800 ms  [paper: familiar +78 ms]"),
                ("first_sacc_ms", f"first saccade >{SACC_MIN_DEG:g} deg to the face (not in paper)")]
    summ, per_sub = [], []
    for col, label in measures:
        pm = R.groupby(["subject", "familiar"])[col].agg(["mean", "count"]).unstack()
        ok = (pm["count"].min(axis=1) >= 3)                     # >= 3 values in each condition
        m = pm["mean"][ok]
        if len(m) < 3:
            summ.append(dict(measure=label, n_participants=len(m)))
            continue
        d = m["familiar"] - m["unfamiliar"]
        tt = stats.ttest_rel(m["familiar"], m["unfamiliar"])
        flips = SL_RNG.choice([-1, 1], size=(N_PERM, len(d)))
        p_perm = (np.abs((flips * d.values).mean(axis=1)) >= abs(d.mean())).mean()
        summ.append(dict(measure=label, n_participants=len(m),
                         familiar_ms=m["familiar"].mean(), unfamiliar_ms=m["unfamiliar"].mean(),
                         diff_ms=d.mean(), d_z=d.mean() / d.std(ddof=1),
                         t=tt.statistic, p_ttest=tt.pvalue, p_signflip=p_perm,
                         n_familiar_longer=int((d > 0).sum())))
        for s in m.index:
            g = R[R["subject"] == s]
            es, p = within_subject_perm(g[col], g["familiar"])
            per_sub.append(dict(measure=col, subject=s, familiar_ms=m.loc[s, "familiar"],
                                unfamiliar_ms=m.loc[s, "unfamiliar"], diff_ms=d[s],
                                n_familiar=int(pm.loc[s, ("count", "familiar")]),
                                n_unfamiliar=int(pm.loc[s, ("count", "unfamiliar")]),
                                effect_size=es, p_perm=p))
    summ, per_sub = pd.DataFrame(summ), pd.DataFrame(per_sub)
    summ.to_csv(STATS_DIR / "stimulus_locked_rt.csv", index=False)
    per_sub.to_csv(STATS_DIR / "stimulus_locked_rt_per_participant.csv", index=False)

    # --- rate modulation + cluster permutation ---
    t_axis = np.arange(SL_EPOCH_S[0], SL_EPOCH_S[1], 0.002)
    curves = {}
    for kind, col, use in (("microsaccades", "ms_on", None), (f"saccades >{SACC_MIN_DEG:g} deg", "sacc_on", None),
                           ("blinks", "blink_on", None)):
        per = {}
        for (s, c), g in R.groupby(["subject", "familiar"]):
            src = g if kind != "microsaccades" else g[~g["blink_in_epoch"]]
            per[(s, c)] = rate_curve(src[col], t_axis)
        subs = sorted({s for s, _ in per if (s, "familiar") in per and (s, "unfamiliar") in per})
        F = np.array([per[(s, "familiar")] for s in subs])
        U = np.array([per[(s, "unfamiliar")] for s in subs])
        t_obs, cl = cluster_perm(F - U)
        curves[kind] = (F, U, cl)
    cl_rows = [dict(signal=k, start_ms=t_axis[c["start"]] * 1000, end_ms=t_axis[c["end"] - 1] * 1000,
                    cluster_mass=c["mass"], p=c["p"]) for k, (_, _, cls) in curves.items() for c in cls]
    pd.DataFrame(cl_rows, columns=["signal", "start_ms", "end_ms", "cluster_mass", "p"]).to_csv(
        STATS_DIR / "stimulus_locked_rate_clusters.csv", index=False)

    # --- across the session (paper Fig. 6) ---
    R["session_bin"] = np.minimum((R["session_pos"] * SESSION_BINS).astype(int), SESSION_BINS - 1) + 1
    sess = R.dropna(subset=["msrt_late_ms"])
    m_sess = smf.ols("msrt_late_ms ~ session_pos * C(familiar, Treatment('unfamiliar')) + C(subject)", sess).fit()
    sess_rows = [dict(term=k, beta=m_sess.params[k], p=m_sess.pvalues[k])
                 for k in m_sess.params.index if "session_pos" in k]
    for c in ("unfamiliar", "familiar"):
        mc = smf.ols("msrt_late_ms ~ session_pos + C(subject)", sess[sess["familiar"] == c]).fit()
        sess_rows.append(dict(term=f"slope within {c}", beta=mc.params["session_pos"], p=mc.pvalues["session_pos"]))
    pd.DataFrame(sess_rows).to_csv(STATS_DIR / "stimulus_locked_session_change.csv", index=False)

    # ---------- figures ----------
    n_sub = R["subject"].nunique()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for ax, (kind, (F, U, cl)) in zip(axes, curves.items()):
        for arr, c in ((U, "unfamiliar"), (F, "familiar")):
            mu = arr.mean(axis=0)
            # Cousineau-Morey within-participant SE
            norm = arr - arr.mean(axis=1, keepdims=True) + arr.mean()
            se = norm.std(axis=0, ddof=1) / np.sqrt(len(arr)) * np.sqrt(2)
            ax.plot(t_axis * 1000, mu, color=COLORS[c], lw=2, label=f"{c}")
            ax.fill_between(t_axis * 1000, mu - se, mu + se, color=COLORS[c], alpha=0.2)
        for c in cl:
            if c["p"] < 0.05:
                ax.axvspan(t_axis[c["start"]] * 1000, t_axis[c["end"] - 1] * 1000, color="grey", alpha=0.25)
        ax.axvline(0, color="k", lw=0.8)
        ax.axvspan(LATE_S[0] * 1000, LATE_S[1] * 1000, color="#EEEEEE", zorder=0)
        best = min([c["p"] for c in cl], default=np.nan)
        ax.set_title(f"{kind} (N = {len(F)})\nsmallest cluster p = {best:.3f}" if np.isfinite(best)
                     else f"{kind} (N = {len(F)})\nno cluster", fontsize=10)
        ax.set_xlabel("time from face onset (ms)"); ax.set_ylabel("rate (Hz)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Rate modulation time-locked to face onset (paper Fig. 2); light band = late window, "
                 "grey = significant cluster", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "10_stimulus_locked_rate_modulation.png", dpi=150); plt.close(fig)

    shown = [m for m in measures if m[0] in set(per_sub["measure"])] if len(per_sub) else []
    fig, axes = plt.subplots(2, max(len(shown), 1), figsize=(4.6 * max(len(shown), 1), 8.4), squeeze=False)
    for j, (col, label) in enumerate(shown):
        ps = per_sub[per_sub["measure"] == col]
        row = summ[summ["measure"] == label].iloc[0]
        ax = axes[0, j]
        for _, r in ps.iterrows():
            ax.plot([0, 1], [r["unfamiliar_ms"], r["familiar_ms"]], "-o", color="grey", alpha=0.5, ms=4)
        ax.plot([0, 1], [row["unfamiliar_ms"], row["familiar_ms"]], "-", color="grey", lw=2, zorder=4)
        for k, cond in enumerate(["unfamiliar", "familiar"]):
            v = ps[f"{cond}_ms"]
            ax.errorbar(k, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="D", ms=11,
                        color=GROUP_COLORS[cond], mec="white", mew=1.2, capsize=4, lw=2, zorder=5,
                        label=f"{cond} mean (N = {len(v)})")
        ax.legend(frameon=False, fontsize=7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["unfamiliar", "familiar"])
        ax.set_title(label.split("[")[0] + f"\ndiff = {row['diff_ms']:+.0f} ms, p = {row['p_ttest']:.3f}, "
                     f"d_z = {row['d_z']:.2f}", fontsize=9)
        ax.set_ylabel("ms"); ax.spines[["top", "right"]].set_visible(False)
        ax = axes[1, j]
        lo = np.nanmin(ps[["familiar_ms", "unfamiliar_ms"]].values) * 0.9
        hi = np.nanmax(ps[["familiar_ms", "unfamiliar_ms"]].values) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        sig = ps["p_perm"] < 0.05
        ax.scatter(ps.loc[~sig, "familiar_ms"], ps.loc[~sig, "unfamiliar_ms"], color="grey", s=40)
        ax.scatter(ps.loc[sig, "familiar_ms"], ps.loc[sig, "unfamiliar_ms"], color="#C44E52", s=40,
                   label="participant p < .05")
        for _, r in ps.iterrows():
            ax.annotate(r["subject"], (r["familiar_ms"], r["unfamiliar_ms"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("familiar (ms)"); ax.set_ylabel("unfamiliar (ms)")
        ax.set_title(f"below diagonal = familiar longer: {int(row['n_familiar_longer'])} of "
                     f"{int(row['n_participants'])}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    if len(shown):
        axes[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Oculomotor RTs time-locked to face onset (paper Figs. 3 and 5)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG_DIR / "11_stimulus_locked_rt.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for c in ("unfamiliar", "familiar"):
        g = sess[sess["familiar"] == c].groupby(["subject", "session_bin"])["msrt_late_ms"].mean().reset_index()
        g["msrt_c"] = g["msrt_late_ms"] - g.groupby("subject")["msrt_late_ms"].transform("mean") + g["msrt_late_ms"].mean()
        a = g.groupby("session_bin")["msrt_c"].agg(["mean", "std", "count"])
        ax.errorbar(a.index + (0.05 if c == "familiar" else -0.05), a["mean"], yerr=a["std"] / np.sqrt(a["count"]),
                    marker="o", color=COLORS[c], capsize=3, label=c)
    sl = {r["term"]: r for r in sess_rows}
    ax.set_title("Late msRT across the session (paper Fig. 6)\n" +
                 ", ".join(f"{c}: slope p = {sl[f'slope within {c}']['p']:.3f}" for c in ("unfamiliar", "familiar")),
                 fontsize=10)
    ax.set_xlabel(f"session part (1-{SESSION_BINS})"); ax.set_ylabel("msRT late (ms)")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "12_stimulus_locked_session_change.png", dpi=150); plt.close(fig)

    pd.set_option("display.width", 220)
    print("\nSTIMULUS-LOCKED (Rosenzweig & Bonneh 2019)")
    print(f"Participants: {n_sub} | trials: {len(R)}")
    print(summ.round(3).to_string(index=False)); print()
    print(per_sub[per_sub["measure"] == "msrt_late_ms"].round(3).to_string(index=False)); print()
    print(pd.DataFrame(cl_rows).round(3).to_string(index=False)); print()
    print(pd.DataFrame(sess_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
