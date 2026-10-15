"""
For one participant:
  1. Read the EDF file directly (no edf2asc needed) and count saccades per trial
     inside a time window (default: while the face is on screen, FACE_ONSET -> FACE_OFFSET).
  2. Merge with the behavioral CSV (reproduction_error_s) and the ratings CSV (familiarity_rating).
  3. Trial-level moderation regression:
         error ~ familiarity_z * saccades_z + duration_planned
     The familiarity x saccades interaction answers the question.
  4. Complementary test: median split on saccade count, familiarity-error correlation
     in each half, compared with a Fisher r-to-z test.

Requires: pip install eyelinkio
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import eyelinkio

####################################################
# SETTINGS
####################################################

DATA_DIR = Path(__file__).resolve().parent / "data" / "pilot"   # relative to this script
OUTPUT_DIR = DATA_DIR / "figures"
STATS_DIR = DATA_DIR / "stats"

OUTPUT_DIR.mkdir(exist_ok=True)
STATS_DIR.mkdir(exist_ok=True)

SUBJECT = "40"

# Window in which saccades are counted (EDF message names)
WINDOW_START_MSG = "FACE_ONSET"
WINDOW_END_MSG = "FACE_OFFSET"

# Recording is binocular; count saccades of one eye only (0 = left, 1 = right)
EYE = 1

####################################################
# FILES
####################################################

edf_path = DATA_DIR / f"{SUBJECT}_001.edf"
behavioral_path = sorted(DATA_DIR.glob(f"{SUBJECT}_sess*_behavioral.csv"))[0]
ratings_path = sorted(DATA_DIR.glob(f"{SUBJECT}_sess*_ratings.csv"))[0]

####################################################
# SACCADES PER TRIAL (from EDF)
####################################################

def image_key(value):
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"\.\w+$", "", name)


def saccades_per_trial(edf_path, eye=EYE):

    edf = eyelinkio.read_edf(str(edf_path))

    messages = [(t, s.decode(errors="replace").strip()) for t, s in edf["discrete"]["messages"]]
    saccades = edf["discrete"]["saccades"]
    saccades = saccades[saccades["eye"] == eye]

    trials = []
    current = None

    for t, msg in messages:

        if msg.startswith("TRIAL_START"):
            current = {"trial_label": msg.split(maxsplit=1)[1]}

        elif current is None:
            continue

        elif msg.startswith("FACE_IMAGE"):
            current["edf_face_image"] = image_key(msg.split(maxsplit=1)[1])

        elif msg == WINDOW_START_MSG:
            current["win_start"] = t

        elif msg == WINDOW_END_MSG:
            current["win_end"] = t

        elif msg.startswith("TRIAL_END"):
            trials.append(current)
            current = None

    rows = []

    for tr in trials:

        # Practice trials are labelled e.g. "practice_1_0"; main trials are plain numbers
        if not tr["trial_label"].isdigit():
            continue

        if "win_start" not in tr or "win_end" not in tr:
            continue

        in_window = (saccades["stime"] >= tr["win_start"]) & (saccades["stime"] < tr["win_end"])
        window_s = tr["win_end"] - tr["win_start"]

        rows.append({
            "trial_num": int(tr["trial_label"]),
            "edf_face_image": tr.get("edf_face_image"),
            "window_s": window_s,
            "n_saccades": int(in_window.sum()),
            "saccade_rate_hz": in_window.sum() / window_s if window_s > 0 else np.nan
        })

    return pd.DataFrame(rows)


print("Reading", edf_path.name)
sacc = saccades_per_trial(edf_path)
print(f"Found {len(sacc)} main trials with a {WINDOW_START_MSG}->{WINDOW_END_MSG} window")

####################################################
# MERGE WITH BEHAVIOR AND RATINGS
####################################################

behavioral = pd.read_csv(behavioral_path)
ratings = pd.read_csv(ratings_path)

behavioral = behavioral[~behavioral["is_practice"]]
behavioral["face_image_key"] = behavioral["face_image"].apply(image_key)
ratings["face_image_key"] = ratings["face_image"].apply(image_key)

df = behavioral.merge(sacc, on="trial_num", how="inner")

mismatch = df[df["face_image_key"] != df["edf_face_image"]]
if len(mismatch) > 0:
    print(f"WARNING: {len(mismatch)} trials where the EDF face image does not match the CSV")

df = df.merge(
    ratings[["face_image_key", "familiarity_rating", "attractiveness_rating"]],
    on="face_image_key",
    how="inner"
)

df = df.dropna(subset=["familiarity_rating", "reproduction_error_s", "n_saccades"])

print(f"Trials in analysis: {len(df)}")

df.to_csv(STATS_DIR / f"{SUBJECT}_trial_saccades_familiarity.csv", index=False)

####################################################
# DESCRIPTIVES
####################################################

print("\n====================")
print("DESCRIPTIVES")
print("====================")
print(df.groupby("face_duration_planned_s")[["n_saccades", "saccade_rate_hz", "reproduction_error_s", "familiarity_rating"]]
      .agg(["mean", "std"]).round(3))

r_fam_sacc, p_fam_sacc = stats.spearmanr(df.familiarity_rating, df.n_saccades)
print(f"\nFamiliarity vs number of saccades: Spearman rho={r_fam_sacc:.3f}, p={p_fam_sacc:.4f}")

####################################################
# MODERATION REGRESSION
####################################################

def ols(y, X, names):

    X = np.column_stack([np.ones(len(y))] + X)
    names = ["Intercept"] + names

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)

    table = pd.DataFrame({"beta": beta, "SE": se, "t": t, "p": p}, index=names)
    return table, r2, dof


def z(x):
    return (x - x.mean()) / x.std()


df["familiarity_z"] = z(df.familiarity_rating)
df["saccades_z"] = z(df.n_saccades)
df["duration_c"] = df.face_duration_planned_s - df.face_duration_planned_s.mean()

y = df.reproduction_error_s.values

model, r2, dof = ols(
    y,
    [df.familiarity_z.values, df.saccades_z.values,
     (df.familiarity_z * df.saccades_z).values, df.duration_c.values],
    ["familiarity_z", "saccades_z", "familiarity_z:saccades_z", "duration_planned_c"]
)

print("\n====================")
print("MODERATION: error ~ familiarity * n_saccades + duration")
print("====================")
print(model.round(4))
print(f"R^2 = {r2:.3f}, df = {dof}")

inter = model.loc["familiarity_z:saccades_z"]
print(f"\n>>> Familiarity x saccades interaction: beta={inter.beta:.4f}, t({dof})={inter.t:.2f}, p={inter.p:.4f}")

# Same model using saccade rate (saccades / s) instead of count,
# since the count is partly driven by face duration
df["rate_z"] = z(df.saccade_rate_hz)

model_rate, r2_rate, dof_rate = ols(
    y,
    [df.familiarity_z.values, df.rate_z.values,
     (df.familiarity_z * df.rate_z).values, df.duration_c.values],
    ["familiarity_z", "rate_z", "familiarity_z:rate_z", "duration_planned_c"]
)

print("\n--- Robustness: saccade RATE instead of count ---")
print(model_rate.round(4))

model.to_csv(STATS_DIR / f"{SUBJECT}_moderation_saccade_count.csv")
model_rate.to_csv(STATS_DIR / f"{SUBJECT}_moderation_saccade_rate.csv")

####################################################
# MEDIAN SPLIT + FISHER Z
####################################################

median = df.n_saccades.median()

# Ties at the median go to the "low" group
df["saccade_group"] = np.where(df.n_saccades > median, "high", "low")

split_rows = []
for group, g in df.groupby("saccade_group"):
    r, p = stats.pearsonr(g.familiarity_rating, g.reproduction_error_s)
    rho, p_rho = stats.spearmanr(g.familiarity_rating, g.reproduction_error_s)
    split_rows.append({"group": group, "N": len(g), "mean_saccades": g.n_saccades.mean(),
                       "Pearson_r": r, "Pearson_p": p, "Spearman_rho": rho, "Spearman_p": p_rho})

split = pd.DataFrame(split_rows).set_index("group")

r_lo, n_lo = split.loc["low", "Pearson_r"], split.loc["low", "N"]
r_hi, n_hi = split.loc["high", "Pearson_r"], split.loc["high", "N"]
z_diff = (np.arctanh(r_hi) - np.arctanh(r_lo)) / np.sqrt(1 / (n_hi - 3) + 1 / (n_lo - 3))
p_diff = 2 * stats.norm.sf(abs(z_diff))

print("\n====================")
print(f"MEDIAN SPLIT (median = {median} saccades)")
print("====================")
print(split.round(4))
print(f"\n>>> Fisher z (high vs low) = {z_diff:.2f}, p = {p_diff:.4f}")

split["Fisher_z_high_vs_low"] = z_diff
split["Fisher_p"] = p_diff
split.to_csv(STATS_DIR / f"{SUBJECT}_median_split_familiarity_error.csv")

####################################################
# PLOTS
####################################################

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
colors = {"low": "tab:blue", "high": "tab:orange"}
for group, g in df.groupby("saccade_group"):
    label = f"{group} saccades (n={len(g)}, r={split.loc[group, 'Pearson_r']:.2f})"
    ax.scatter(g.familiarity_rating, g.reproduction_error_s, s=20, alpha=0.7, color=colors[group], label=label)
    m, b = np.polyfit(g.familiarity_rating, g.reproduction_error_s, 1)
    xs = np.linspace(df.familiarity_rating.min(), df.familiarity_rating.max(), 50)
    ax.plot(xs, m * xs + b, color=colors[group], linewidth=1.5)
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("Familiarity rating")
ax.set_ylabel("Reproduction error (s)")
ax.set_title(f"Interaction p = {inter.p:.3f}")
ax.legend(fontsize=8, frameon=False)

ax = axes[1]
for dur, g in df.groupby("face_duration_planned_s"):
    ax.scatter(g.n_saccades + np.random.uniform(-0.15, 0.15, len(g)), g.reproduction_error_s,
               s=20, alpha=0.7, label=f"{dur} s face")
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel(f"Saccades ({WINDOW_START_MSG} - {WINDOW_END_MSG})")
ax.set_ylabel("Reproduction error (s)")
ax.legend(fontsize=8, frameon=False)

fig.suptitle(f"Subject {SUBJECT}: does saccade count moderate familiarity -> error?")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f"{SUBJECT}_saccades_familiarity_error.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nDone!")
print("Figure saved to:", OUTPUT_DIR / f"{SUBJECT}_saccades_familiarity_error.png")
print("Stats (CSV) saved to:", STATS_DIR)
