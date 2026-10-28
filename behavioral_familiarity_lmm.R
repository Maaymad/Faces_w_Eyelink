# Familiarity x duration on reproduction error -- same model as the previous
# (online) experiment's f_interaction (Faces_analysis/AS Final R code-ALL.Rmd),
# with reproduction_error_s (reproduced - shown, in seconds) as the outcome.
#
#   reproduction_error_s ~ familiarity * duration_factor * stimuli_condition
#                          + (1 | participant) + (1 | face)
#
# familiarity = the participant's own rating (0-100), as in the previous model.
# duration_factor: short (0.8 s, reference) / long (1.6 s).
# stimuli_condition: IL (reference) / Neutral / UK.
# Note: because error = reproduced - shown and duration is in the model, the
# familiarity terms are the same as for reproduced duration (in s, not ms).
#
# Uses every participant in data/ with a behavioral and a ratings CSV from the
# same session. Run from the project folder:
#   Rscript behavioral_familiarity_lmm.R
# Exclusions (run first; see "exclusions" below): reproductions > 4 s, trials
# > 3 MAD from their participant x duration median, and participants
# with a flat / reversed / > 1.5 timing slope. Reported in exclusion_report.txt,
# listed per trial in trial_exclusions.csv (read by the eye-movement analysis).
# Output: data/analysis_familiarity/behavioral/lmm_results.txt and figures in
#         data/analysis_familiarity/behavioral/figures_R/:
#   0_exclusions_trials_per_participant  every trial, discarded ones marked
#   0b_exclusions_trial_counts           kept / discarded trials per participant
#   1_reproduced_vs_shown      reproduced duration per participant and duration
#                              (bias and central tendency)
#   2_error_by_familiarity     trial errors vs familiarity rating, with the
#                              model-2 fit (95% CI) per duration
#   3_error_by_condition       error by face category x duration
#                              (the confound behind the familiarity effect)
#   3b_error_by_familiarity    the same as 3, by familiar vs unfamiliar
#   4_familiar_vs_unfamiliar   familiar (rating >= 50) vs unfamiliar, per duration
#   5_familiarity_slopes       familiarity effect (0 -> 100 rating) per duration,
#                              models 2 and 3, with 95% CI
#   6_accuracy_precision       accuracy (|error|) and precision (CV) for familiar
#                              vs unfamiliar faces, per duration
#   7_accuracy_precision_slopes familiarity effect on accuracy and precision,
#                              absolute and relative to the duration shown
#
# Accuracy and precision, defined as in the previous experiment's R code:
#   abs_error      = |reproduced - shown| (s)                     -> accuracy
#   cv_resid_trial = |reproduced - participant's mean at that duration|
#                    / participant's mean at that duration        -> precision
#   rel_error      = (reproduced - shown) / shown     -> signed error, relative
#   rel_abs_error  = |reproduced - shown| / shown     -> accuracy, relative
#   Errors grow with the duration shown (scalar timing / Weber), so absolute
#   errors make the 1.6 s effects look ~2x the 0.8 s ones; the relative
#   versions put both durations on the same scale for the familiarity x
#   duration comparison. (CV is already relative.)
#   (lower = more accurate / more precise). Models: as in the previous
#   experiment (f_accuracy, f_cv without participant_origin -- all participants
#   here are from one origin), plus the same with condition x duration
#   controlled, because familiarity is nearly confounded with face category.
#
# Timing analyses (results: timing_slope_variance.txt):
#   8_timing_slopes   linear fit of reproduced on shown duration, separately for
#                     familiar and unfamiliar faces. Slope 1 = veridical timing;
#                     slope < 1 = regression to the mean (central tendency).
#                     Steeper/shallower slope for familiar faces is tested with
#                       reproduced ~ shown * familiar (+ condition control),
#                     random intercept and slope of shown per participant, and
#                     with per-participant slopes (paired t-test).
#   9_variance        variance (SD) of reproduced durations for each duration
#                     and face condition (familiar / unfamiliar; also face
#                     category in the text output), per participant; tested on
#                     log SD with participant as random intercept.
#   8b / 9b           the same with three face groups (timing_slope_variance_3groups.txt):
#                     baseline (neutral database faces), unfamiliar celebrities
#                     (rated < 50) and familiar celebrities (rated >= 50). Comparing
#                     unfamiliar celebrities with the baseline separates familiarity
#                     from "celebrity photo vs database photo".
#   10_error_by_familiarity_bins(_centered)
#                     reproduction error by familiarity-rating bins of 10
#                     (0-10, ..., 90-100), all participants pooled, 0.8 s and
#                     1.6 s separately; mean +- SE over trials, n per bin, faint
#                     lines = participants. "_centered" removes each
#                     participant's mean error at that duration. Table:
#                     error_by_familiarity_bins.csv

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(ggplot2)
})

# Find the project's data/ folder (relative to this script) whether this runs
# with Rscript, source(), or line by line in RStudio (whose working directory
# is usually not the project folder). Each way of finding the script's folder
# is tried on its own, so one failing does not skip the others.
find_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(f)) return(dirname(normalizePath(f)))                     # Rscript
  of <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(of)) return(dirname(normalizePath(of)))                 # source()
  rs <- tryCatch(if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable())
                   rstudioapi::getSourceEditorContext()$path else "",
                 error = function(e) "")
  if (nzchar(rs)) return(dirname(normalizePath(rs)))                   # RStudio editor
  getwd()
}
script_dir <- find_script_dir()
# Use the first candidate that actually holds participant files (another
# folder named "data", e.g. in the home directory, is skipped).
candidates <- unique(c(file.path(script_dir, "data"), file.path(getwd(), "data")))
has_participants <- function(d) dir.exists(d) &&
  length(list.files(d, pattern = "_behavioral\\.csv$")) > 0
found <- Filter(has_participants, candidates)
if (length(found) == 0) stop("Could not find a data/ folder with participant files. Tried:\n  ",
                            paste(candidates, collapse = "\n  "),
                            "\nOpen the script from the project folder in RStudio, or setwd() to it.")
data_dir <- found[[1]]
cat("Data folder:", data_dir, "\n")
out_dir <- file.path(data_dir, "analysis_familiarity", "behavioral")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

image_key <- function(x) sub("\\.[A-Za-z]+$", "", basename(gsub("\\\\", "/", x)))

beh_files <- sort(list.files(data_dir, pattern = "_behavioral\\.csv$", full.names = TRUE))
rows <- list()
for (bf in beh_files) {
  rf <- sub("_behavioral\\.csv$", "_ratings.csv", bf)
  if (!file.exists(rf)) next
  beh <- read.csv(bf, colClasses = c(participant_id = "character"))
  rat <- read.csv(rf)
  beh <- beh[tolower(as.character(beh$is_practice)) != "true", ]
  beh$face <- image_key(beh$face_image)
  rat$face <- image_key(rat$face_image)
  d <- merge(beh, rat[, c("face", "familiarity_rating", "attractiveness_rating")], by = "face")
  d$participant <- strsplit(basename(bf), "_")[[1]][1]
  rows[[length(rows) + 1]] <- d
}
if (length(rows) == 0) stop("No participants found in ", data_dir,
                            " (need <id>_sess..._behavioral.csv and the matching _ratings.csv).")
full <- do.call(rbind, rows)
if (!"reproduction_error_s" %in% names(full)) stop("The behavioral files have no reproduction_error_s column.")
full <- full[!is.na(full$reproduction_error_s) & !is.na(full$familiarity_rating), ]
full$familiarity <- full$familiarity_rating
full$duration_factor <- factor(ifelse(full$face_duration_planned_s > 1, "long", "short"),
                               levels = c("short", "long"))
full$stimuli_condition <- factor(c(israeli = "IL", neutral = "Neutral", uk = "UK")[full$face_nationality],
                                 levels = c("IL", "Neutral", "UK"))

# ---------- exclusions ----------
# Applied in this order; every discarded trial and participant is listed with
# its reason in exclusion_report.txt and trial_exclusions.csv. The eye-movement
# analysis (familiarity_oculomotor_analysis.py) reads trial_exclusions.csv and
# drops the same trials and participants, so run this script first.
#   1. trials with a reproduction above MAX_REPRO_S (not a real reproduction)
#   2. trials more than MAD_K scaled MADs from the median of their cell
#      (participant x duration; scaled MAD = 1.4826 * MAD)
#   3. participants whose timing slope (reproduced on shown duration, from the
#      kept trials) is flat (not > 0, one-sided p >= .05), reversed (< 0), or
#      above MAX_SLOPE
MAX_REPRO_S <- 4.0
MAD_K <- 3
MAX_SLOPE <- 1.5

full$exclusion <- NA_character_
full$exclusion[full$reproduced_duration_s > MAX_REPRO_S] <- sprintf("reproduction > %.1f s", MAX_REPRO_S)
excl_cell <- interaction(full$participant, full$duration_factor, drop = TRUE)
pending <- is.na(full$exclusion)
full$mad_dist <- NA_real_
full$mad_dist[pending] <- ave(full$reproduced_duration_s[pending], excl_cell[pending],
                              FUN = function(x) {
                                s <- mad(x)
                                if (!is.finite(s) || s == 0) rep(0, length(x)) else abs(x - median(x)) / s
                              })
full$exclusion[pending & full$mad_dist > MAD_K] <- sprintf("> %g MAD from cell median", MAD_K)

subj_slopes <- do.call(rbind, lapply(split(full[is.na(full$exclusion), ], full$participant[is.na(full$exclusion)],
                                           drop = TRUE), function(d) {
  ct <- coef(summary(lm(reproduced_duration_s ~ face_duration_planned_s, data = d)))
  b <- ct["face_duration_planned_s", ]
  p_pos <- pt(b["t value"], df = nrow(d) - 2, lower.tail = FALSE)
  why <- if (b["Estimate"] < 0) "reversed slope (0.8 s reproduced longer than 1.6 s)"
         else if (p_pos >= 0.05) "flat slope (not > 0, one-sided p >= .05)"
         else if (b["Estimate"] > MAX_SLOPE) sprintf("slope > %.1f", MAX_SLOPE)
         else NA_character_
  data.frame(participant = d$participant[1], n_trials = nrow(d),
             mean_short = mean(d$reproduced_duration_s[d$duration_factor == "short"]),
             mean_long = mean(d$reproduced_duration_s[d$duration_factor == "long"]),
             slope = unname(b["Estimate"]), slope_se = unname(b["Std. Error"]),
             p_one_sided = unname(p_pos), excluded_reason = why)
}))
bad_subj <- subj_slopes$participant[!is.na(subj_slopes$excluded_reason)]
full$subject_excluded <- full$participant %in% bad_subj

trial_excl <- full[, c("participant", "trial_num", "face", "face_duration_planned_s", "stimuli_condition",
                       "reproduced_duration_s", "mad_dist", "exclusion")]
trial_excl$subject_excluded <- full$subject_excluded
trial_excl$excluded <- !is.na(trial_excl$exclusion) | trial_excl$subject_excluded
trial_excl$reason <- ifelse(!is.na(trial_excl$exclusion), trial_excl$exclusion,
                            ifelse(trial_excl$subject_excluded,
                                   paste("participant excluded:",
                                         subj_slopes$excluded_reason[match(trial_excl$participant,
                                                                           subj_slopes$participant)]), ""))
trial_excl$exclusion <- NULL
trial_excl <- trial_excl[order(trial_excl$participant, trial_excl$trial_num), ]
write.csv(trial_excl, file.path(out_dir, "trial_exclusions.csv"), row.names = FALSE)
write.csv(subj_slopes, file.path(out_dir, "participant_slopes.csv"), row.names = FALSE)

sink(file.path(out_dir, "exclusion_report.txt"), split = TRUE)
cat("EXCLUSIONS\n==========\n")
cat(sprintf("Rules: reproduction > %.1f s; > %g scaled MADs from the median of participant x duration;\n",
            MAX_REPRO_S, MAD_K))
cat(sprintf("participants with a timing slope < 0, not > 0 (one-sided p >= .05), or > %.1f.\n\n", MAX_SLOPE))
cat("Trials before exclusions:", nrow(full), "from", nlevels(factor(full$participant)), "participants\n\n")
cat("1-2. Discarded trials (participants that are kept or excluded alike):\n")
td <- full[!is.na(full$exclusion), ]
td <- td[order(td$participant, td$trial_num), ]
if (nrow(td)) {
  for (i in seq_len(nrow(td)))
    cat(sprintf("  %s trial %3d  %-8s %.1f s %-7s reproduced %.2f s  (%s MAD)  -> %s\n",
                td$participant[i], td$trial_num[i], td$face[i], td$face_duration_planned_s[i],
                td$stimuli_condition[i], td$reproduced_duration_s[i],
                ifelse(is.na(td$mad_dist[i]), "-", sprintf("%.1f", td$mad_dist[i])), td$exclusion[i]))
} else cat("  none\n")
cat("\n  Count per reason:\n"); print(table(td$exclusion))
cat(sprintf("\n  (For comparison: %d trials would be discarded with %d MADs instead of %g.)\n",
            sum(full$mad_dist > 4, na.rm = TRUE), 4, MAD_K))
cat("\n3. Timing slope per participant (kept trials):\n")
print(transform(subj_slopes, mean_short = round(mean_short, 3), mean_long = round(mean_long, 3),
                slope = round(slope, 3), slope_se = round(slope_se, 3), p_one_sided = signif(p_one_sided, 3)),
      row.names = FALSE)
cat("\nDiscarded participants:", if (length(bad_subj)) paste(bad_subj, collapse = ", ") else "none", "\n")
for (s in bad_subj) cat(sprintf("  %s: %s\n", s, subj_slopes$excluded_reason[subj_slopes$participant == s]))
cat("\nPer participant (trials):\n")
per_subj <- data.frame(
  participant = levels(factor(full$participant)),
  total = as.vector(table(factor(full$participant))),
  over_max = as.vector(tapply(grepl("^reproduction", full$exclusion), factor(full$participant), sum)),
  mad = as.vector(tapply(grepl("MAD", full$exclusion), factor(full$participant), sum)))
per_subj$kept <- per_subj$total - per_subj$over_max - per_subj$mad
per_subj$participant_excluded <- per_subj$participant %in% bad_subj
print(per_subj, row.names = FALSE)
kept_all <- full[is.na(full$exclusion) & !full$subject_excluded, ]
cat(sprintf("\nAnalyzed: %d trials from %d participants (%d trials discarded in total).\n",
            nrow(kept_all), length(unique(kept_all$participant)), nrow(full) - nrow(kept_all)))
sink()

# figure: every trial per participant, with the discarded ones marked
excl_fig_dir <- file.path(out_dir, "figures_R")
dir.create(excl_fig_dir, recursive = TRUE, showWarnings = FALSE)
pd_ex <- full
pd_ex$status <- factor(ifelse(is.na(pd_ex$exclusion), "kept",
                              ifelse(grepl("MAD", pd_ex$exclusion), sprintf("> %g MAD", MAD_K),
                                     sprintf("> %.0f s", MAX_REPRO_S))),
                       levels = c("kept", sprintf("> %g MAD", MAD_K), sprintf("> %.0f s", MAX_REPRO_S)))
subj_lab <- with(subj_slopes, setNames(
  sprintf("%s  slope %.2f%s", participant, slope,
          ifelse(is.na(excluded_reason), "", "  EXCLUDED")), participant))
pd_ex$panel <- factor(subj_lab[as.character(pd_ex$participant)], levels = subj_lab)
pd_ex$duration <- factor(paste(pd_ex$face_duration_planned_s, "s"), levels = c("0.8 s", "1.6 s"))
dur_cols <- c(`0.8 s` = "#4C72B0", `1.6 s` = "#DD8452")
excl_panels <- data.frame(panel = factor(subj_lab[!is.na(subj_slopes$excluded_reason)], levels = subj_lab))
g_ex <- ggplot(pd_ex, aes(familiarity_rating, reproduced_duration_s, colour = duration)) +
  (if (nrow(excl_panels)) geom_rect(data = excl_panels, inherit.aes = FALSE,
                                    aes(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf),
                                    fill = "#FDE0DD", alpha = 0.6)) +
  geom_hline(data = data.frame(y = c(0.8, 1.6), duration = names(dur_cols)),
             aes(yintercept = y, colour = duration), linetype = "dashed", linewidth = 0.4) +
  geom_vline(xintercept = 50, linetype = "dotted", colour = "grey60") +
  geom_point(aes(shape = status, size = status), alpha = 0.75) +
  geom_smooth(data = pd_ex[pd_ex$status == "kept", ], method = "lm", formula = y ~ x,
              se = FALSE, linewidth = 0.8) +
  scale_shape_manual(values = c(16, 4, 8), drop = FALSE) +
  scale_size_manual(values = c(1.2, 2.8, 2.8), drop = FALSE) +
  scale_colour_manual(values = dur_cols) +
  scale_x_continuous(breaks = c(0, 50, 100), limits = c(-3, 103)) +
  facet_wrap(~ panel, ncol = 4) +
  labs(x = "familiarity rating (participant's own, 0-100)", y = "reproduced duration (s)",
       shape = "trial", size = "trial", colour = "shown duration",
       title = "All trials per participant: discarded trials (x / *) and discarded participants (red)",
       subtitle = sprintf("lines = linear fit of kept trials on familiarity; dashed = veridical (0.8 / 1.6 s); slope in panel title = reproduced on shown duration\nrules: > %.0f s, > %g MAD per participant x duration, timing slope <= 0 / flat / > %.1f",
                          MAX_REPRO_S, MAD_K, MAX_SLOPE)) +
  theme_bw(base_size = 9) + theme(legend.position = "bottom", plot.subtitle = element_text(size = 7))
ggsave(file.path(excl_fig_dir, "0_exclusions_trials_per_participant.png"), g_ex,
       width = 10, height = 2.4 * ceiling(nrow(subj_slopes) / 4) + 1, dpi = 150)
cnt <- rbind(data.frame(participant = per_subj$participant, status = "kept", n = per_subj$kept),
             data.frame(participant = per_subj$participant, status = sprintf("> %g MAD", MAD_K), n = per_subj$mad),
             data.frame(participant = per_subj$participant, status = sprintf("> %.0f s", MAX_REPRO_S),
                        n = per_subj$over_max))
cnt$status <- factor(cnt$status, levels = rev(levels(pd_ex$status)))
cnt$participant_excluded <- cnt$participant %in% bad_subj
g_cnt <- ggplot(cnt, aes(participant, n, fill = status)) +
  geom_col(aes(alpha = participant_excluded)) +
  scale_alpha_manual(values = c(`FALSE` = 1, `TRUE` = 0.35), labels = c("kept", "excluded"),
                     name = "participant") +
  scale_fill_manual(values = setNames(c("grey55", "#E6550D", "#B10026"),
                                      c("kept", sprintf("> %g MAD", MAD_K), sprintf("> %.0f s", MAX_REPRO_S)))) +
  geom_text(data = per_subj, inherit.aes = FALSE,
            aes(participant, total + 3, label = ifelse(participant_excluded, "excl.", kept)), size = 2.8) +
  labs(x = NULL, y = "trials", fill = "trial",
       title = "Trials per participant after exclusions (number = kept trials)") +
  theme_bw(base_size = 10)
ggsave(file.path(excl_fig_dir, "0b_exclusions_trial_counts.png"), g_cnt, width = 9, height = 4, dpi = 150)

# everything below uses the kept trials only
full <- full[is.na(full$exclusion) & !full$subject_excluded, ]
full$participant <- droplevels(factor(full$participant))
full$participant <- factor(full$participant)
full$appeal <- full$attractiveness_rating
# accuracy and precision (as in the previous experiment)
full$abs_error <- abs(full$reproduction_error_s)
full$rel_error <- full$reproduction_error_s / full$face_duration_planned_s
full$rel_abs_error <- full$abs_error / full$face_duration_planned_s
cell <- interaction(full$participant, full$duration_factor, drop = TRUE)
participant_mean <- ave(full$reproduced_duration_s, cell, FUN = mean)
full$cv_resid_trial <- abs(full$reproduced_duration_s - participant_mean) / participant_mean
full$face <- factor(full$face)

sink(file.path(out_dir, "lmm_results.txt"), split = TRUE)
cat("Participants:", paste(levels(full$participant), collapse = ", "),
    "| trials:", nrow(full), "| faces:", nlevels(full$face), "\n\n")

cat("========== Interaction model (as in the previous experiment) ==========\n")
f_interaction <- reproduction_error_s ~ familiarity * duration_factor * stimuli_condition +
  (1 | participant) + (1 | face)
m_int <- lmer(f_interaction, data = full)
print(summary(m_int))
cat("\n-- Type III ANOVA (Satterthwaite) --\n")
print(anova(m_int))

cat("\n-- Familiarity slope (s per rating point) by duration and condition --\n")
print(summary(emtrends(m_int, ~ duration_factor | stimuli_condition, var = "familiarity"),
              infer = TRUE))

cat("\n========== Simpler model: familiarity x duration, condition as covariate ==========\n")
f_fxd <- reproduction_error_s ~ familiarity * duration_factor + stimuli_condition +
  (1 | participant) + (1 | face)
m_fxd <- lmer(f_fxd, data = full)
print(summary(m_fxd))
cat("\n-- Familiarity slope (s per rating point) by duration --\n")
print(summary(emtrends(m_fxd, ~ duration_factor, var = "familiarity"), infer = TRUE))
cat("\nFamiliarity effect from 0 to 100 rating = slope x 100 (seconds).\n")

cat("\n========== Check: familiarity x duration, controlling for condition x duration ==========\n")
cat("Familiarity is nearly confounded with condition (Israeli faces familiar), so a\n")
cat("condition-by-duration difference could show up as familiarity x duration.\n")
f_ctrl <- reproduction_error_s ~ familiarity * duration_factor + stimuli_condition * duration_factor +
  (1 | participant) + (1 | face)
m_ctrl <- lmer(f_ctrl, data = full)
print(coef(summary(m_ctrl)))
print(summary(emtrends(m_ctrl, ~ duration_factor, var = "familiarity"), infer = TRUE))

cat("\n\n################ ACCURACY AND PRECISION ################\n")
cat("abs_error = |reproduced - shown| (s): lower = more accurate\n")
cat("cv_resid_trial = |reproduced - participant mean| / participant mean (per duration): lower = more precise\n")
acc_prec_models <- list(
  "A1. accuracy, as in the previous experiment" =
    abs_error ~ familiarity * duration_factor + (1 | participant),
  "A2. accuracy, controlling condition x duration" =
    abs_error ~ familiarity * duration_factor + stimuli_condition * duration_factor +
      (1 | participant) + (1 | face),
  "P1. precision (CV), as in the previous experiment (without origin)" =
    cv_resid_trial ~ familiarity * duration_factor * stimuli_condition + appeal +
      (1 | participant) + (1 | face),
  "P2. precision (CV), familiarity x duration, controlling condition x duration" =
    cv_resid_trial ~ familiarity * duration_factor + stimuli_condition * duration_factor + appeal +
      (1 | participant) + (1 | face))
acc_prec_models[["A3. relative accuracy |error|/shown, as in the previous experiment"]] <-
  rel_abs_error ~ familiarity * duration_factor + (1 | participant)
acc_prec_models[["A4. relative accuracy |error|/shown, controlling condition x duration"]] <-
  rel_abs_error ~ familiarity * duration_factor + stimuli_condition * duration_factor +
    (1 | participant) + (1 | face)
acc_prec_models[["R2. relative signed error (error/shown): familiarity x duration + condition"]] <-
  rel_error ~ familiarity * duration_factor + stimuli_condition + (1 | participant) + (1 | face)
acc_prec_models[["R3. relative signed error: + condition x duration"]] <-
  rel_error ~ familiarity * duration_factor + stimuli_condition * duration_factor +
    (1 | participant) + (1 | face)
acc_prec_fits <- list()
for (nm in names(acc_prec_models)) {
  cat("\n==========", nm, "==========\n")
  fit <- lmer(acc_prec_models[[nm]], data = full)
  acc_prec_fits[[nm]] <- fit
  print(coef(summary(fit)))
  cat("-- familiarity slope by duration (",
      ifelse(grepl("^A[12]", nm), "s", ifelse(grepl("^P", nm), "CV units", "proportion of shown duration")),
      "per rating point) --\n")
  print(summary(emtrends(fit, ~ duration_factor, var = "familiarity"), infer = TRUE))
}
sink()

####################################################
# FIGURES
####################################################

fig_dir <- file.path(out_dir, "figures_R")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dur_cols <- c(short = "#C44E52", long = "#55A868")
dur_labels <- c(short = "0.8 s", long = "1.6 s")
theme_set(theme_classic(base_size = 12))
full$duration_label <- factor(dur_labels[as.character(full$duration_factor)], levels = dur_labels)
full$familiar <- factor(ifelse(full$familiarity >= 50, "familiar", "unfamiliar"),
                        levels = c("unfamiliar", "familiar"))

# mean +- SE across participants (each participant's mean counts once)
group_summary <- function(d, by) {
  pm <- aggregate(reproduction_error_s ~ ., data = d[, c("participant", by, "reproduction_error_s")], FUN = mean)
  gm <- aggregate(reproduction_error_s ~ ., data = pm[, c(by, "reproduction_error_s")],
                  FUN = function(v) c(mean = mean(v), se = sd(v) / sqrt(length(v))))
  gm <- do.call(data.frame, gm)
  names(gm)[names(gm) == "reproduction_error_s.mean"] <- "mean"
  names(gm)[names(gm) == "reproduction_error_s.se"] <- "se"
  list(participant = pm, group = gm)
}

# 1. Reproduced vs shown duration
g1 <- ggplot(full, aes(participant, reproduced_duration_s, fill = duration_label)) +
  geom_hline(yintercept = c(0.8, 1.6), colour = dur_cols, linetype = "dashed") +
  geom_boxplot(outlier.shape = NA, alpha = 0.5, position = position_dodge(0.8)) +
  geom_point(aes(colour = duration_label), size = 0.9, alpha = 0.5,
             position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.8)) +
  scale_fill_manual(values = unname(dur_cols), name = "shown") +
  scale_colour_manual(values = unname(dur_cols), guide = "none") +
  labs(x = "participant", y = "reproduced duration (s)",
       title = "Reproduced duration per participant",
       subtitle = "dashed = the true durations (0.8 s, 1.6 s)")
ggsave(file.path(fig_dir, "1_reproduced_vs_shown.png"), g1, width = 8, height = 4.5, dpi = 150)

# 2. Error vs familiarity with the model-2 fit
grid <- expand.grid(familiarity = seq(0, 100, 5), duration_factor = levels(full$duration_factor))
pred <- as.data.frame(summary(emmeans(m_fxd, ~ familiarity * duration_factor,
                                      at = list(familiarity = seq(0, 100, 5)))))
pred$duration_label <- factor(dur_labels[as.character(pred$duration_factor)], levels = dur_labels)
g2 <- ggplot(full, aes(familiarity, reproduction_error_s)) +
  geom_hline(yintercept = 0, colour = "grey50") +
  geom_jitter(aes(colour = stimuli_condition), width = 1, height = 0, size = 1.3, alpha = 0.6) +
  geom_ribbon(data = pred, aes(x = familiarity, ymin = lower.CL, ymax = upper.CL), inherit.aes = FALSE,
              fill = "grey30", alpha = 0.2) +
  geom_line(data = pred, aes(x = familiarity, y = emmean), inherit.aes = FALSE, linewidth = 1) +
  facet_wrap(~ duration_label) +
  scale_colour_manual(values = c(IL = "#DD8452", Neutral = "#4C72B0", UK = "#8172B3"), name = "faces") +
  labs(x = "familiarity rating", y = "reproduction error (s)\n(reproduced - shown)",
       title = "Reproduction error by familiarity",
       subtitle = "line = model 2 (familiarity x duration + condition), 95% CI")
ggsave(file.path(fig_dir, "2_error_by_familiarity.png"), g2, width = 9, height = 4.5, dpi = 150)

# 3. Error by face category x duration
s3 <- group_summary(full, c("stimuli_condition", "duration_label"))
g3 <- ggplot(s3$group, aes(stimuli_condition, mean, colour = duration_label, group = duration_label)) +
  geom_hline(yintercept = 0, colour = "grey50") +
  geom_line(data = s3$participant, aes(y = reproduction_error_s,
                                        group = interaction(participant, duration_label)),
            alpha = 0.25, position = position_dodge(0.3)) +
  geom_point(size = 3, position = position_dodge(0.3)) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.15, position = position_dodge(0.3)) +
  geom_line(position = position_dodge(0.3)) +
  scale_colour_manual(values = unname(dur_cols), name = "shown") +
  labs(x = "face category", y = "reproduction error (s)",
       title = "Error by face category and duration",
       subtitle = "mean +- SE across participants; faint = each participant")
ggsave(file.path(fig_dir, "3_error_by_condition.png"), g3, width = 6.5, height = 4.5, dpi = 150)

# 3b. The same, by familiarity instead of face category
s3b <- group_summary(full, c("familiar", "duration_label"))
g3b <- ggplot(s3b$group, aes(familiar, mean, colour = duration_label, group = duration_label)) +
  geom_hline(yintercept = 0, colour = "grey50") +
  geom_line(data = s3b$participant, aes(y = reproduction_error_s,
                                         group = interaction(participant, duration_label)),
            alpha = 0.25, position = position_dodge(0.3)) +
  geom_point(size = 3, position = position_dodge(0.3)) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.15, position = position_dodge(0.3)) +
  geom_line(position = position_dodge(0.3)) +
  scale_colour_manual(values = unname(dur_cols), name = "shown") +
  labs(x = "familiarity (rating >= 50 = familiar)", y = "reproduction error (s)",
       title = "Error by familiarity and duration",
       subtitle = "mean +- SE across participants; faint = each participant")
ggsave(file.path(fig_dir, "3b_error_by_familiarity.png"), g3b, width = 6.5, height = 4.5, dpi = 150)

# 4. Familiar vs unfamiliar per duration
s4 <- group_summary(full, c("familiar", "duration_label"))
g4 <- ggplot(s4$group, aes(familiar, mean)) +
  geom_hline(yintercept = 0, colour = "grey50") +
  geom_line(data = s4$participant, aes(y = reproduction_error_s, group = participant),
            colour = "grey60", alpha = 0.6) +
  geom_point(data = s4$participant, aes(y = reproduction_error_s), colour = "grey60", size = 1.5) +
  geom_line(aes(group = 1, colour = duration_label), linewidth = 1.2) +
  geom_point(aes(colour = duration_label), size = 3.5) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se, colour = duration_label), width = 0.1) +
  facet_wrap(~ duration_label) +
  scale_colour_manual(values = unname(dur_cols), guide = "none") +
  labs(x = NULL, y = "reproduction error (s)",
       title = "Familiar (rating >= 50) vs unfamiliar faces",
       subtitle = "colour = mean +- SE across participants; grey = each participant")
ggsave(file.path(fig_dir, "4_familiar_vs_unfamiliar.png"), g4, width = 7, height = 4.5, dpi = 150)

# 5. Familiarity slopes per duration, models 2 and 3
slopes <- rbind(
  data.frame(summary(emtrends(m_fxd, ~ duration_factor, var = "familiarity"), infer = TRUE),
             model = "2. familiarity x duration\n+ condition"),
  data.frame(summary(emtrends(m_ctrl, ~ duration_factor, var = "familiarity"), infer = TRUE),
             model = "3. + condition x duration\n(controls the confound)"))
slopes$duration_label <- factor(dur_labels[as.character(slopes$duration_factor)], levels = dur_labels)
for (v in c("familiarity.trend", "lower.CL", "upper.CL")) slopes[[v]] <- slopes[[v]] * 100 * 1000
g5 <- ggplot(slopes, aes(familiarity.trend, duration_label, colour = duration_label)) +
  geom_vline(xintercept = 0, colour = "grey50") +
  geom_errorbar(aes(xmin = lower.CL, xmax = upper.CL), width = 0.2, orientation = "y") +
  geom_point(size = 3) +
  geom_text(aes(label = sprintf("p = %.3f", p.value)), vjust = -1.2, size = 3.3, colour = "black") +
  facet_wrap(~ model, ncol = 1) +
  scale_colour_manual(values = unname(dur_cols), guide = "none") +
  labs(x = "familiarity effect, rating 0 -> 100 (ms of reproduction error)", y = NULL,
       title = "Familiarity effect per duration (95% CI)")
ggsave(file.path(fig_dir, "5_familiarity_slopes.png"), g5, width = 7, height = 5, dpi = 150)

# 6. Accuracy and precision, familiar vs unfamiliar
ap_long <- rbind(
  data.frame(full[, c("participant", "familiar", "duration_label")], value = full$abs_error,
             measure = "accuracy: |error| (s)"),
  data.frame(full[, c("participant", "familiar", "duration_label")], value = full$cv_resid_trial,
             measure = "precision: CV of residual"))
pm6 <- aggregate(value ~ participant + familiar + duration_label + measure, data = ap_long, FUN = mean)
gm6 <- aggregate(value ~ familiar + duration_label + measure, data = pm6,
                 FUN = function(v) c(mean = mean(v), se = sd(v) / sqrt(length(v))))
gm6 <- do.call(data.frame, gm6)
names(gm6)[names(gm6) == "value.mean"] <- "mean"; names(gm6)[names(gm6) == "value.se"] <- "se"
g6 <- ggplot(gm6, aes(familiar, mean, colour = duration_label, group = duration_label)) +
  geom_line(data = pm6, aes(y = value, group = interaction(participant, duration_label)), alpha = 0.25) +
  geom_point(size = 3.2, position = position_dodge(0.15)) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.1, position = position_dodge(0.15)) +
  geom_line(linewidth = 1.1, position = position_dodge(0.15)) +
  facet_wrap(~ measure, scales = "free_y") +
  scale_colour_manual(values = unname(dur_cols), name = "shown") +
  labs(x = NULL, y = "lower = better",
       title = "Accuracy and precision: familiar (rating >= 50) vs unfamiliar faces",
       subtitle = "mean +- SE across participants; faint = each participant")
ggsave(file.path(fig_dir, "6_accuracy_precision.png"), g6, width = 9, height = 4.5, dpi = 150)

# 7. Familiarity slopes for accuracy and precision
sl7 <- do.call(rbind, lapply(names(acc_prec_fits), function(nm) {
  s <- data.frame(summary(emtrends(acc_prec_fits[[nm]], ~ duration_factor, var = "familiarity"),
                          infer = TRUE))
  s$model <- nm; s
}))
sl7$duration_label <- factor(dur_labels[as.character(sl7$duration_factor)], levels = dur_labels)
kind <- substr(sl7$model, 1, 2)
sl7$measure <- c(A1 = "accuracy: |error| (ms)", A2 = "accuracy: |error| (ms)",
                 A3 = "relative accuracy: |error| (% of shown)", A4 = "relative accuracy: |error| (% of shown)",
                 P1 = "precision: CV (x 100)", P2 = "precision: CV (x 100)",
                 R2 = "relative signed error (% of shown)", R3 = "relative signed error (% of shown)")[kind]
scale7 <- ifelse(kind %in% c("A1", "A2"), 100 * 1000, 100 * 100)   # rating 0 -> 100
for (v in c("familiarity.trend", "lower.CL", "upper.CL")) sl7[[v]] <- sl7[[v]] * scale7
sl7$panel <- paste0(sl7$measure, "\n",
                    ifelse(grepl("controlling|\\+ condition x duration", sl7$model),
                           "controlling condition x duration",
                           ifelse(kind == "R2", "condition as covariate", "as in the previous experiment")))
sl7$panel <- factor(sl7$panel, levels = unique(sl7$panel))
g7 <- ggplot(sl7, aes(familiarity.trend, duration_label, colour = duration_label)) +
  geom_vline(xintercept = 0, colour = "grey50") +
  geom_errorbar(aes(xmin = lower.CL, xmax = upper.CL), width = 0.2, orientation = "y") +
  geom_point(size = 3) +
  geom_text(aes(label = sprintf("p = %.3f", p.value)), vjust = -1.2, size = 3, colour = "black") +
  facet_wrap(~ panel, ncol = 2, scales = "free_x") +
  scale_colour_manual(values = unname(dur_cols), guide = "none") +
  labs(x = "familiarity effect, rating 0 -> 100 (negative = more accurate / precise)", y = NULL,
       title = "Does familiarity change accuracy or precision? (95% CI)")
ggsave(file.path(fig_dir, "7_accuracy_precision_slopes.png"), g7, width = 10, height = 12, dpi = 150)

####################################################
# TIMING SLOPES AND VARIANCE
####################################################

timing_out <- file.path(out_dir, "timing_slope_variance.txt")
sink(timing_out, split = TRUE)
full$shown <- full$face_duration_planned_s

cat("========== 1. Linear fit: reproduced ~ shown, familiar vs unfamiliar ==========\n")
cat("Slope 1 = veridical; < 1 = regression to the mean. With two durations the slope\n")
cat("is (mean at 1.6 s - mean at 0.8 s) / 0.8.\n\n")
fit_slope <- function(formula) {
  m <- tryCatch(lmer(formula, data = full), error = function(e) NULL)
  if (is.null(m) || isSingular(m)) {
    f2 <- update(formula, . ~ . - (1 + shown | participant) + (1 | participant))
    cat("(random slope not estimable -> random intercept only)\n")
    m <- lmer(f2, data = full)
  }
  m
}
m_slope <- fit_slope(reproduced_duration_s ~ shown * familiar + (1 + shown | participant))
print(coef(summary(m_slope)))
cat("\n-- slope per condition --\n")
print(summary(emtrends(m_slope, ~ familiar, var = "shown"), infer = TRUE))
cat("\n-- familiar minus unfamiliar slope (positive = steeper for familiar) --\n")
print(summary(pairs(emtrends(m_slope, ~ familiar, var = "shown"), reverse = TRUE), infer = TRUE))

cat("\n-- same, controlling face category (shown x condition) --\n")
m_slope_c <- fit_slope(reproduced_duration_s ~ shown * familiar + shown * stimuli_condition +
                         (1 + shown | participant))
print(summary(pairs(emtrends(m_slope_c, ~ familiar, var = "shown"), reverse = TRUE), infer = TRUE))

cat("\n-- per-participant slopes (lm per participant and condition) --\n")
pp <- do.call(rbind, lapply(split(full, list(full$participant, full$familiar), drop = TRUE), function(d)
  data.frame(participant = d$participant[1], familiar = d$familiar[1],
             slope = coef(lm(reproduced_duration_s ~ shown, d))[["shown"]],
             intercept = coef(lm(reproduced_duration_s ~ shown, d))[["(Intercept)"]], n = nrow(d))))
pp_wide <- reshape(pp[, c("participant", "familiar", "slope")], idvar = "participant",
                   timevar = "familiar", direction = "wide")
pp_wide$diff <- pp_wide$slope.familiar - pp_wide$slope.unfamiliar
print(pp_wide, row.names = FALSE)
tt <- t.test(pp_wide$slope.familiar, pp_wide$slope.unfamiliar, paired = TRUE)
cat(sprintf("\nPaired t-test (familiar - unfamiliar slope): mean diff = %.3f, t(%d) = %.2f, p = %.3f; steeper for familiar in %d of %d participants\n",
            mean(pp_wide$diff), tt$parameter, tt$statistic, tt$p.value,
            sum(pp_wide$diff > 0), nrow(pp_wide)))

cat("\n\n========== 2. Variance for each interval and face condition ==========\n")
var_tab <- function(by) {
  v <- aggregate(reproduced_duration_s ~ ., data = full[, c("participant", by, "reproduced_duration_s")],
                 FUN = function(x) c(n = length(x), mean = mean(x), sd = sd(x), var = var(x)))
  v <- do.call(data.frame, v)
  names(v) <- sub("reproduced_duration_s\\.", "", names(v))
  v$cv <- v$sd / v$mean
  v
}
vf <- var_tab(c("duration_label", "familiar"))
cat("-- per participant, duration x familiarity --\n"); print(vf, row.names = FALSE, digits = 3)
cat("\n-- mean across participants --\n")
print(aggregate(cbind(var, sd, cv) ~ duration_label + familiar, data = vf, FUN = mean), digits = 3)
cat("\n-- model: log(SD) ~ familiar * duration + (1 | participant) --\n")
m_var <- lmer(log(sd) ~ familiar * duration_label + (1 | participant), data = vf)
print(coef(summary(m_var)))
cat("\n-- familiar vs unfamiliar SD ratio per duration (ratio < 1 = less variable for familiar) --\n")
print(summary(pairs(emmeans(m_var, ~ familiar | duration_label), reverse = TRUE),
              infer = TRUE, type = "response"))
vc <- var_tab(c("duration_label", "stimuli_condition"))
cat("\n-- by face category (mean across participants) --\n")
print(aggregate(cbind(var, sd, cv) ~ duration_label + stimuli_condition, data = vc, FUN = mean), digits = 3)
sink()

# 8. Slopes figure
cond_cols <- c(unfamiliar = "#4C72B0", familiar = "#DD8452")
gm8 <- aggregate(reproduced_duration_s ~ participant + shown + familiar, data = full, FUN = mean)
gm8b <- aggregate(reproduced_duration_s ~ shown + familiar, data = gm8,
                  FUN = function(v) c(mean = mean(v), se = sd(v) / sqrt(length(v))))
gm8b <- do.call(data.frame, gm8b)
names(gm8b) <- sub("reproduced_duration_s\\.", "", names(gm8b))
sl_lab <- as.data.frame(summary(emtrends(m_slope, ~ familiar, var = "shown")))
sl_lab$label <- sprintf("%s: slope = %.2f", sl_lab$familiar, sl_lab$shown.trend)
d_fam <- summary(pairs(emtrends(m_slope, ~ familiar, var = "shown"), reverse = TRUE))
g8a <- ggplot(gm8b, aes(shown, mean, colour = familiar)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_line(data = gm8, aes(y = reproduced_duration_s, group = interaction(participant, familiar)),
            alpha = 0.25) +
  geom_point(size = 3, position = position_dodge(0.04)) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.04, position = position_dodge(0.04)) +
  geom_line(linewidth = 1.1, position = position_dodge(0.04)) +
  annotate("text", x = 0.78, y = max(gm8$reproduced_duration_s) * 0.98, hjust = 0, size = 3.4,
           label = paste0(paste(sl_lab$label, collapse = "\n"),
                          sprintf("\ndifference = %.2f, p = %.3f", d_fam$estimate, d_fam$p.value))) +
  scale_x_continuous(breaks = c(0.8, 1.6), limits = c(0.7, 1.7)) +
  scale_colour_manual(values = cond_cols, name = NULL) +
  labs(x = "shown duration (s)", y = "reproduced duration (s)",
       title = "Reproduced vs shown duration",
       subtitle = "dashed = veridical (slope 1); faint = each participant")
pp$familiar <- factor(pp$familiar, levels = c("unfamiliar", "familiar"))
g8b <- ggplot(pp, aes(familiar, slope)) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "grey50") +
  geom_line(aes(group = participant), colour = "grey60") +
  geom_point(aes(colour = familiar), size = 3) +
  scale_colour_manual(values = cond_cols, guide = "none") +
  labs(x = NULL, y = "slope (reproduced / shown)",
       title = "Per-participant slopes",
       subtitle = sprintf("paired t-test p = %.3f", tt$p.value))
# two panels side by side (base grid, no extra packages)
png(file.path(fig_dir, "8_timing_slopes.png"), width = 11, height = 4.6, units = "in", res = 150)
grid::grid.newpage()
grid::pushViewport(grid::viewport(layout = grid::grid.layout(1, 2, widths = grid::unit(c(1.6, 1), "null"))))
print(g8a, vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 1))
print(g8b, vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 2))
invisible(dev.off())

# 9. Variance figure
vf$familiar <- factor(vf$familiar, levels = c("unfamiliar", "familiar"))
gm9 <- aggregate(sd ~ duration_label + familiar, data = vf,
                 FUN = function(v) c(mean = mean(v), se = sd(v) / sqrt(length(v))))
gm9 <- do.call(data.frame, gm9); names(gm9) <- sub("sd\\.", "", names(gm9))
g9 <- ggplot(gm9, aes(familiar, mean, colour = familiar)) +
  geom_line(data = vf, aes(y = sd, group = participant), colour = "grey60", alpha = 0.6) +
  geom_point(data = vf, aes(y = sd), colour = "grey60", size = 1.5) +
  geom_point(size = 3.5) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.12) +
  facet_wrap(~ duration_label) +
  scale_colour_manual(values = cond_cols, guide = "none") +
  labs(x = NULL, y = "SD of reproduced duration (s)",
       title = "Variability of reproduction for each interval and face condition",
       subtitle = "colour = mean +- SE across participants; grey = each participant")
ggsave(file.path(fig_dir, "9_variance.png"), g9, width = 7.5, height = 4.5, dpi = 150)

####################################################
# THREE FACE GROUPS: baseline (neutral) / unfamiliar celebrities / familiar celebrities
####################################################

grp_levels <- c("baseline", "unfamiliar", "familiar")
full$face_group <- factor(ifelse(full$stimuli_condition == "Neutral", "baseline",
                                 as.character(full$familiar)), levels = grp_levels)
grp_cols <- c(baseline = "grey45", unfamiliar = "#4C72B0", familiar = "#DD8452")

sink(file.path(out_dir, "timing_slope_variance_3groups.txt"), split = TRUE)
cat("Face groups: baseline = neutral (database) faces; unfamiliar / familiar = Israeli and UK\n")
cat("celebrities rated < 50 / >= 50 by the participant.\n\n")
print(table(full$participant, full$face_group))

cat("\n========== 1. Linear fit: reproduced ~ shown, by face group ==========\n")
m_slope3 <- fit_slope(reproduced_duration_s ~ shown * face_group + (1 + shown | participant))
print(coef(summary(m_slope3)))
cat("\n-- slope per group --\n")
print(summary(emtrends(m_slope3, ~ face_group, var = "shown"), infer = TRUE))
cat("\n-- pairwise slope differences (later minus earlier group; positive = steeper) --\n")
print(summary(pairs(emtrends(m_slope3, ~ face_group, var = "shown"), reverse = TRUE, adjust = "none"),
              infer = TRUE))

cat("\n-- per-participant slopes --\n")
pp3 <- do.call(rbind, lapply(split(full, list(full$participant, full$face_group), drop = TRUE), function(d)
  data.frame(participant = d$participant[1], face_group = d$face_group[1], n = nrow(d),
             slope = if (length(unique(d$shown)) == 2) coef(lm(reproduced_duration_s ~ shown, d))[["shown"]] else NA)))
pp3_wide <- reshape(pp3[, c("participant", "face_group", "slope")], idvar = "participant",
                    timevar = "face_group", direction = "wide")
print(pp3_wide, row.names = FALSE, digits = 3)

cat("\n\n========== 2. Variance for each interval and face group ==========\n")
vg <- var_tab(c("duration_label", "face_group"))
print(vg, row.names = FALSE, digits = 3)
cat("\n-- mean across participants --\n")
print(aggregate(cbind(var, sd, cv) ~ duration_label + face_group, data = vg, FUN = mean), digits = 3)
cat("\n-- model: log(SD) ~ face_group * duration + (1 | participant) --\n")
m_var3 <- lmer(log(sd) ~ face_group * duration_label + (1 | participant), data = vg)
print(coef(summary(m_var3)))
cat("\n-- SD ratios between groups per duration (ratio < 1 = less variable than the comparison) --\n")
print(summary(pairs(emmeans(m_var3, ~ face_group | duration_label), reverse = TRUE, adjust = "none"),
              infer = TRUE, type = "response"))
sink()

# 8b. Slopes, three groups
gm8c <- aggregate(reproduced_duration_s ~ participant + shown + face_group, data = full, FUN = mean)
gm8d <- aggregate(reproduced_duration_s ~ shown + face_group, data = gm8c,
                  FUN = function(v) c(mean = mean(v), se = sd(v) / sqrt(length(v))))
gm8d <- do.call(data.frame, gm8d); names(gm8d) <- sub("reproduced_duration_s\\.", "", names(gm8d))
sl3 <- as.data.frame(summary(emtrends(m_slope3, ~ face_group, var = "shown")))
g8c <- ggplot(gm8d, aes(shown, mean, colour = face_group)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60") +
  geom_point(size = 3, position = position_dodge(0.06)) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.05, position = position_dodge(0.06)) +
  geom_line(linewidth = 1.1, position = position_dodge(0.06)) +
  annotate("text", x = 0.75, y = max(gm8d$mean + gm8d$se) * 1.02, hjust = 0, vjust = 1, size = 3.4,
           label = paste(sprintf("%s: slope = %.2f", sl3$face_group, sl3$shown.trend), collapse = "\n")) +
  scale_x_continuous(breaks = c(0.8, 1.6), limits = c(0.7, 1.7)) +
  scale_colour_manual(values = grp_cols, name = NULL) +
  labs(x = "shown duration (s)", y = "reproduced duration (s)",
       title = "Reproduced vs shown duration, three face groups",
       subtitle = "mean +- SE across participants; dashed = veridical")
pp3$face_group <- factor(pp3$face_group, levels = grp_levels)
g8d <- ggplot(pp3[!is.na(pp3$slope), ], aes(face_group, slope)) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "grey60") +
  geom_line(aes(group = participant), colour = "grey65") +
  geom_point(aes(colour = face_group), size = 3) +
  scale_colour_manual(values = grp_cols, guide = "none") +
  labs(x = NULL, y = "slope (reproduced / shown)", title = "Per-participant slopes")
png(file.path(fig_dir, "8b_timing_slopes_3groups.png"), width = 11.5, height = 4.6, units = "in", res = 150)
grid::grid.newpage()
grid::pushViewport(grid::viewport(layout = grid::grid.layout(1, 2, widths = grid::unit(c(1.6, 1), "null"))))
print(g8c, vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 1))
print(g8d, vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 2))
invisible(dev.off())

# 9b. Variance, three groups
vg$face_group <- factor(vg$face_group, levels = grp_levels)
gm9b <- aggregate(sd ~ duration_label + face_group, data = vg,
                  FUN = function(v) c(mean = mean(v), se = sd(v) / sqrt(length(v))))
gm9b <- do.call(data.frame, gm9b); names(gm9b) <- sub("sd\\.", "", names(gm9b))
g9b <- ggplot(gm9b, aes(face_group, mean, colour = face_group)) +
  geom_line(data = vg, aes(y = sd, group = participant), colour = "grey65", alpha = 0.7) +
  geom_point(data = vg, aes(y = sd), colour = "grey65", size = 1.5) +
  geom_point(size = 3.5) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.12) +
  facet_wrap(~ duration_label) +
  scale_colour_manual(values = grp_cols, guide = "none") +
  labs(x = NULL, y = "SD of reproduced duration (s)",
       title = "Variability of reproduction, three face groups",
       subtitle = "colour = mean +- SE across participants; grey = each participant")
ggsave(file.path(fig_dir, "9b_variance_3groups.png"), g9b, width = 8.5, height = 4.5, dpi = 150)

# 10. Reproduction error by familiarity-rating bins of 10
bin_edges <- seq(0, 100, 10)
bin_labels <- paste0(head(bin_edges, -1), "-", bin_edges[-1])
full$fam_bin <- cut(full$familiarity, bin_edges, labels = bin_labels, include.lowest = TRUE)
cell_mean <- ave(full$reproduction_error_s, interaction(full$participant, full$duration_factor, drop = TRUE))
full$error_centered <- full$reproduction_error_s - cell_mean

bin_summary <- function(col) {
  s <- aggregate(full[[col]], by = list(duration_label = full$duration_label, fam_bin = full$fam_bin),
                 FUN = function(v) c(mean = mean(v), se = if (length(v) > 1) sd(v) / sqrt(length(v)) else NA,
                                     n = length(v)))
  s <- do.call(data.frame, s)
  names(s) <- sub("^x\\.", "", names(s))
  s
}
plot_bins <- function(col, ylab, title, file) {
  s <- bin_summary(col)
  pm <- aggregate(full[[col]], by = list(participant = full$participant, duration_label = full$duration_label,
                                         fam_bin = full$fam_bin), FUN = mean)
  g <- ggplot(s, aes(fam_bin, mean, group = 1)) +
    geom_hline(yintercept = 0, colour = "grey40") +
    geom_line(data = pm, aes(y = x, group = participant), colour = "grey60", alpha = 0.35) +
    geom_line(aes(colour = duration_label), linewidth = 1.1) +
    geom_point(aes(colour = duration_label), size = 2.6) +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se, colour = duration_label), width = 0.2) +
    geom_text(aes(label = paste0("n=", n)), vjust = -1.4, size = 2.6, colour = "grey30") +
    facet_wrap(~ duration_label) +
    scale_colour_manual(values = unname(dur_cols), guide = "none") +
    scale_x_discrete(drop = FALSE) +
    labs(x = "familiarity rating", y = ylab, title = title,
         subtitle = sprintf("all participants (N = %d), mean +- SE over trials; grey = each participant",
                            nlevels(full$participant))) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  ggsave(file.path(fig_dir, file), g, width = 11, height = 4.6, dpi = 150)
  s
}
b_raw <- plot_bins("reproduction_error_s", "reproduction error (s)\n(reproduced - shown)",
                   "Reproduction error by familiarity", "10_error_by_familiarity_bins.png")
b_cen <- plot_bins("error_centered", "reproduction error, centered\nwithin participant (s)",
                   "Reproduction error by familiarity (participant mean removed)",
                   "10_error_by_familiarity_bins_centered.png")
bins_tab <- merge(b_raw, b_cen[, c("duration_label", "fam_bin", "mean")],
                  by = c("duration_label", "fam_bin"), suffixes = c("_error_s", "_error_centered_s"))
write.csv(bins_tab[order(bins_tab$duration_label, bins_tab$fam_bin), ],
          file.path(out_dir, "error_by_familiarity_bins.csv"), row.names = FALSE)

cat("Figures saved in", fig_dir, "\n")
