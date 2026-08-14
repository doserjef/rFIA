# Test maWeights() and filterAnnual() ---------------------------------------
# Non-TI (SMA/LMA/EMA/ANNUAL) method-weighting internals shared by every
# sumToEU()-based estimator (all exported estimators except fsi(), which
# reimplements this branching independently in R/fsiHelper.R). These are
# pure-computation unit tests on small synthetic tibbles built to match the
# real column shapes handlePops()/sumToEU() produce (verified against a real
# tpa(fiaRI, method = 'ANNUAL') capture) -- no FIA data cache or network is
# required, so no skip_on_cran()/skip_if_not() gating.

# maWeights() ------------------------------------------------------------

test_that("SMA weights are uniform (1 / n panels)", {
  pops <- tibble::tibble(YEAR = 2020, STATECD = 1, INVYR = c(2018, 2019, 2020))
  wgts <- rFIA:::maWeights(pops, "SMA", NULL)
  expect_equal(wgts$wgt, 1 / 3)
})

test_that("LMA weights increase linearly with panel recency and sum to 1", {
  pops <- tibble::tibble(YEAR = 2020, STATECD = 1, INVYR = c(2018, 2019, 2020))
  wgts <- rFIA:::maWeights(pops, "LMA", NULL)
  wgts <- wgts[order(wgts$INVYR), ]
  # Closed form: wgt_i = rank_i / (n*(n+1)/2)
  expect_equal(wgts$wgt, c(2018, 2019, 2020) |> seq_along() / 6)
  expect_true(all(diff(wgts$wgt) > 0))
  expect_equal(sum(wgts$wgt), 1)

  # 5-panel case
  pops5 <- tibble::tibble(YEAR = 2020, STATECD = 1, INVYR = 2016:2020)
  wgts5 <- rFIA:::maWeights(pops5, "LMA", NULL)
  wgts5 <- wgts5[order(wgts5$INVYR), ]
  expect_equal(wgts5$wgt, (1:5) / 15)
  expect_equal(sum(wgts5$wgt), 1)
})

test_that("EMA weights (default lambda = 0.5) decay geometrically and sum to 1", {
  pops <- tibble::tibble(YEAR = 2020, STATECD = 1, INVYR = c(2018, 2019, 2020))
  wgts <- rFIA:::maWeights(pops, "EMA", 0.5)
  wgts <- wgts[order(wgts$INVYR), ]
  expect_equal(wgts$wgt, c(1, 2, 4) / 7, tolerance = 1e-6)
  expect_true(all(diff(wgts$wgt) > 0))
  expect_equal(sum(wgts$wgt), 1)
})

test_that("EMA weights sum to 1 and decay monotonically across a lambda grid", {
  pops <- tibble::tibble(YEAR = 2020, STATECD = 1, INVYR = c(2018, 2019, 2020))
  for (lam in c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
    wgts <- rFIA:::maWeights(pops, "EMA", lam)
    wgts <- wgts[order(wgts$INVYR), ]
    expect_equal(sum(wgts$wgt), 1, tolerance = 1e-8, label = paste0("lambda=", lam))
    expect_true(all(diff(wgts$wgt) > 0), label = paste0("lambda=", lam))
  }
})

test_that("EMA(lambda -> 1) monotonically approaches SMA (never exactly equal)", {
  pops <- tibble::tibble(YEAR = 2020, STATECD = 1, INVYR = c(2018, 2019, 2020))
  sma_wgt <- 1 / 3
  dists <- sapply(c(0.5, 0.9, 0.99, 0.999), \(lam) {
    wgts <- rFIA:::maWeights(pops, "EMA", lam)
    max(abs(wgts$wgt - sma_wgt))
  })
  # Distance to the uniform SMA weight should shrink monotonically as
  # lambda -> 1, per vignettes/alternativeEstimators.Rmd:28.
  expect_true(all(diff(dists) < 0))
  expect_lt(dists[length(dists)], 0.01)
})

test_that("EMA(lambda -> 0) concentrates weight on the most recent panel", {
  pops <- tibble::tibble(YEAR = 2020, STATECD = 1, INVYR = c(2018, 2019, 2020))
  wgts <- rFIA:::maWeights(pops, "EMA", 0.01)
  wgts <- wgts[order(wgts$INVYR), ]
  expect_gt(wgts$wgt[wgts$INVYR == 2020], 0.98)
  expect_equal(sum(wgts$wgt), 1, tolerance = 1e-6)
})

test_that("EMA rejects out-of-range or boundary lambda with a clear error", {
  # Originally (see git history / tpa.md "Findings" #1) lambda had no range
  # check anywhere in the package, despite man/tpa.Rd documenting it as
  # numeric (0,1): the exact boundaries (0 or 1) made the weighting formula
  # 0/0 = NaN for most or all panels, and out-of-range values produced a
  # negative weight or an inverted recency ordering rather than erroring.
  # maWeights() now validates lambda up front instead of silently returning
  # degenerate weights.
  pops <- tibble::tibble(YEAR = 2020, STATECD = 1, INVYR = c(2018, 2019, 2020))
  for (lam in c(0, 1, -0.5, 1.5, NA_real_)) {
    expect_error(rFIA:::maWeights(pops, "EMA", lam), "lambda",
                 label = paste0("lambda=", lam))
  }
})

test_that("EMA vector-lambda (ribbon mode) weights sum to 1 per lambda and match scalar mode exactly", {
  pops <- tibble::tibble(YEAR = 2020, STATECD = 1, INVYR = c(2018, 2019, 2020))
  lambdas <- c(0.2, 0.5, 0.8)
  wgts_vec <- rFIA:::maWeights(pops, "EMA", lambdas)

  expect_equal(nrow(wgts_vec), 3 * length(lambdas))
  for (lam in lambdas) {
    sub <- wgts_vec[wgts_vec$lambda == lam, ]
    expect_equal(sum(sub$wgt), 1, tolerance = 1e-8, label = paste0("lambda=", lam))
  }

  # A single lambda pulled from vector mode must match a scalar-mode call
  # with that same lambda -- these are two separately-coded branches
  # (util.R:1121 vs util.R:1136) implementing the same formula.
  wgts_scalar <- rFIA:::maWeights(pops, "EMA", 0.5)
  sub_vec <- wgts_vec[wgts_vec$lambda == 0.5, c("YEAR", "STATECD", "INVYR", "wgt")]
  sub_vec <- sub_vec[order(sub_vec$INVYR), ]
  wgts_scalar <- wgts_scalar[order(wgts_scalar$INVYR), ]
  expect_equal(sub_vec$wgt, wgts_scalar$wgt)
})

# filterAnnual() -----------------------------------------------------------
# Synthetic xEU/ESTN_UNIT shaped to match a real captured call (verified via
# `assignInNamespace()` tracing during `tpa(fiaRI, method = 'ANNUAL')`):
# ESTN_UNIT_CN is numeric (large FIA `CN` identifiers), not character, and
# grpBy always includes 'YEAR' at this point in the pipeline.

test_that("filterAnnual aggregates a natural (INVYR == YEAR) panel across estimation units within a state", {
  ESTN_UNIT <- tibble::tibble(CN = c(1e14, 2e14), STATECD = c(1, 1))
  xEU <- tibble::tribble(
    ~ESTN_UNIT_CN, ~YEAR, ~INVYR, ~nPlots.x, ~val_mean,
    1e14,          2018,  2018,   5,         100,
    2e14,          2018,  2018,   5,         50
  )
  out <- rFIA:::filterAnnual(xEU, "YEAR", nPlots.x, ESTN_UNIT)
  expect_equal(nrow(out), 1)
  expect_equal(out$YEAR, 2018)
  expect_equal(out$val_mean, 150)
  expect_equal(out$nPlots.x, 10)
})

# A single panel (INVYR) is often a constituent of more than one FIA
# evaluation's multi-panel window -- e.g. RI's real POP_EVAL table has a
# 2013 evaluation covering panels 2009-2013, and a 2014 evaluation covering
# panels 2009-2014, so panel 2009 is "hosted" by both (confirmed directly
# against RI's local FIADB extract, not assumed). filterAnnual()'s job is
# to pick, for each *panel*, the single best hosting eval to draw its
# standalone estimate from -- not to pick, for each *reporting year*, the
# best candidate panel (an earlier, incorrect reading of this function; see
# git history). YEAR must therefore be EXCLUDED from the comparison group
# (comparison happens across YEAR values for a fixed INVYR), and the output
# is relabeled YEAR = INVYR, since method = 'ANNUAL' reports one row per
# actually-sampled panel-year, not one row per evaluation.

test_that("filterAnnual picks the highest-nplts hosting eval when a panel has no self-hosting eval", {
  # Panel INVYR = 2009 has no eval of its own (no candidate row has
  # YEAR == 2009) but is hosted by both the 2013 eval (fewer plots) and the
  # 2014 eval (more plots) -- mirrors RI's real eval structure exactly.
  ESTN_UNIT <- tibble::tibble(CN = 1e14, STATECD = 1)
  xEU <- tibble::tribble(
    ~ESTN_UNIT_CN, ~YEAR, ~INVYR, ~nPlots.x, ~val_mean,
    1e14,          2013,  2009,   20,        100,  # panel 2009, hosted by the 2013 eval
    1e14,          2014,  2009,   30,        150,  # panel 2009, hosted by the 2014 eval (more plots)
    1e14,          2013,  2013,   50,        500   # panel 2013's own self-hosting eval
  )
  out <- rFIA:::filterAnnual(xEU, "YEAR", nPlots.x, ESTN_UNIT)
  out <- out[order(out$YEAR), ]

  expect_equal(nrow(out), 2)
  expect_equal(out$YEAR, c(2009, 2013))
  expect_equal(out$INVYR, c(2009, 2013))
  expect_equal(out$val_mean, c(150, 500)) # 2009 draws from its higher-nplts (2014) hosting
})

test_that("filterAnnual prefers a panel's self-hosting eval over a higher-nplts non-self-hosting one", {
  ESTN_UNIT <- tibble::tibble(CN = 1e14, STATECD = 1)
  xEU <- tibble::tribble(
    ~ESTN_UNIT_CN, ~YEAR, ~INVYR, ~nPlots.x, ~val_mean,
    1e14,          2013,  2009,   999,       999,  # huge nplts, but NOT self-hosting
    1e14,          2009,  2009,   5,         111   # tiny nplts, but self-hosting
  )
  out <- rFIA:::filterAnnual(xEU, "YEAR", nPlots.x, ESTN_UNIT)

  expect_equal(nrow(out), 1)
  expect_equal(out$YEAR, 2009)
  expect_equal(out$val_mean, 111)
})

test_that("filterAnnual aggregates estimation-unit-level estimates before comparing hosting evals", {
  # Two ESTN_UNITs, each contributing to both of panel 2009's candidate
  # hosting evals -- the state-level nplts/val_mean totals (summed across
  # ESTN_UNITs) are what should decide which hosting eval wins, not any
  # single ESTN_UNIT's individual figures.
  ESTN_UNIT <- tibble::tibble(CN = c(1e14, 2e14), STATECD = c(1, 1))
  xEU <- tibble::tribble(
    ~ESTN_UNIT_CN, ~YEAR, ~INVYR, ~nPlots.x, ~val_mean,
    1e14,          2013,  2009,   10,        50,
    2e14,          2013,  2009,   10,        50,   # 2013-hosted total: nplts=20, val=100
    1e14,          2014,  2009,   8,         40,
    2e14,          2014,  2009,   8,         40    # 2014-hosted total: nplts=16, val=80
  )
  out <- rFIA:::filterAnnual(xEU, "YEAR", nPlots.x, ESTN_UNIT)

  expect_equal(nrow(out), 1)
  expect_equal(out$YEAR, 2009)
  expect_equal(out$nPlots.x, 20) # 2013-hosted total (20) beats 2014-hosted total (16)
  expect_equal(out$val_mean, 100)
})

test_that("filterAnnual breaks an exact nplts tie deterministically (first row per group)", {
  ESTN_UNIT <- tibble::tibble(CN = 1e14, STATECD = 1)
  xEU <- tibble::tribble(
    ~ESTN_UNIT_CN, ~YEAR, ~INVYR, ~nPlots.x, ~val_mean,
    1e14,          2013,  2009,   10,        111,
    1e14,          2014,  2009,   10,        222  # identical nplts -- genuine tie
  )
  out <- rFIA:::filterAnnual(xEU, "YEAR", nPlots.x, ESTN_UNIT)
  expect_equal(nrow(out), 1)
  expect_equal(out$YEAR, 2009)
  expect_equal(out$val_mean, 111) # first row in the tie wins
})

test_that("filterAnnual is not fooled by a garbage YEAR = NA hosting candidate", {
  # An incomplete estimation-unit row with YEAR = NA (an unrelated
  # upstream join-completeness artifact, confirmed present in real RI data)
  # must not cause `INVYR %in% YEAR` to return NA and silently drop the
  # panel's real candidates (this was a regression introduced and caught
  # while implementing this fix -- an earlier version used `any(INVYR ==
  # YEAR)`, which is not NA-safe).
  ESTN_UNIT <- tibble::tibble(CN = 1e14, STATECD = 1)
  xEU <- tibble::tribble(
    ~ESTN_UNIT_CN, ~YEAR, ~INVYR, ~nPlots.x, ~val_mean,
    1e14,          2013,  2009,   20,        100,
    1e14,          2014,  2009,   30,        150,
    1e14,          NA,    2009,   1,         NA    # garbage row
  )
  out <- rFIA:::filterAnnual(xEU, "YEAR", nPlots.x, ESTN_UNIT)
  out <- out[!is.na(out$YEAR), ]

  expect_equal(nrow(out), 1)
  expect_equal(out$YEAR, 2009)
  expect_equal(out$val_mean, 150) # still correctly picks the higher-nplts real candidate
})
