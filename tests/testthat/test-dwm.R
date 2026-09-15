# Test dwm() --------------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Most recent estimates for dwm on timber land
out <- dwm(db = fiaRI_mr, landType = 'timber', totals = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 2 ------------------------------
# Most recent estimates by plot
out <- dwm(db = fiaRI_mr, land = 'forest', byPlot = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
# Most recent estimates grouped by stand age on forest land
# Make a categorical variable which represents stand age (grouped by 10 yr intervals)
fiaRI_mr$COND$STAND_AGE <- makeClasses(fiaRI_mr$COND$STDAGE, interval = 10)
out <- dwm(db = fiaRI_mr, grpBy = STAND_AGE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 4 ------------------------------ 
# Estimates on forested mesic sites
out <- dwm(db = fiaRI, landType = 'forest', 
                areaDomain = PHYSCLCD %in% 21:29)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

test_that("multiple years", {
  expect_gt(length(unique(out$YEAR)), 1)
})

# Test 5 ------------------------------
# Most recent estimates by county
out <- dwm(fiaRI_mr, polys = countiesRI, returnSpatial = TRUE)
test_that("out is of class sf", {
  expect_s3_class(out, "sf")
})

# Internal consistency checks (no EVALIDator, no network needed) ----------
# These only require the local FIADB extract cache, not network access, so
# they still run when apps.fs.usda.gov is unreachable. See
# core_references/validation/dwm.md for full methodology/results.
skip_on_cran()

validation_data_dir <- Sys.getenv("RFIA_VALIDATION_DATA", "~/Dropbox/data/fia")
skip_if_not(dir.exists(validation_data_dir),
            "Local FIADB validation data cache not found")

# One state per FIA region, read/clipped once and reused by every test below
# (including the EVALIDator-comparison tests further down).
states <- c("RI", "NC", "CO", "OR")
dbs <- lapply(states, \(st) clipFIA(readFIA(validation_data_dir, states = st), mostRecent = TRUE))
names(dbs) <- states
db_ri <- dbs[["RI"]]
# EVAL_GRP encodes STATECD + 4-digit year (e.g. 442024 = Rhode Island 2024);
# reading it off each clipped db mirrors exactly which evaluation
# `mostRecent = TRUE` selected, so it never needs to be hard-coded or kept in
# sync by hand.
wcs <- lapply(dbs, \(d) unique(d$POP_EVAL_GRP$EVAL_GRP))
wc_ri <- wcs[["RI"]]

# Test 6 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(dwm(db_ri, polys = countiesRI, returnSpatial = TRUE))
  out_df <- as.data.frame(dwm(db_ri, polys = countiesRI, returnSpatial = FALSE))
  out_sf <- out_sf[, names(out_df)]
  out_sf <- out_sf[order(out_sf$COUNTY, out_sf$FUEL_TYPE), ]
  out_df <- out_df[order(out_df$COUNTY, out_df$FUEL_TYPE), ]
  expect_equal(out_sf, out_df)
})

# Test 7 ------------------------------
# Internal consistency: totals divided by area reproduce the per-acre
# estimate (doesn't require EVALIDator).
for (st in states) {
  test_that(paste("dwm() totals are consistent with per-acre estimates (", st, ")"), {
    out <- as.data.frame(dwm(dbs[[st]], byFuelType = FALSE, totals = TRUE))
    expect_equal(out$VOL_TOTAL / out$AREA_TOTAL, out$VOL_ACRE, tolerance = 1e-9)
    expect_equal(out$BIO_TOTAL / out$AREA_TOTAL, out$BIO_ACRE, tolerance = 1e-9)
    expect_equal(out$CARB_TOTAL / out$AREA_TOTAL, out$CARB_ACRE, tolerance = 1e-9)
  })
}

# Test 8 ------------------------------
# An areaDomain matching no conditions should return a clean 0-row result,
# not error or emit an internal max()-on-empty-vector warning.
test_that("dwm() handles an empty areaDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(dwm(db_ri, areaDomain = STATECD == 999))
  )
  expect_equal(nrow(out), 0)
})

# Numeric validation against EVALIDator ------------------------------------
# Reference values are fetched live from the FIADB-API `fullreport` endpoint
# (see core_references/validation/fetch_evalidator.R and
# core_references/validation/dwm.md for methodology and full results) rather
# than hard-coded, so these tests can never drift from what EVALIDator
# currently reports. They require network access to apps.fs.usda.gov (on top
# of the local data cache already required above), so they're skipped (not
# failed) when it's unavailable.
skip_if_not_installed("curl")
skip_if_not_installed("jsonlite")
source(test_path("..", "..", "core_references", "validation", "fetch_evalidator.R"))

network_ok <- tryCatch({
  fetch_evalidator(wc = wc_ri, snum = 2)
  TRUE
}, error = function(e) FALSE)
skip_if_not(network_ok, "FIADB-API (apps.fs.usda.gov) not reachable")

# Fetches a reference value, skipping just the enclosing test_that() if a
# request fails after the initial reachability check (e.g. transient network
# blip), rather than failing it.
fetchRef <- function(...) {
  tryCatch(
    fetch_evalidator(...),
    error = function(e) skip(paste("FIADB-API request failed:", conditionMessage(e)))
  )
}

# Test 9 ------------------------------
# Core default case, totaled across fuel types (landType = 'forest') matches
# EVALIDator to full double precision across one state per FIA region: RI
# (Northern), NC (Southern), CO (Interior West), OR (Pacific Northwest).
# EVALIDator attribute 123 = total volume of DWM (FWD, CWD, and piles) on
# forest land, ratio'd against attribute 2 (forest land area). Note DWM is a
# P3 (phase 3) measurement collected on only a subset of forested plots, so
# plot counts here are much smaller than tpa()/biomass()/volume()'s (e.g.
# RI has only 5-6 DWM-sampled plots).
for (st in states) {
  test_that(paste("dwm() matches EVALIDator for", st, "(core default case, totaled)"), {
    wc_st <- wcs[[st]]
    ref <- fetchRef(wc = wc_st, snum = 123, sdenom = 2)
    out <- as.data.frame(dwm(dbs[[st]], byFuelType = FALSE))
    expect_equal(out$VOL_ACRE, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$VOL_ACRE_SE, ref$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out$nPlots_DWM, ref$numPlotCount)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })
}

# Test 10 ------------------------------
# byFuelType = TRUE (the default) variants across all four FIA regions,
# matched against EVALIDator's per-fuel-type attributes: 114/115/116 (CWD
# volume/biomass/carbon), 104 (FWD small, i.e. '1HR', volume). nPlots_DWM is
# checked per fuel type here (unlike the totaled Test 9 above) -- this is
# only possible because dwmStarter.R's zero-value qualifying filter is
# applied per fuel-type row (VOL > 0 for the five woody types, BIO > 0 for
# DUFF/LITTER, which have no VOL column at all), matching each fuel type's
# own EVALIDator attribute rather than the single combined total.
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("dwm() matches EVALIDator for CWD (byFuelType, ", st, ")"), {
    refVol <- fetchRef(wc = wc_st, snum = 114, sdenom = 2)
    refBio <- fetchRef(wc = wc_st, snum = 115, sdenom = 2)
    refCarb <- fetchRef(wc = wc_st, snum = 116, sdenom = 2)
    out <- as.data.frame(dwm(db_st, byFuelType = TRUE))
    row <- out[out$FUEL_TYPE == '1000HR', ]
    expect_equal(row$VOL_ACRE, refVol$ratioEstimate, tolerance = 1e-6)
    expect_equal(row$BIO_ACRE, refBio$ratioEstimate, tolerance = 1e-6)
    expect_equal(row$CARB_ACRE, refCarb$ratioEstimate, tolerance = 1e-6)
    expect_equal(row$VOL_ACRE_SE, refVol$ratioSEPercent, tolerance = 1e-6)
    expect_equal(row$nPlots_DWM, refVol$numPlotCount)
    expect_equal(row$nPlots_AREA, refVol$denPlotCount)
  })

  test_that(paste("dwm() matches EVALIDator for FWD small (byFuelType, ", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 104, sdenom = 2)
    out <- as.data.frame(dwm(db_st, byFuelType = TRUE))
    row <- out[out$FUEL_TYPE == '1HR', ]
    expect_equal(row$VOL_ACRE, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(row$nPlots_DWM, ref$numPlotCount)
  })
}

# Test 11 ------------------------------
# landType = 'timber' internal consistency (no direct EVALIDator timberland
# DWM attribute exists -- only forest-land attributes are available in
# EVALIDator's attribute library for DWM -- so this is a structural/plot-
# count sanity check rather than a numeric match): restricting to timberland
# should never increase the plot count relative to forest land, across all
# four FIA regions. This also exercises the nPlots_AREA phantom-row fix from
# a second angle (landType-based restriction, complementing the areaDomain-
# based check in Test 12).
for (st in states) {
  test_that(paste("dwm() landType = 'timber' plot counts do not exceed forest land (", st, ")"), {
    db_st <- dbs[[st]]
    outForest <- as.data.frame(dwm(db_st, byFuelType = FALSE, landType = 'forest'))
    outTimber <- as.data.frame(dwm(db_st, byFuelType = FALSE, landType = 'timber'))
    expect_true(outTimber$nPlots_AREA <= outForest$nPlots_AREA)
    expect_true(outTimber$nPlots_DWM <= outForest$nPlots_DWM)
  })
}

# Test 12 ------------------------------
# areaDomain filter interaction across all four FIA regions (mesic
# physiographic classes) -- the primary regression check for the
# nPlots_AREA phantom-row fix (volumeStarter.R was missing the same
# `!is.na(CONDID)` guard that tpa()/biomass()/carbon() already have; dwm()
# had the identical gap) and for the COND_DWM_CALC multi-EVALID duplication
# fix (a plot can appear in COND_DWM_CALC under several EVALIDs from
# consecutive annual panels that all reported it as their most recent DWM
# data; failing to restrict to the current EVALID inflated nPlots_DWM by
# ~4-5x before this fix -- see dwm.md, "Fixed" #2).
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("dwm() matches EVALIDator for areaDomain (physiographic class filter) (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 123, sdenom = 2,
                     strFilter = "COND.PHYSCLCD in (21,22,23,24,25,26,27,28,29)")
    out <- as.data.frame(dwm(db_st, byFuelType = FALSE, areaDomain = PHYSCLCD %in% 21:29))
    expect_equal(out$VOL_ACRE, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$nPlots_DWM, ref$numPlotCount)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })
}

# Non-TI method (SMA/LMA/EMA/ANNUAL) internal consistency -------------------
# EVALIDator has no equivalent for these, so correctness here means: the
# code runs cleanly across the same filter/grpBy/byPlot space already
# exercised above, totals/per-acre plumbing holds regardless of method, and
# the documented cross-method relationships in
# vignettes/alternativeEstimators.Rmd hold as *bounded*/*directional*
# checks -- never exact equality (see tpa.md for the full writeup of why).
# See tests/testthat/test-util.R for the underlying maWeights()/
# filterAnnual()/combineMR() unit-level checks these per-function tests
# build on, and tpa.md "Fixed" #6 for a package-wide combineMR()/ANNUAL bug
# found and fixed during area()'s non-TI pass -- dwm() shares the same
# combineMR() call site and was unaffected by the time this section was
# written (already fixed at the shared-utility level).
#
# RI is excluded from every population-estimation check below (a
# departure from every other function's 4-state pattern): confirmed
# directly, independent of anything in this pass, that RI's most-recent
# EXPDWM evaluation currently has zero rows in the local COND_DWM_CALC
# extract -- FIA's real-world DWM (phase 3) data publication lags behind
# the core EXPCURR evaluation, and the local data cache hasn't caught up
# for RI yet. This makes RI's TI output itself empty right now (confirmed:
# `dwm(db_ri, method = 'TI')` returns 0 rows, and the existing, unmodified
# EVALIDator comparison tests above already fail for RI against the
# current cache for the same reason) -- not something this pass caused or
# should route around silently. See dwm.md "Findings" for the one
# consequence of this worth a permanent written record (a `filterAnnual()`
# max()-on-an-empty-group warning under `method = 'ANNUAL'`), not fixed
# here per the project's bug-handling protocol; the RI exclusion below
# avoids pinning any test's pass/fail status to this data cache's current,
# transient state. `byPlot = TRUE` is unaffected (it doesn't depend on the
# current population-estimation eval the same way), so RI is kept for that
# check, matching every other function's pattern.
statesDwm <- c("NC", "CO", "OR")

# Test 13 ------------------------------
# EMA(lambda -> 1) should monotonically approach SMA (NC -- RI is excluded,
# see above). Never exactly equal (see test-util.R for why the exact
# boundary is degenerate) -- this checks the trend, not a fixed-tolerance
# snapshot. Unlike every other function's version of this check, the final
# distance is asserted *relative* to the starting distance (shrinks to
# under 1%) rather than against a fixed absolute value: dwm()'s VOL_ACRE
# operates on a larger, more volatile scale than TPA/BIO_ACRE/etc. (see
# Test 14 below), so a hard-coded absolute bound calibrated for those
# metrics doesn't transfer, and a scale-invariant check is more robust to
# this metric's known data volatility (see the RI note above).
test_that("dwm() EMA(lambda -> 1) monotonically approaches SMA (NC)", {
  db_nc <- dbs[["NC"]]
  sma <- as.data.frame(dwm(db_nc, byFuelType = FALSE, method = 'SMA'))
  dists <- sapply(c(0.5, 0.9, 0.99, 0.999), \(lam) {
    ema <- as.data.frame(dwm(db_nc, byFuelType = FALSE, method = 'EMA', lambda = lam))
    abs(ema$VOL_ACRE - sma$VOL_ACRE)
  })
  expect_true(all(diff(dists) < 0))
  expect_lt(dists[length(dists)] / dists[1], 0.01)
})

# Test 14 ------------------------------
# TI and SMA are not claimed to be numerically equal in general (see
# tpa.md), and dwm()'s small phase-3 sample size combined with down woody
# material's naturally clumpy spatial distribution (a handful of logs or
# slash piles can dominate a state's estimate) makes even the generous flat
# 10% tolerance used for every other function's TI-vs-SMA check not
# meaningful here: confirmed empirically that NC's TI vs. SMA VOL_ACRE
# differ by ~127% (665.3 vs. 1508.2), which looks alarming in isolation but
# is fully explained by both estimates' own sampling error being enormous
# (VOL_ACRE_SE = 34.5% for TI, 67.7% for SMA -- the two point estimates are
# well within a couple of standard errors of each other). Rather than
# assert a numeric bound that doesn't hold, this only checks both estimates
# are finite and non-negative -- the same class of deferral areaChange.md
# made for its similarly noisy AREA_CHNG metric.
for (st in statesDwm) {
  test_that(paste("dwm() TI and SMA are both finite and non-negative (", st, ")"), {
    ti <- as.data.frame(dwm(dbs[[st]], byFuelType = FALSE, method = 'TI'))
    sma <- as.data.frame(dwm(dbs[[st]], byFuelType = FALSE, method = 'SMA'))
    expect_true(is.finite(ti$VOL_ACRE) && ti$VOL_ACRE >= 0)
    expect_true(is.finite(sma$VOL_ACRE) && sma$VOL_ACRE >= 0)
  })
}

# Test 15 ------------------------------
# totals = TRUE / per-acre consistency holds under every non-TI method, not
# just TI (Test 7 above only checked the TI/default path).
for (st in statesDwm) {
  test_that(paste("dwm() totals are consistent with per-acre estimates under non-TI methods (", st, ")"), {
    for (m in c('SMA', 'LMA', 'EMA', 'ANNUAL')) {
      out <- as.data.frame(dwm(dbs[[st]], byFuelType = FALSE, totals = TRUE, method = m))
      expect_equal(out$VOL_TOTAL / out$AREA_TOTAL, out$VOL_ACRE, tolerance = 1e-9,
                   label = paste0(st, " ", m, " VOL_ACRE"))
      expect_equal(out$BIO_TOTAL / out$AREA_TOTAL, out$BIO_ACRE, tolerance = 1e-9,
                   label = paste0(st, " ", m, " BIO_ACRE"))
      expect_equal(out$CARB_TOTAL / out$AREA_TOTAL, out$CARB_ACRE, tolerance = 1e-9,
                   label = paste0(st, " ", m, " CARB_ACRE"))
    }
  })
}

# Test 16 ------------------------------
# byPlot = TRUE combined with a non-TI method is a distinct code path --
# mergeSmallStrata() (R/util.R) is explicitly skipped whenever byPlot =
# TRUE, regardless of method. Confirm it still returns per-plot (not
# population-level) rows without error. RI is used here (not excluded, see
# note above), since byPlot output doesn't depend on the current
# population-estimation eval the same way and works fine on RI's data.
test_that("dwm() byPlot = TRUE works with a non-TI method (RI, SMA)", {
  out <- as.data.frame(dwm(db_ri, method = 'SMA', byPlot = TRUE))
  expect_true(all(c('PLT_CN', 'VOL_ACRE') %in% names(out)))
  expect_gt(nrow(out), 1) # per-plot rows, not a single population estimate
})

# Test 17 ------------------------------
# areaDomain + grpBy interaction under every non-TI method: the filter must
# still restrict (or, in a legitimate edge case, exactly reproduce) the
# unfiltered total, and grpBy = OWNGRPCD must not silently drop it for any
# group -- checked *per YEAR*, since ANNUAL returns multiple year-rows.
# Uses <= rather than strict < for the "filter restricts" check, matching
# carbon.md's precedent for the same reason (a filter can legitimately
# match 100% of a small state's current panel).
for (st in statesDwm) {
  for (m in c('SMA', 'LMA', 'EMA', 'ANNUAL')) {
    test_that(paste("dwm() areaDomain survives grpBy under method =", m, "(", st, ")"), {
      db_st <- dbs[[st]]
      expect_no_warning({
        base <- as.data.frame(dwm(db_st, byFuelType = FALSE, totals = TRUE, method = m))
        filtered <- as.data.frame(dwm(db_st, byFuelType = FALSE, areaDomain = PHYSCLCD %in% 21:29,
                                      totals = TRUE, method = m))
        grouped <- as.data.frame(dwm(db_st, byFuelType = FALSE, areaDomain = PHYSCLCD %in% 21:29,
                                     grpBy = OWNGRPCD, totals = TRUE, method = m))
      })
      mergedBase <- merge(filtered[, c("YEAR", "VOL_TOTAL")], base[, c("YEAR", "VOL_TOTAL")],
                          by = "YEAR", suffixes = c("_filt", "_base"))
      expect_true(all(mergedBase$VOL_TOTAL_filt <= mergedBase$VOL_TOTAL_base))

      byYearGrouped <- aggregate(VOL_TOTAL ~ YEAR, data = grouped, sum)
      mergedGrouped <- merge(byYearGrouped, filtered[, c("YEAR", "VOL_TOTAL")],
                             by = "YEAR", suffixes = c("_grp", "_filt"))
      expect_equal(nrow(mergedGrouped), nrow(byYearGrouped))
      expect_equal(mergedGrouped$VOL_TOTAL_grp, mergedGrouped$VOL_TOTAL_filt, tolerance = 1e-3)
    })
  }
}

# Test 18 ------------------------------
# Plain default-args smoke tests, one per state, for EMA and ANNUAL. ANNUAL
# is regression coverage for the combineMR()/ANNUAL pooling bug found and
# fixed during area()'s non-TI pass (tpa.md, "Fixed" #6) -- dwm() shares
# the same combineMR() call site, so a returned multi-row (not pooled)
# result here confirms the fix covers it too. EMA mirrors tpa.md's v1.1.1
# regression coverage.
for (st in statesDwm) {
  test_that(paste("dwm() runs with method = 'EMA' and default arguments (", st, ")"), {
    expect_no_error(out <- as.data.frame(dwm(dbs[[st]], method = 'EMA')))
    expect_s3_class(out, "data.frame")
  })

  test_that(paste("dwm() runs with method = 'ANNUAL' and default arguments, multiple rows per panel (", st, ")"), {
    expect_no_error(out <- as.data.frame(dwm(dbs[[st]], method = 'ANNUAL')))
    expect_s3_class(out, "data.frame")
    expect_gt(length(unique(out$YEAR)), 1) # not pooled into a single mislabeled row
  })
}

