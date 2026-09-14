# Test tpa() --------------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Most recent estimates for timberland
out <- areaChange(db = fiaRI_mr, landType = 'timber')

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 2 ------------------------------
# Most recent estimates for forest land by plot
out <- areaChange(db = fiaRI_mr, landType = 'forest', byPlot = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
# Estimates for live white pine (> 12" DBH)
out <- areaChange(fiaRI_mr,
           treeDomain = SPCD == 129 & DIA > 22) # Species code for white pine

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 4 ------------------------------
# Most recent estimates grouped by stand age on forest land
# Make a categorical variable which represents stand age (grouped by 10 yr intervals)
fiaRI_mr$COND$STAND_AGE <- makeClasses(fiaRI_mr$COND$STDAGE, interval = 10)
out <- areaChange(db = fiaRI_mr, grpBy = STAND_AGE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 5 ------------------------------ 
# Estimates for areaChange with trees greater than 20 in DBH
out <- areaChange(db = fiaRI, landType = 'forest', treeDomain = DIA > 20)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

test_that("multiple years", {
  expect_gt(length(unique(out$YEAR)), 1)
})

# Test 7 ------------------------------
# Most recent estimates for all stems on forest land 
# grouped by user-defined areaChangel units
out <- areaChange(fiaRI_mr,
           polys = countiesRI,
           returnSpatial = TRUE, method = 'EMA')
plot.out <- plotFIA(out, AREA_CHNG) # Plot of TPA with color scale
test_that("out is of class sf", {
  expect_s3_class(out, "sf")
})
test_that('plot.out is a ggplot', {
  expect_s3_class(plot.out, 'gg')
})

# Internal consistency checks (no EVALIDator, no network needed) ----------
# These only require the local FIADB extract cache, not network access, so
# they still run when apps.fs.usda.gov is unreachable.
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
# `mostRecent` actually selected, so it never needs to be hard-coded.
wcs <- lapply(dbs, \(d) unique(d$POP_EVAL_GRP$EVAL_GRP))
wc_ri <- wcs[["RI"]]

# Test 8 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(areaChange(db_ri, polys = countiesRI, landType = 'forest',
                                     returnSpatial = TRUE))
  out_df <- as.data.frame(areaChange(db_ri, polys = countiesRI, landType = 'forest',
                                     returnSpatial = FALSE))
  common <- intersect(names(out_sf), names(out_df))
  out_sf <- out_sf[order(out_sf$polyID), common]
  out_df <- out_df[order(out_df$polyID), common]
  expect_equal(out_sf, out_df)
})

# Test 9 ------------------------------
# chngType = 'net' is defined as the net result of diversion and reversion
# processes (see man/areaChange.Rd, "Estimation Details"): net AREA_CHNG must
# equal (reversion AREA_CHNG - diversion AREA_CHNG) from the chngType =
# 'component' breakdown, for both landType = 'forest' and 'timber', across
# all four FIA regions.
for (st in states) {
  for (lt in c('forest', 'timber')) {
    label1 <- if (lt == 'forest') 'Forest' else 'Timber'
    label2 <- if (lt == 'forest') 'Non-forest' else 'Non-timber'

    test_that(paste("areaChange() net AREA_CHNG equals reversion - diversion (",
                     st, ",", lt, ")"), {
      db_st <- dbs[[st]]
      net <- as.data.frame(areaChange(db_st, landType = lt, chngType = 'net'))
      comp <- as.data.frame(areaChange(db_st, landType = lt, chngType = 'component'))

      diversion <- comp$AREA_CHNG[comp$STATUS1 == label1 & comp$STATUS2 == label2]
      reversion <- comp$AREA_CHNG[comp$STATUS1 == label2 & comp$STATUS2 == label1]

      expect_equal(net$AREA_CHNG, reversion - diversion, tolerance = 1e-6)
    })
  }
}

# Test 10 ------------------------------
# A treeDomain matching no trees should return a clean 0-row result, not
# error or emit an internal max()-on-empty-vector warning (combineMR() is
# shared with tpa()/area(); see tpa.md/area.md, "Fixed").
test_that("areaChange() handles an empty treeDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(areaChange(db_ri, landType = 'forest', treeDomain = SPCD == 999))
  )
  expect_equal(nrow(out), 0)
})

# Numeric validation against EVALIDator ------------------------------------
# Reference values are fetched live from the FIADB-API `fullreport` endpoint
# (see core_references/validation/fetch_evalidator.R and
# core_references/validation/areaChange.md for methodology and full results)
# rather than hard-coded, so these tests can never drift from what
# EVALIDator currently reports. They require network access to
# apps.fs.usda.gov (on top of the local data cache already required above),
# so they're skipped (not failed) when it's unavailable.
skip_if_not_installed("curl")
skip_if_not_installed("jsonlite")
source(test_path("..", "..", "core_references", "validation", "fetch_evalidator.R"))

network_ok <- tryCatch({
  fetch_evalidator(wc = wc_ri, snum = 127)
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

# Test 11 ------------------------------
# EVALIDator's EXPCHNG-tagged attributes for area change (126-139) are not
# signed net-change deltas -- they are base-population area totals computed
# from SUBP_COND_CHNG_MTRX proportions, categorized by whether *both* or
# *either* measurement was forest/timberland (see areaChange.md,
# "Methodological note"). Attribute 127/129 = area of conditions that were
# forest/timberland at BOTH measurements -- this matches
# areaChange(chngType = 'component')'s "STATUS1 == STATUS2" (no-change) row
# exactly. Attribute 128/130 = area of conditions that were forest/timberland
# at EITHER measurement -- this matches the sum of PREV_AREA across all three
# component categories (no-change + diversion + reversion), which is exactly
# the population that this bug (nonsampled conditions misclassified as a
# genuine land-use change) previously inflated -- see areaChange.md, "Fixed".
for (st in states) {
  wc_st <- wcs[[st]]

  for (spec in list(list(lt = 'forest', label = 'Forest', snum_both = 127, snum_either = 128),
                    list(lt = 'timber', label = 'Timber', snum_both = 129, snum_either = 130))) {

    test_that(paste("areaChange() matches EVALIDator for landType = '", spec$lt,
                     "', 'both' population (", st, ")"), {
      ref <- fetchRef(wc = wc_st, snum = spec$snum_both)
      out <- as.data.frame(areaChange(dbs[[st]], landType = spec$lt, chngType = 'component'))
      stable <- out[out$STATUS1 == spec$label & out$STATUS2 == spec$label, ]
      expect_equal(stable$PREV_AREA, ref$estimate, tolerance = 1e-6)
      expect_equal(stable$PREV_AREA_SE, ref$sePercent, tolerance = 1e-6)
      expect_equal(stable$nPlots_AREA, ref$plotCount)
    })

    test_that(paste("areaChange() matches EVALIDator for landType = '", spec$lt,
                     "', 'either' population (", st, ")"), {
      ref <- fetchRef(wc = wc_st, snum = spec$snum_either)
      out <- as.data.frame(areaChange(dbs[[st]], landType = spec$lt, chngType = 'component'))
      expect_equal(sum(out$PREV_AREA), ref$estimate, tolerance = 1e-6)
    })
  }
}

# Non-TI method (SMA/LMA/EMA/ANNUAL) internal consistency -------------------
# EVALIDator has no equivalent for these, so correctness here means: the
# code runs cleanly across the same filter/grpBy/byPlot space already
# exercised above, areaChange()'s own internal identities hold regardless of
# method, and the documented cross-method relationships in
# vignettes/alternativeEstimators.Rmd hold as *bounded*/*directional* checks
# -- never exact equality (see tpa.md for the full writeup of why). See
# tests/testthat/test-util.R for the underlying maWeights()/filterAnnual()/
# combineMR() unit-level checks these per-function tests build on, and
# tpa.md "Fixed" #6 for a package-wide combineMR()/ANNUAL bug found and
# fixed during area()'s non-TI pass (also affected areaChange(), confirmed
# below -- fixed before this section was written, so no new bug here).
#
# `PREV_AREA` (a plain, nonnegative area total -- the same role
# `AREA_TOTAL` plays for `area()`) is used for the EMA/SMA convergence and
# TI-vs-SMA bounded-agreement checks below, not the signed `AREA_CHNG`:
# `AREA_CHNG` is driven by a small subpopulation of transitioning plots and
# can be tiny or near a sign flip (confirmed empirically: RI's TI vs SMA
# `AREA_CHNG` differ by ~49% relatively, vs. ~6% for `PREV_AREA`), so a
# relative-tolerance bound on it is not meaningful -- consistent with
# areaChange.md's existing deferral of a numeric `treeDomain` effect check
# on `AREA_CHNG` for the same reason.

# Test 12 ------------------------------
# EMA(lambda -> 1) should monotonically approach SMA (RI), using PREV_AREA.
# Never exactly equal -- lambda never literally reaches 1 in a real call
# (see test-util.R for why the exact boundary is degenerate) -- so this
# checks the trend, not a fixed-tolerance snapshot. Mirrors tpa.md/area.md.
test_that("areaChange() EMA(lambda -> 1) monotonically approaches SMA (RI)", {
  sma <- as.data.frame(areaChange(db_ri, landType = 'forest', chngType = 'net', method = 'SMA'))
  dists <- sapply(c(0.5, 0.9, 0.99, 0.999), \(lam) {
    ema <- as.data.frame(areaChange(db_ri, landType = 'forest', chngType = 'net', method = 'EMA', lambda = lam))
    abs(ema$PREV_AREA - sma$PREV_AREA)
  })
  expect_true(all(diff(dists) < 0))
  expect_lt(dists[length(dists)], 100)
})

# Test 13 ------------------------------
# TI and SMA are not claimed to be numerically equal in general (see
# tpa.md). Reusing the same flat 10% relative tolerance established
# empirically there (panel plot-count CV is a property of each state's
# panel structure, not the estimator): RI/NC/CO/OR's PREV_AREA landed within
# ~6.5% of each other (-6.49%/1.12%/-1.72%/-2.35%), inside the 10% bound.
for (st in states) {
  test_that(paste("areaChange() TI and SMA agree within a bounded tolerance (", st, ")"), {
    ti <- as.data.frame(areaChange(dbs[[st]], landType = 'forest', chngType = 'net', method = 'TI'))
    sma <- as.data.frame(areaChange(dbs[[st]], landType = 'forest', chngType = 'net', method = 'SMA'))
    expect_equal(sma$PREV_AREA, ti$PREV_AREA, tolerance = 0.10)
  })
}

# Test 14 ------------------------------
# areaChange()'s own internal identity (Test 9 above, TI-only) re-checked
# under every non-TI method: net AREA_CHNG must equal reversion - diversion
# from the component breakdown -- checked *per YEAR*, since method =
# 'ANNUAL' returns one row per sampled panel-year. A year's diversion or
# reversion category can be legitimately absent from the component output
# (zero qualifying plots that year -- confirmed on RI, a small state where
# several individual annual panels have a diversion event but no reversion
# event, or vice versa, e.g. 2019 has only a Forest -> Non-forest row), so a
# missing category is treated as AREA_CHNG = 0 for that year rather than
# requiring both rows to be present.
for (st in states) {
  for (lt in c('forest', 'timber')) {
    label1 <- if (lt == 'forest') 'Forest' else 'Timber'
    label2 <- if (lt == 'forest') 'Non-forest' else 'Non-timber'

    for (m in c('SMA', 'LMA', 'EMA', 'ANNUAL')) {
      test_that(paste("areaChange() net AREA_CHNG equals reversion - diversion under method =",
                       m, "(", st, ",", lt, ")"), {
        db_st <- dbs[[st]]
        net <- as.data.frame(areaChange(db_st, landType = lt, chngType = 'net', method = m))
        comp <- as.data.frame(areaChange(db_st, landType = lt, chngType = 'component', method = m))

        diversion <- comp[comp$STATUS1 == label1 & comp$STATUS2 == label2, c('YEAR', 'AREA_CHNG')]
        reversion <- comp[comp$STATUS1 == label2 & comp$STATUS2 == label1, c('YEAR', 'AREA_CHNG')]
        merged <- merge(net[, c('YEAR', 'AREA_CHNG')], diversion, by = 'YEAR', all.x = TRUE, suffixes = c('_net', '_div'))
        merged <- merge(merged, reversion, by = 'YEAR', all.x = TRUE)
        names(merged)[4] <- 'AREA_CHNG_rev'
        merged$AREA_CHNG_div[is.na(merged$AREA_CHNG_div)] <- 0
        merged$AREA_CHNG_rev[is.na(merged$AREA_CHNG_rev)] <- 0

        expect_equal(nrow(merged), length(unique(net$YEAR))) # every year present, none dropped
        expect_equal(merged$AREA_CHNG_net, merged$AREA_CHNG_rev - merged$AREA_CHNG_div, tolerance = 1e-4)
      })
    }
  }
}

# Test 15 ------------------------------
# byPlot = TRUE combined with a non-TI method is a distinct code path --
# mergeSmallStrata() (R/util.R) is explicitly skipped whenever byPlot =
# TRUE, regardless of method. Confirm it still returns per-plot (not
# population-level) rows without error. Mirrors tpa.md/area.md.
test_that("areaChange() byPlot = TRUE works with a non-TI method (RI, SMA)", {
  out <- as.data.frame(areaChange(db_ri, landType = 'forest', method = 'SMA', byPlot = TRUE))
  expect_true(all(c('PLT_CN', 'PROP_CHNG', 'PREV_PROP_FOREST') %in% names(out)))
  expect_gt(nrow(out), 1) # per-plot rows, not a single population estimate
})

# Test 16 ------------------------------
# treeDomain + grpBy interaction under every non-TI method. Specifying
# treeDomain expands areaChange()'s output with TREE_DOMAIN1/TREE_DOMAIN2
# indicator columns (whether the domain was satisfied at each measurement),
# so unlike area()'s simpler single-row-per-group case, the "genuine
# restriction" and "grpBy preserves the total" checks below sum PREV_AREA
# across ALL rows (all TREE_DOMAIN1/2 x STATUS1/STATUS2 combinations) per
# YEAR, not just one STATUS1/STATUS2 subset.
for (st in states) {
  for (m in c('SMA', 'LMA', 'EMA', 'ANNUAL')) {
    test_that(paste("areaChange() treeDomain survives grpBy under method =", m, "(", st, ")"), {
      db_st <- dbs[[st]]
      expect_no_warning({
        base <- as.data.frame(areaChange(db_st, landType = 'forest', chngType = 'component', method = m))
        filtered <- as.data.frame(areaChange(db_st, landType = 'forest', treeDomain = DIA > 20,
                                             chngType = 'component', method = m))
        grouped <- as.data.frame(areaChange(db_st, landType = 'forest', treeDomain = DIA > 20,
                                            grpBy = OWNGRPCD, chngType = 'component', method = m))
      })
      baseYr <- aggregate(PREV_AREA ~ YEAR, data = base, sum)
      filtYr <- aggregate(PREV_AREA ~ YEAR, data = filtered, sum)
      grpYr <- aggregate(PREV_AREA ~ YEAR, data = grouped, sum)

      mergedBase <- merge(filtYr, baseYr, by = 'YEAR', suffixes = c('_filt', '_base'))
      expect_true(all(mergedBase$PREV_AREA_filt < mergedBase$PREV_AREA_base))

      mergedGrouped <- merge(grpYr, filtYr, by = 'YEAR', suffixes = c('_grp', '_filt'))
      expect_equal(nrow(mergedGrouped), nrow(baseYr))
      expect_equal(mergedGrouped$PREV_AREA_grp, mergedGrouped$PREV_AREA_filt, tolerance = 1e-3)
    })
  }
}

# Test 17 ------------------------------
# Plain default-args smoke tests, one per state, for EMA and ANNUAL. ANNUAL
# is new regression coverage specifically for the combineMR()/ANNUAL
# pooling bug found and fixed during area()'s non-TI pass (tpa.md, "Fixed"
# #6) -- confirmed to affect areaChange() too, since it shares the same
# combineMR() call in R/areaChange.R. EMA mirrors tpa.md's/area.md's v1.1.1
# regression coverage.
for (st in states) {
  test_that(paste("areaChange() runs with method = 'EMA' and default arguments (", st, ")"), {
    expect_no_error(out <- as.data.frame(areaChange(dbs[[st]], method = 'EMA')))
    expect_s3_class(out, "data.frame")
  })

  test_that(paste("areaChange() runs with method = 'ANNUAL' and default arguments, one row per panel (", st, ")"), {
    expect_no_error(out <- as.data.frame(areaChange(dbs[[st]], method = 'ANNUAL')))
    expect_s3_class(out, "data.frame")
    expect_gt(length(unique(out$YEAR)), 1) # not pooled into a single mislabeled row
  })
}
