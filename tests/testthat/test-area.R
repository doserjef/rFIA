# Test tpa() --------------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Most recent estimates for timberland
out <- area(db = fiaRI_mr, landType = 'timber', totals = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 2 ------------------------------
# Most recent estimates for forest land by plot
out <- area(db = fiaRI_mr, landType = 'forest', byPlot = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
# Estimates for live white pine (> 12" DBH)
out <- area(fiaRI_mr,
           treeDomain = SPCD == 129 & DIA > 22) # Species code for white pine

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 4 ------------------------------
# Most recent estimates grouped by stand age on forest land
# Make a categorical variable which represents stand age (grouped by 10 yr intervals)
fiaRI_mr$COND$STAND_AGE <- makeClasses(fiaRI_mr$COND$STDAGE, interval = 10)
out <- area(db = fiaRI_mr, grpBy = STAND_AGE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 5 ------------------------------ 
# Estimates for area with trees greater than 20 in DBH
out <- area(db = fiaRI, landType = 'forest', treeDomain = DIA > 20)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

test_that("multiple years", {
  expect_gt(length(unique(out$YEAR)), 1)
})

# Test 6 ------------------------------
# Most recent estimates for all stems on forest land by species
out <- area(db = fiaRI_mr, landType = 'forest', grpBy = SPCD)
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 7 ------------------------------
# Most recent estimates for all stems on forest land 
# grouped by user-defined areal units
out <- area(fiaRI_mr,
           polys = countiesRI,
           returnSpatial = TRUE)
plot.out <- plotFIA(out, AREA_TOTAL) # Plot of TPA with color scale
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
# (including the EVALIDator-comparison tests further down), since clipping a
# full state extract (NC/CO/OR) takes several seconds and multiple tests
# need each state.
states <- c("RI", "NC", "CO", "OR")
dbs <- lapply(states, \(st) clipFIA(readFIA(validation_data_dir, states = st), mostRecent = TRUE))
names(dbs) <- states
db_ri <- dbs[["RI"]]
# EVAL_GRP encodes STATECD + 4-digit year (e.g. 442024 = Rhode Island 2024);
# reading it off each clipped db mirrors exactly which evaluation
# `mostRecent` actually selected, so it never needs to be hard-coded or kept
# in sync by hand.
wcs <- lapply(dbs, \(d) unique(d$POP_EVAL_GRP$EVAL_GRP))
wc_ri <- wcs[["RI"]]

# Test 8 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(area(db_ri, polys = countiesRI, landType = 'forest',
                               returnSpatial = TRUE, totals = TRUE))
  out_df <- as.data.frame(area(db_ri, polys = countiesRI, landType = 'forest',
                               returnSpatial = FALSE, totals = TRUE))
  common <- intersect(names(out_sf), names(out_df))
  out_sf <- out_sf[order(out_sf$polyID), common]
  out_df <- out_df[order(out_df$polyID), common]
  expect_equal(out_sf, out_df)
})

# Test 9 ------------------------------
# treeDomain must have a genuine effect on the estimate, and that effect
# must survive a grpBy join without being silently dropped for some or all
# groups -- the historical area()/areaChange() bug pattern from v1.1.1,
# where a treeDomain combined with grpBy was ignored and area() just
# returned the area of all forest land for every group.
for (st in states) {
  test_that(paste("area() treeDomain survives grpBy without being silently dropped (", st, ")"), {
    db_st <- dbs[[st]]
    base <- as.data.frame(area(db_st, landType = 'forest', totals = TRUE))
    filtered <- as.data.frame(area(db_st, landType = 'forest', treeDomain = DIA > 20, totals = TRUE))
    grouped <- as.data.frame(area(db_st, landType = 'forest', treeDomain = DIA > 20,
                                  grpBy = OWNGRPCD, totals = TRUE))
    # The filter must actually restrict area (not silently ignored).
    expect_lt(filtered$AREA_TOTAL, base$AREA_TOTAL)
    # And grpBy must not silently drop it for any group -- summing across
    # groups must reproduce the same, filtered total.
    expect_equal(sum(grouped$AREA_TOTAL), filtered$AREA_TOTAL, tolerance = 1e-6)
  })
}

# Test 10 ------------------------------
# PERC_AREA is defined relative to the full landType land base, not per
# group (see man/area.Rd), so grouping by a mutually exclusive COND-level
# variable (FORTYPCD) should have percentages summing to exactly 100%. And
# byLandType = TRUE's four mutually exclusive categories should sum to the
# same total as landType = 'all' (see area.md, "Fixed" #4).
for (st in states) {
  test_that(paste("area() PERC_AREA sums to 100% across FORTYPCD groups (", st, ")"), {
    out <- as.data.frame(area(dbs[[st]], landType = 'forest', grpBy = FORTYPCD, totals = TRUE))
    expect_equal(sum(out$PERC_AREA), 100, tolerance = 1e-6)
  })

  test_that(paste("area() byLandType sums to landType = 'all' (", st, ")"), {
    out <- as.data.frame(area(dbs[[st]], byLandType = TRUE, totals = TRUE))
    all_ <- as.data.frame(area(dbs[[st]], landType = 'all', totals = TRUE))
    expect_equal(sum(out$AREA_TOTAL), all_$AREA_TOTAL, tolerance = 1e-6)
  })
}

# Test 11 ------------------------------
# A treeDomain matching no trees should return a clean 0-row result, not
# error or emit an internal max()-on-empty-vector warning (combineMR() is
# shared with tpa(); see tpa.md, "Fixed" #2).
test_that("area() handles an empty treeDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(area(db_ri, landType = 'forest', treeDomain = SPCD == 999))
  )
  expect_equal(nrow(out), 0)
})

# Numeric validation against EVALIDator ------------------------------------
# Reference values are fetched live from the FIADB-API `fullreport` endpoint
# (see core_references/validation/fetch_evalidator.R and
# core_references/validation/area.md for methodology and full results)
# rather than hard-coded, so these tests can never drift from what
# EVALIDator currently reports. They require network access to
# apps.fs.usda.gov (on top of the local data cache already required above),
# so they're skipped (not failed) when it's unavailable.
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

# Test 12 ------------------------------
# Core default case (landType = 'forest'/'timber') matches EVALIDator to
# full double precision across one state per FIA region: RI (Northern), NC
# (Southern), CO (Interior West), OR (Pacific Northwest). EVALIDator
# attribute 2 = forest land area, 3 = timberland area, both non-ratio
# (area() itself has no denominator here, since PERC_AREA = 100% for the
# unrestricted default case).
for (st in states) {
  wc_st <- wcs[[st]]

  test_that(paste("area() matches EVALIDator for landType = 'forest' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 2)
    out <- as.data.frame(area(dbs[[st]], landType = 'forest', totals = TRUE))
    expect_equal(out$AREA_TOTAL, ref$estimate, tolerance = 1e-6)
    expect_equal(out$AREA_TOTAL_SE, ref$sePercent, tolerance = 1e-6)
    expect_equal(out$nPlots_AREA_NUM, ref$plotCount)
    expect_equal(out$nPlots_AREA_DEN, ref$plotCount)
  })

  test_that(paste("area() matches EVALIDator for landType = 'timber' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 3)
    out <- as.data.frame(area(dbs[[st]], landType = 'timber', totals = TRUE))
    expect_equal(out$AREA_TOTAL, ref$estimate, tolerance = 1e-6)
    expect_equal(out$AREA_TOTAL_SE, ref$sePercent, tolerance = 1e-6)
    expect_equal(out$nPlots_AREA_NUM, ref$plotCount)
    expect_equal(out$nPlots_AREA_DEN, ref$plotCount)
  })
}

# Test 13 ------------------------------
# landType variants that previously silently undercounted due to the
# PLOT_STATUS_CD == 1 bug (see area.md, "Fixed" #1), and 'all' specifically,
# which previously overcounted by including nonsampled conditions (area.md,
# "Fixed" #4). EVALIDator attribute 79 ("area of sampled land and water") is
# the EXPCURR-tagged ground truth for these -- attribute 1 is EXPALL-tagged
# and not directly comparable to area()'s own EXPCURR-based estimates (see
# area.md, "Notes").
for (st in states) {
  wc_st <- wcs[[st]]

  test_that(paste("area() matches EVALIDator for landType = 'water' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 79, strFilter = "COND.COND_STATUS_CD in (3,4)")
    out <- as.data.frame(area(dbs[[st]], landType = 'water', totals = TRUE))
    expect_equal(out$AREA_TOTAL, ref$estimate, tolerance = 1e-6)
    expect_equal(out$nPlots_AREA_NUM, ref$plotCount)
  })

  test_that(paste("area() matches EVALIDator for landType = 'non-forest' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 79, strFilter = "COND.COND_STATUS_CD = 2")
    out <- as.data.frame(area(dbs[[st]], landType = 'non-forest', totals = TRUE))
    expect_equal(out$AREA_TOTAL, ref$estimate, tolerance = 1e-6)
    expect_equal(out$nPlots_AREA_NUM, ref$plotCount)
  })

  test_that(paste("area() matches EVALIDator for landType = 'all' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 79)
    out <- as.data.frame(area(dbs[[st]], landType = 'all', totals = TRUE))
    expect_equal(out$AREA_TOTAL, ref$estimate, tolerance = 1e-6)
    expect_equal(out$nPlots_AREA_NUM, ref$plotCount)
  })
}

# Test 14 ------------------------------
# areaDomain filter interaction, all four FIA regions, matched against
# EVALIDator attribute 2 (forest land area) restricted by an equivalent
# strFilter. An areaDomain restricts the numerator and denominator
# identically for landType = 'forest' with no treeDomain, so PERC_AREA =
# 100% and AREA_TOTAL is directly comparable to a plain (non-ratio)
# EVALIDator area estimate.
for (st in states) {
  test_that(paste("area() matches EVALIDator for areaDomain (physiographic class filter) (", st, ")"), {
    wc_st <- wcs[[st]]
    ref <- fetchRef(wc = wc_st, snum = 2,
                     strFilter = "COND.PHYSCLCD in (21,22,23,24,25,26,27,28,29)")
    out <- as.data.frame(area(dbs[[st]], landType = 'forest',
                              areaDomain = PHYSCLCD %in% 21:29, totals = TRUE)) # mesic classes
    expect_equal(out$AREA_TOTAL, ref$estimate, tolerance = 1e-6)
    expect_equal(out$AREA_TOTAL_SE, ref$sePercent, tolerance = 1e-6)
    expect_equal(out$nPlots_AREA_NUM, ref$plotCount)
  })
}

# Test 15 ------------------------------
# areaDomain combined with a non-forest landType (see area.md, "Fixed" #3 --
# udAreaDomain() previously hard-coded a forest-only filter, silently
# zeroing out any non-forest landType + areaDomain combination). RI only,
# using COUNTYCD (populated across all COND_STATUS_CD values, unlike most
# forest-specific COND variables like PHYSCLCD) as a non-trivial filter with
# real forest/non-forest/water overlap.
test_that("area() matches EVALIDator for landType = 'water' + areaDomain (RI)", {
  ref <- fetchRef(wc = wc_ri, snum = 79,
                   strFilter = "COND.COND_STATUS_CD in (3,4) AND COND.COUNTYCD = 7")
  out <- as.data.frame(area(db_ri, landType = 'water', areaDomain = COUNTYCD == 7, totals = TRUE))
  expect_equal(out$AREA_TOTAL, ref$estimate, tolerance = 1e-6)
  expect_equal(out$nPlots_AREA_NUM, ref$plotCount)
})

test_that("area() matches EVALIDator for landType = 'non-forest' + areaDomain (RI)", {
  ref <- fetchRef(wc = wc_ri, snum = 79,
                   strFilter = "COND.COND_STATUS_CD = 2 AND COND.COUNTYCD = 7")
  out <- as.data.frame(area(db_ri, landType = 'non-forest', areaDomain = COUNTYCD == 7, totals = TRUE))
  expect_equal(out$AREA_TOTAL, ref$estimate, tolerance = 1e-6)
  expect_equal(out$nPlots_AREA_NUM, ref$plotCount)
})

# Non-TI method (SMA/LMA/EMA/ANNUAL) internal consistency -------------------
# EVALIDator has no equivalent for these, so correctness here means: the
# code runs cleanly across the same filter/grpBy/byPlot space already
# exercised above, area()'s own internal identities hold regardless of
# method, and the documented cross-method relationships in
# vignettes/alternativeEstimators.Rmd hold as *bounded*/*directional*
# checks -- never exact equality (see tpa.md for the full writeup of why).
# See tests/testthat/test-util.R for the underlying maWeights()/
# filterAnnual()/combineMR() unit-level checks these per-function tests
# build on, and tpa.md "Fixed" #6 for a package-wide combineMR()/ANNUAL bug
# found and fixed during this pass (also affected tpa(), not just area()).

# Test 16 ------------------------------
# EMA(lambda -> 1) should monotonically approach SMA (RI). Never exactly
# equal -- lambda never literally reaches 1 in a real call (see
# test-util.R for why the exact boundary is degenerate) -- so this checks
# the trend, not a fixed-tolerance snapshot. Mirrors tpa.md's Test 16.
test_that("area() EMA(lambda -> 1) monotonically approaches SMA (RI)", {
  sma <- as.data.frame(area(db_ri, landType = 'forest', method = 'SMA', totals = TRUE))
  dists <- sapply(c(0.5, 0.9, 0.99, 0.999), \(lam) {
    ema <- as.data.frame(area(db_ri, landType = 'forest', method = 'EMA', lambda = lam, totals = TRUE))
    abs(ema$AREA_TOTAL - sma$AREA_TOTAL)
  })
  expect_true(all(diff(dists) < 0))
  expect_lt(dists[length(dists)], 100)
})

# Test 17 ------------------------------
# TI and SMA are not claimed to be numerically equal in general -- TI
# implicitly weights each panel by its plot count, SMA weights every panel
# equally regardless of size. Reusing the same flat 10% relative tolerance
# established empirically in tpa.md (panel plot-count CV is a property of
# each state's panel structure, not of the estimator, so it applies
# unchanged here): RI/NC/CO/OR landed within ~5% of each other for
# AREA_TOTAL too (-0.65%/0.76%/-2.15%/-4.98%), well inside the 10% bound.
for (st in states) {
  test_that(paste("area() TI and SMA agree within a bounded tolerance (", st, ")"), {
    ti <- as.data.frame(area(dbs[[st]], landType = 'forest', method = 'TI', totals = TRUE))
    sma <- as.data.frame(area(dbs[[st]], landType = 'forest', method = 'SMA', totals = TRUE))
    expect_equal(sma$AREA_TOTAL, ti$AREA_TOTAL, tolerance = 0.10)
  })
}

# Test 18 ------------------------------
# area()'s own internal identities (Test 10 above, TI-only) re-checked under
# every non-TI method: PERC_AREA sums to 100% across FORTYPCD groups, and
# byLandType = TRUE sums to landType = 'all' -- both checked *per YEAR*
# rather than pooled across years, since method = 'ANNUAL' returns one row
# per sampled panel-year (unlike SMA/LMA/EMA, which return a single row).
for (st in states) {
  for (m in c('SMA', 'LMA', 'EMA', 'ANNUAL')) {
    test_that(paste("area() PERC_AREA sums to 100% across FORTYPCD groups under method =", m, "(", st, ")"), {
      out <- as.data.frame(area(dbs[[st]], landType = 'forest', grpBy = FORTYPCD, method = m, totals = TRUE))
      byYear <- aggregate(PERC_AREA ~ YEAR, data = out, sum)
      expect_equal(byYear$PERC_AREA, rep(100, nrow(byYear)), tolerance = 1e-6)
    })

    test_that(paste("area() byLandType sums to landType = 'all' under method =", m, "(", st, ")"), {
      out <- as.data.frame(area(dbs[[st]], byLandType = TRUE, method = m, totals = TRUE))
      all_ <- as.data.frame(area(dbs[[st]], landType = 'all', method = m, totals = TRUE))
      byYear <- aggregate(AREA_TOTAL ~ YEAR, data = out, sum)
      merged <- merge(byYear, all_[, c("YEAR", "AREA_TOTAL")], by = "YEAR", suffixes = c("_byLT", "_all"))
      expect_equal(nrow(merged), nrow(byYear)) # every year matched, none dropped
      expect_equal(merged$AREA_TOTAL_byLT, merged$AREA_TOTAL_all, tolerance = 1e-4)
    })
  }
}

# Test 19 ------------------------------
# byPlot = TRUE combined with a non-TI method is a distinct code path --
# mergeSmallStrata() (R/util.R) is explicitly skipped whenever byPlot =
# TRUE, regardless of method. Confirm it still returns per-plot (not
# population-level) rows without error. Mirrors tpa.md's Test 19.
test_that("area() byPlot = TRUE works with a non-TI method (RI, SMA)", {
  out <- as.data.frame(area(db_ri, landType = 'forest', method = 'SMA', byPlot = TRUE))
  expect_true(all(c('PLT_CN', 'PROP_FOREST') %in% names(out)))
  expect_gt(nrow(out), 1) # per-plot rows, not a single population estimate
})

# Test 20 ------------------------------
# treeDomain + grpBy interaction (the historical v1.1.1 area()/areaChange()
# bug pattern already checked for TI in Test 9 above) re-run under every
# non-TI method: the filter must still restrict area, and grpBy must not
# silently drop it for any group -- checked *per YEAR*, since ANNUAL returns
# multiple year-rows.
for (st in states) {
  for (m in c('SMA', 'LMA', 'EMA', 'ANNUAL')) {
    test_that(paste("area() treeDomain survives grpBy under method =", m, "(", st, ")"), {
      db_st <- dbs[[st]]
      expect_no_warning({
        base <- as.data.frame(area(db_st, landType = 'forest', totals = TRUE, method = m))
        filtered <- as.data.frame(area(db_st, landType = 'forest', treeDomain = DIA > 20, totals = TRUE, method = m))
        grouped <- as.data.frame(area(db_st, landType = 'forest', treeDomain = DIA > 20,
                                      grpBy = OWNGRPCD, totals = TRUE, method = m))
      })
      mergedBase <- merge(filtered[, c("YEAR", "AREA_TOTAL")], base[, c("YEAR", "AREA_TOTAL")],
                          by = "YEAR", suffixes = c("_filt", "_base"))
      expect_true(all(mergedBase$AREA_TOTAL_filt < mergedBase$AREA_TOTAL_base))

      byYearGrouped <- aggregate(AREA_TOTAL ~ YEAR, data = grouped, sum)
      mergedGrouped <- merge(byYearGrouped, filtered[, c("YEAR", "AREA_TOTAL")],
                             by = "YEAR", suffixes = c("_grp", "_filt"))
      expect_equal(nrow(mergedGrouped), nrow(byYearGrouped))
      expect_equal(mergedGrouped$AREA_TOTAL_grp, mergedGrouped$AREA_TOTAL_filt, tolerance = 1e-4)
    })
  }
}

# Test 21 ------------------------------
# Plain default-args EMA smoke test, one per state -- regression coverage
# for the v1.1.1 "error when setting method = 'EMA'" bug (NEWS.md), which
# previously had zero dedicated regression tests anywhere in the package.
# Mirrors tpa.md's Test 21.
for (st in states) {
  test_that(paste("area() runs with method = 'EMA' and default arguments (", st, ")"), {
    expect_no_error(out <- as.data.frame(area(dbs[[st]], method = 'EMA')))
    expect_s3_class(out, "data.frame")
  })
}

# Test 22 ------------------------------
# Regression test for the combineMR()/ANNUAL pooling bug found during this
# pass (see area.md "Fixed" and tpa.md "Fixed" #6): on a
# clipFIA(mostRecent = TRUE) db, method = 'ANNUAL' previously relabeled
# every constituent panel to the same YEAR and summed them into one badly
# inflated row (RI: AREA_TOTAL = 2,638,587 vs. the true 2025 value of
# 427,243) instead of returning one row per real sampled panel. RI only
# (this exact repro is what surfaced the bug; full 4-state coverage of
# ANNUAL's shape is already exercised in Tests 18/20 above).
for (st in states) {
  test_that(paste("area() runs with method = 'ANNUAL' and default arguments, one row per panel (", st, ")"), {
    expect_no_error(out <- as.data.frame(area(dbs[[st]], method = 'ANNUAL')))
    expect_s3_class(out, "data.frame")
    expect_gt(nrow(out), 1) # not pooled into a single mislabeled row
  })
}

test_that("area() method = 'ANNUAL' on a clipFIA(mostRecent = TRUE) db matches the unclipped history (RI)", {
  ann_clipped <- as.data.frame(area(db_ri, landType = 'forest', totals = TRUE, method = 'ANNUAL'))
  fiaRI_full <- readFIA(validation_data_dir, states = "RI")
  ann_full <- as.data.frame(area(fiaRI_full, landType = 'forest', totals = TRUE, method = 'ANNUAL'))

  latest <- ann_clipped[which.max(ann_clipped$YEAR), ]
  fullMatch <- ann_full[ann_full$YEAR == latest$YEAR, ]
  expect_equal(latest$AREA_TOTAL, fullMatch$AREA_TOTAL, tolerance = 1e-6)
  expect_equal(latest$nPlots_AREA_NUM, fullMatch$nPlots_AREA_NUM)
})
