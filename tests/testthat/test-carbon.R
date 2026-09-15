# Test carbon() -------------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Return carbon for all forestland by county and return spatial object
out <- carbon(db = fiaRI_mr, polys = countiesRI, returnSpatial = TRUE, byPool = FALSE)
plot.out <- plotFIA(out, CARB_ACRE)
test_that("out is correct", {
  expect_s3_class(out, 'sf')
})
test_that('plot.out is a ggplot', {
  expect_s3_class(plot.out, 'gg')
})

# Test 2 ------------------------------
# Carbon by pool and component for most recent survey on timberland
out <- carbon(db = fiaRI_mr, byPool = TRUE, byComponent = TRUE, landType = 'timber')
test_that('out is correct', {
  expect_s3_class(out, 'tbl_df')
})

# Test 3 ------------------------------
# Carbon on all land by pool 
out <- carbon(db = fiaRI_mr, byPool = TRUE, landType = 'all')
test_that('out is correct', {
  expect_s3_class(out, 'tbl_df')
})

# Test 4 ------------------------------
# carbon on timberland by plot 
out <- carbon(db = fiaRI_mr, landType = 'timber', byPlot = TRUE)
test_that('out is correct', {
  expect_s3_class(out, 'tbl_df')
})

# Test 5 ------------------------------
out <- carbon(db = fiaRI)

test_that('out is correct', {
  expect_s3_class(out, 'tbl_df')
})

# Test 6 ------------------------------
# Over time with method = 'LMA'
out <- carbon(db = fiaRI, method = 'LMA', polys = countiesRI)
test_that('out is correct', {
  expect_s3_class(out, 'tbl_df')
})

# Internal consistency checks (no EVALIDator, no network needed) ----------
# These only require the local FIADB extract cache, not network access, so
# they still run when apps.fs.usda.gov is unreachable. See
# core_references/validation/carbon.md for full methodology/results.
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

# Test 7 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(carbon(db_ri, polys = countiesRI, returnSpatial = TRUE, byPool = FALSE))
  out_df <- as.data.frame(carbon(db_ri, polys = countiesRI, returnSpatial = FALSE, byPool = FALSE))
  out_sf <- out_sf[, names(out_df)]
  out_sf <- out_sf[order(out_sf$COUNTY), ]
  out_df <- out_df[order(out_df$COUNTY), ]
  expect_equal(out_sf, out_df)
})

# Test 8 ------------------------------
# Internal consistency: totals divided by area reproduce the per-acre
# estimate (doesn't require EVALIDator).
for (st in states) {
  test_that(paste("carbon() totals are consistent with per-acre estimates (", st, ")"), {
    out <- as.data.frame(carbon(dbs[[st]], byPool = FALSE, totals = TRUE))
    expect_equal(out$CARB_TOTAL / out$AREA_TOTAL, out$CARB_ACRE, tolerance = 1e-9)
  })
}

# Test 9 ------------------------------
# An areaDomain matching no conditions should return a clean 0-row result,
# not a row of NaN/garbage values. Regression test for a bug found during
# this validation pass: carbon()'s condition-level ("a") data frame is built
# via `data %>% distinct(PLT_CN, CONDID, .keep_all = TRUE)`, but `data` comes
# from a left_join of PLOT onto a COND table already filtered down to
# domain-matching conditions only. When a plot has NO matching condition,
# that left_join still preserves the plot as a single CONDID = NA row (the
# `distinct()` collapses what would otherwise be duplicate all-NA tree rows
# down to one). Without a guard dropping these, carbon()'s numerator
# ("tPlt") is built by joining onto this condition-level frame, so a
# fully-empty areaDomain produced 5 (one per pool) rows of CARB_ACRE = NaN
# with nPlots_AREA equal to the FULL, unfiltered state plot count, instead
# of a clean empty result -- see core_references/validation/carbon.md,
# "Fixed" section, for the full root-cause writeup. tpa()'s equivalent
# code already guards against this (`dplyr::filter(!is.na(CONDID))`); the
# same guard was added to carbonStarter.R (and biomassStarter.R, which had
# the identical bug) as part of this pass.
test_that("carbon() handles an empty areaDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(carbon(db_ri, areaDomain = STATECD == 999))
  )
  expect_equal(nrow(out), 0)
})

# Numeric validation against EVALIDator ------------------------------------
# Reference values are fetched live from the FIADB-API `fullreport` endpoint
# (see core_references/validation/fetch_evalidator.R and
# core_references/validation/carbon.md for methodology and full results)
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

# Metric-tonnes-per-acre-on-forest-land conversion factor FIADB/EVALIDator
# uses to go from the raw short-ton COND/TREE carbon columns to the units
# carbon() reports (matches the constant hard-coded in carbonStarter.R).
kMg <- 0.90718474

# Test 10 ------------------------------
# Core default case (landType = 'forest', byPool = TRUE) matches EVALIDator
# to full double precision across one state per FIA region: RI (Northern),
# NC (Southern), CO (Interior West), OR (Pacific Northwest). EVALIDator
# attributes 98-102 are the IPCC forest carbon pools (live aboveground, live
# belowground, dead wood, litter, soil organic), already reported in metric
# tonnes on forest land; attribute 103 is their sum (all 5 pools). Both
# CARB_ACRE and nPlots_AREA are checked -- the latter is a regression check
# for the nPlots_AREA phantom-row bug fixed as part of this pass (see Test 9
# above and carbon.md).
poolAttrs <- c(AG_LIVE = 98, BG_LIVE = 99, DEAD_WOOD = 100, LITTER = 101, SOIL_ORG = 102)
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("carbon() matches EVALIDator by pool (core default case) (", st, ")"), {
    out <- as.data.frame(carbon(db_st, byPool = TRUE))
    for (pool in names(poolAttrs)) {
      ref <- fetchRef(wc = wc_st, snum = poolAttrs[[pool]], sdenom = 2)
      row <- out[out$POOL == pool, ]
      expect_equal(row$CARB_ACRE, ref$ratioEstimate, tolerance = 1e-6, label = pool)
      expect_equal(row$nPlots_AREA, ref$denPlotCount, label = paste(pool, "nPlots_AREA"))
    }
  })

  test_that(paste("carbon() matches EVALIDator for grand total (all 5 pools) (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 103, sdenom = 2)
    out <- as.data.frame(carbon(db_st, byPool = FALSE))
    expect_equal(out$CARB_ACRE, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })
}

# Test 11 ------------------------------
# landType = 'timber', byComponent = TRUE across all four FIA regions.
# Matched against EVALIDator's short-ton component attributes for
# timberland (62 = AG understory, 63 = BG understory, 65 = litter,
# 66 = soil organic, 64 = down dead [COND.CARBON_DOWN_DEAD only, despite its
# "stumps, coarse roots, CWD" label -- confirmed via its SQL definition],
# 61000 = standing dead trees AG+BG carbon, >= 1in dbh), converted from
# short tons to metric tonnes with the same 0.90718474 factor carbon() uses
# internally.
compAttrsTimber <- c(AG_UNDER_LIVE = 62, BG_UNDER_LIVE = 63, LITTER = 65,
                     SOIL_ORG = 66, DOWN_DEAD = 64, STAND_DEAD = 61000)
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("carbon() matches EVALIDator by component, landType = 'timber' (", st, ")"), {
    out <- as.data.frame(carbon(db_st, byComponent = TRUE, landType = 'timber'))
    for (comp in names(compAttrsTimber)) {
      ref <- fetchRef(wc = wc_st, snum = compAttrsTimber[[comp]], sdenom = 3)
      row <- out[out$COMPONENT == comp, ]
      expect_equal(row$CARB_ACRE, ref$ratioEstimate * kMg, tolerance = 1e-6, label = comp)
    }
  })
}

# Test 12 ------------------------------
# byComponent = TRUE on forest land (default landType) across all four FIA
# regions, complementing the pool-level check in Test 10. Matched against
# EVALIDator's forest-land short-ton component attributes: 48 (AG
# understory), 49 (BG understory), 50 (down dead), 51 (litter), 52 (soil
# organic), 47000 (standing dead trees AG+BG carbon, >= 1in dbh).
compAttrsForest <- c(AG_UNDER_LIVE = 48, BG_UNDER_LIVE = 49, DOWN_DEAD = 50,
                     LITTER = 51, SOIL_ORG = 52, STAND_DEAD = 47000)
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("carbon() matches EVALIDator by component, forest land (", st, ")"), {
    out <- as.data.frame(carbon(db_st, byComponent = TRUE))
    for (comp in names(compAttrsForest)) {
      ref <- fetchRef(wc = wc_st, snum = compAttrsForest[[comp]], sdenom = 2)
      row <- out[out$COMPONENT == comp, ]
      expect_equal(row$CARB_ACRE, ref$ratioEstimate * kMg, tolerance = 1e-6, label = comp)
    }
  })
}

# Test 13 ------------------------------
# areaDomain filter interaction across all four FIA regions (mesic
# physiographic classes) -- doubles as the primary regression test for the
# nPlots_AREA phantom-row fix (Test 9), since it exercises the fix on a
# filter that excludes some, but not all, plots (the fully-empty case in
# Test 9 exercises the other extreme).
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("carbon() matches EVALIDator for areaDomain (physiographic class filter) (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 103, sdenom = 2,
                     strFilter = "COND.PHYSCLCD in (21,22,23,24,25,26,27,28,29)")
    out <- as.data.frame(carbon(db_st, byPool = FALSE, areaDomain = PHYSCLCD %in% 21:29))
    expect_equal(out$CARB_ACRE, ref$ratioEstimate, tolerance = 1e-6)
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
# found and fixed during area()'s non-TI pass -- carbon() shares the same
# combineMR() call site and was unaffected by the time this section was
# written (already fixed at the shared-utility level), confirmed below by
# the ANNUAL smoke test returning multiple rows per state, not pooled ones.
# carbon() has no treeDomain/bySpecies (see carbon.md, "carbon() has no
# treeDomain"), so the domain-filter check below uses areaDomain + grpBy
# only, mirroring area()'s Test 9/Test 20 pattern rather than tpa.md's
# treeDomain-based one.

# Test 14 ------------------------------
# EMA(lambda -> 1) should monotonically approach SMA (RI). Never exactly
# equal (see test-util.R for why the exact boundary is degenerate) -- this
# checks the trend, not a fixed-tolerance snapshot. Mirrors tpa.md's Test 16.
test_that("carbon() EMA(lambda -> 1) monotonically approaches SMA (RI)", {
  sma <- as.data.frame(carbon(db_ri, byPool = FALSE, method = 'SMA'))
  dists <- sapply(c(0.5, 0.9, 0.99, 0.999), \(lam) {
    ema <- as.data.frame(carbon(db_ri, byPool = FALSE, method = 'EMA', lambda = lam))
    abs(ema$CARB_ACRE - sma$CARB_ACRE)
  })
  expect_true(all(diff(dists) < 0))
  expect_lt(dists[length(dists)], 1)
})

# Test 15 ------------------------------
# TI and SMA are not claimed to be numerically equal in general (see
# tpa.md). Reusing the same flat 10% relative tolerance established
# empirically there (panel plot-count CV is a property of each state's
# panel structure, not the estimator): RI/NC/CO/OR's CARB_ACRE landed
# within ~2.9% of each other (0.36%/-2.02%/0.55%/2.86%), well inside the
# 10% bound.
for (st in states) {
  test_that(paste("carbon() TI and SMA agree within a bounded tolerance (", st, ")"), {
    ti <- as.data.frame(carbon(dbs[[st]], byPool = FALSE, method = 'TI'))
    sma <- as.data.frame(carbon(dbs[[st]], byPool = FALSE, method = 'SMA'))
    expect_equal(sma$CARB_ACRE, ti$CARB_ACRE, tolerance = 0.10)
  })
}

# Test 16 ------------------------------
# totals = TRUE / per-acre consistency holds under every non-TI method, not
# just TI (Test 8 above only checked the TI/default path).
for (st in states) {
  test_that(paste("carbon() totals are consistent with per-acre estimates under non-TI methods (", st, ")"), {
    for (m in c('SMA', 'LMA', 'EMA', 'ANNUAL')) {
      out <- as.data.frame(carbon(dbs[[st]], byPool = FALSE, totals = TRUE, method = m))
      expect_equal(out$CARB_TOTAL / out$AREA_TOTAL, out$CARB_ACRE, tolerance = 1e-9,
                   label = paste0(st, " ", m, " CARB_ACRE"))
    }
  })
}

# Test 17 ------------------------------
# byPlot = TRUE combined with a non-TI method is a distinct code path --
# mergeSmallStrata() (R/util.R) is explicitly skipped whenever byPlot =
# TRUE, regardless of method. Confirm it still returns per-plot (not
# population-level) rows without error.
test_that("carbon() byPlot = TRUE works with a non-TI method (RI, SMA)", {
  out <- as.data.frame(carbon(db_ri, byPool = FALSE, method = 'SMA', byPlot = TRUE))
  expect_true(all(c('PLT_CN', 'CARB_ACRE') %in% names(out)))
  expect_gt(nrow(out), 1) # per-plot rows, not a single population estimate
})

# Test 18 ------------------------------
# areaDomain + grpBy interaction under every non-TI method: the filter must
# still restrict the total, and grpBy = OWNGRPCD must not silently drop it
# for any group -- checked *per YEAR*, since ANNUAL returns multiple
# year-rows. Uses <= rather than strict < for the "filter restricts"
# check: confirmed on RI/ANUAL's most data-complete panel (2025) that every
# plot that year happens to already fall within the mesic physiographic
# classes, so the filtered and unfiltered totals are legitimately identical
# for that one panel -- not a bug (every other year/state/method shows a
# real reduction).
for (st in states) {
  for (m in c('SMA', 'LMA', 'EMA', 'ANNUAL')) {
    test_that(paste("carbon() areaDomain survives grpBy under method =", m, "(", st, ")"), {
      db_st <- dbs[[st]]
      expect_no_warning({
        base <- as.data.frame(carbon(db_st, byPool = FALSE, totals = TRUE, method = m))
        filtered <- as.data.frame(carbon(db_st, byPool = FALSE, areaDomain = PHYSCLCD %in% 21:29,
                                         totals = TRUE, method = m))
        grouped <- as.data.frame(carbon(db_st, byPool = FALSE, areaDomain = PHYSCLCD %in% 21:29,
                                        grpBy = OWNGRPCD, totals = TRUE, method = m))
      })
      mergedBase <- merge(filtered[, c("YEAR", "CARB_TOTAL")], base[, c("YEAR", "CARB_TOTAL")],
                          by = "YEAR", suffixes = c("_filt", "_base"))
      expect_true(all(mergedBase$CARB_TOTAL_filt <= mergedBase$CARB_TOTAL_base))

      byYearGrouped <- aggregate(CARB_TOTAL ~ YEAR, data = grouped, sum)
      mergedGrouped <- merge(byYearGrouped, filtered[, c("YEAR", "CARB_TOTAL")],
                             by = "YEAR", suffixes = c("_grp", "_filt"))
      expect_equal(nrow(mergedGrouped), nrow(byYearGrouped))
      expect_equal(mergedGrouped$CARB_TOTAL_grp, mergedGrouped$CARB_TOTAL_filt, tolerance = 1e-3)
    })
  }
}

# Test 19 ------------------------------
# Plain default-args smoke tests, one per state, for EMA and ANNUAL. ANNUAL
# is regression coverage for the combineMR()/ANNUAL pooling bug found and
# fixed during area()'s non-TI pass (tpa.md, "Fixed" #6) -- carbon() shares
# the same combineMR() call site, so a returned multi-row (not pooled)
# result here confirms the fix covers it too. EMA mirrors tpa.md's v1.1.1
# regression coverage.
for (st in states) {
  test_that(paste("carbon() runs with method = 'EMA' and default arguments (", st, ")"), {
    expect_no_error(out <- as.data.frame(carbon(dbs[[st]], method = 'EMA')))
    expect_s3_class(out, "data.frame")
  })

  test_that(paste("carbon() runs with method = 'ANNUAL' and default arguments, multiple rows per panel (", st, ")"), {
    expect_no_error(out <- as.data.frame(carbon(dbs[[st]], method = 'ANNUAL')))
    expect_s3_class(out, "data.frame")
    expect_gt(length(unique(out$YEAR)), 1) # not pooled into a single mislabeled row
  })
}
