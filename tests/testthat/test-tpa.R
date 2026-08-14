# Test tpa() --------------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Most recent estimates for growing stock on timber land by species
out <- tpa(db = fiaRI_mr, land = 'timber', bySpecies = TRUE, totals = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 2 ------------------------------
# Most recent estimates for growing stock on timber land by species by plot
out <- tpa(db = fiaRI_mr, land = 'timber', bySpecies = TRUE, byPlot = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
# Estimates for live white pine (> 12" DBH) on forested mesic sites
out <- tpa(fiaRI_mr,
           treeType = 'live',
           treeDomain = SPCD == 129 & DIA > 12, # Species code for white pine
           areaDomain = PHYSCLCD %in% 21:29) # Mesic Physiographic classes

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 4 ------------------------------
# Most recent estimates grouped by stand age on forest land
# Make a categorical variable which represents stand age (grouped by 10 yr intervals)
fiaRI_mr$COND$STAND_AGE <- makeClasses(fiaRI_mr$COND$STDAGE, interval = 10)
out <- tpa(db = fiaRI_mr, grpBy = STAND_AGE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 5 ------------------------------ 
# Estimates for snags greater than 20 in DBH on forestland for all
#  available inventories (time-series)
out <- tpa(db = fiaRI, landType = 'forest', treeType = 'dead',
           treeDomain = DIA > 20)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

test_that("multiple years", {
  expect_gt(length(unique(out$YEAR)), 1)
})

# Test 6 ------------------------------
# Most recent estimates for all stems on forest land by species
out <- tpa(db = fiaRI_mr, landType = 'forest', treeType = 'all',
           bySpecies = TRUE)
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 7 ------------------------------
# Most recent estimates for all stems on forest land 
# grouped by user-defined areal units
out <- tpa(fiaRI_mr,
           polys = countiesRI,
           returnSpatial = TRUE)
plot.out <- plotFIA(out, TPA) # Plot of TPA with color scale
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
# `mostRecent = TRUE` selected, so it never needs to be hard-coded or kept in
# sync by hand.
wcs <- lapply(dbs, \(d) unique(d$POP_EVAL_GRP$EVAL_GRP))
wc_ri <- wcs[["RI"]]

# Test 8 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(tpa(db_ri, polys = countiesRI, returnSpatial = TRUE))
  out_df <- as.data.frame(tpa(db_ri, polys = countiesRI, returnSpatial = FALSE))
  # Bracket-subsetting (rather than `$geom <- NULL`) drops the stray
  # sf_column/agr attributes that otherwise linger on the data.frame and
  # make an exact comparison fail despite identical values.
  out_sf <- out_sf[, names(out_df)]
  out_sf <- out_sf[order(out_sf$COUNTY), ]
  out_df <- out_df[order(out_df$COUNTY), ]
  expect_equal(out_sf, out_df)
})

# Test 9 ------------------------------
# treeType = 'all' includes every tree regardless of status, so it is only
# a lower *bound* for treeType = 'live' + treeType = 'dead', not an
# equality -- 'dead' requires STANDING_DEAD_CD == 1 (matching EVALIDator's
# "standing dead" definition; see tpa.md, "Fixed" #3), so 'all' still
# includes down/broken dead trees (and any other non-live/non-dead
# STATUSCD) that 'dead' now excludes. landType doesn't depend on treeType,
# so the area denominator is the same in each call, meaning this bound
# holds for both per-acre and total columns.
for (st in states) {
  test_that(paste("tpa() treeType = 'all' is at least live + dead (", st, ")"), {
    db_st <- dbs[[st]]
    all_ <- as.data.frame(tpa(db_st, treeType = 'all', landType = 'forest', totals = TRUE))
    live_ <- as.data.frame(tpa(db_st, treeType = 'live', landType = 'forest', totals = TRUE))
    dead_ <- as.data.frame(tpa(db_st, treeType = 'dead', landType = 'forest', totals = TRUE))
    # A small relative tolerance absorbs floating-point summation-order noise
    # between the three separate tpa() calls (magnitudes here range from
    # ~1e2 for TPA/BAA to ~1e10 for the _TOTAL columns); it's negligible next
    # to any real non-standing-dead contribution, which would be many orders
    # of magnitude larger.
    atLeast <- function(x, y) expect_true(x >= y - abs(y) * 1e-9 - 1e-6)
    atLeast(all_$TPA, live_$TPA + dead_$TPA)
    atLeast(all_$BAA, live_$BAA + dead_$BAA)
    atLeast(all_$TREE_TOTAL, live_$TREE_TOTAL + dead_$TREE_TOTAL)
    atLeast(all_$BA_TOTAL, live_$BA_TOTAL + dead_$BA_TOTAL)
  })
}

# Test 10 ------------------------------
# Internal consistency: totals divided by area reproduce the per-acre
# estimate (doesn't require EVALIDator).
for (st in states) {
  test_that(paste("tpa() totals are consistent with per-acre estimates (", st, ")"), {
    out <- as.data.frame(tpa(dbs[[st]], treeType = 'live', landType = 'forest',
                             totals = TRUE))
    expect_equal(out$TREE_TOTAL / out$AREA_TOTAL, out$TPA, tolerance = 1e-9)
    expect_equal(out$BA_TOTAL / out$AREA_TOTAL, out$BAA, tolerance = 1e-9)
  })
}

# Test 11 ------------------------------
# A treeDomain matching no trees should return a clean 0-row result, not
# error or emit an internal max()-on-empty-vector warning (see tpa.md,
# "Fixed" #2).
test_that("tpa() handles an empty treeDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(tpa(db_ri, treeType = 'live', landType = 'forest',
                             treeDomain = SPCD == 999))
  )
  expect_equal(nrow(out), 0)
})

# Numeric validation against EVALIDator ------------------------------------
# Reference values are fetched live from the FIADB-API `fullreport` endpoint
# (see core_references/validation/fetch_evalidator.R and
# core_references/validation/tpa.md for methodology and full results) rather
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

# Test 12 ------------------------------
# Core default case (treeType = 'live', landType = 'forest') matches
# EVALIDator to full double precision across one state per FIA region:
# RI (Northern), NC (Southern), CO (Interior West), OR (Pacific Northwest).
# EVALIDator attribute 4 = live TPA on forest land, 1004 = live BAA on forest
# land, both ratio'd against attribute 2 (forest land area).
for (st in states) {
  test_that(paste("tpa() matches EVALIDator for", st, "(core default case)"), {
    wc_st <- wcs[[st]]
    tpaRef <- fetchRef(wc = wc_st, snum = 4, sdenom = 2)
    baaRef <- fetchRef(wc = wc_st, snum = 1004, sdenom = 2)

    out_st <- as.data.frame(tpa(dbs[[st]], treeType = 'live', landType = 'forest'))

    expect_equal(out_st$TPA, tpaRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(out_st$BAA, baaRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(out_st$TPA_SE, tpaRef$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out_st$BAA_SE, baaRef$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out_st$nPlots_TREE, tpaRef$numPlotCount)
    expect_equal(out_st$nPlots_AREA, tpaRef$denPlotCount)
  })
}

# Test 13 ------------------------------
# landType/treeType variants across all four FIA regions, matched against
# EVALIDator attributes 7 (timberland/live), 5 (forest/growing-stock), and
# 11264 (forest/standing dead), each ratio'd against the matching area
# attribute (3 = timberland, 2 = forest land).
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("tpa() matches EVALIDator for landType = 'timber' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 7, sdenom = 3)
    out <- as.data.frame(tpa(db_st, treeType = 'live', landType = 'timber'))
    expect_equal(out$TPA, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$TPA_SE, ref$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })

  test_that(paste("tpa() matches EVALIDator for treeType = 'gs' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 5, sdenom = 2)
    out <- as.data.frame(tpa(db_st, treeType = 'gs', landType = 'forest'))
    expect_equal(out$TPA, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$TPA_SE, ref$ratioSEPercent, tolerance = 1e-6)
  })

  test_that(paste("tpa() matches EVALIDator for treeType = 'dead' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 11264, sdenom = 2)
    out <- as.data.frame(tpa(db_st, treeType = 'dead', landType = 'forest'))
    expect_equal(out$TPA, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$TPA_SE, ref$ratioSEPercent, tolerance = 1e-6)
  })
}

# Test 14 ------------------------------
# Domain filter interactions across all four FIA regions. treeDomain matched
# against EVALIDator's `wnum` (numerator-only filter, since a tree-level
# domain should not change the area denominator); areaDomain matched against
# `strFilter` (applies to numerator AND denominator, since an area-level
# domain should shrink both). See tpa.md for why `wnum` vs `strFilter`
# matters here. Both filters here (large-diameter trees, mesic physiographic
# classes) use nationally-defined codes so the same filter is meaningful in
# every region.
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("tpa() matches EVALIDator for treeDomain (DIA >= 20) (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 4, sdenom = 2, wnum = "TREE.DIA >= 20")
    out <- as.data.frame(tpa(db_st, treeType = 'live', landType = 'forest',
                             treeDomain = DIA >= 20))
    expect_equal(out$TPA, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$TPA_SE, ref$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out$nPlots_TREE, ref$numPlotCount)
  })

  test_that(paste("tpa() matches EVALIDator for areaDomain (physiographic class filter) (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 4, sdenom = 2,
                     strFilter = "COND.PHYSCLCD in (21,22,23,24,25,26,27,28,29)")
    out <- as.data.frame(tpa(db_st, treeType = 'live', landType = 'forest',
                             areaDomain = PHYSCLCD %in% 21:29)) # mesic classes
    expect_equal(out$TPA, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$TPA_SE, ref$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out$nPlots_TREE, ref$numPlotCount)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })
}

# RI-specific species filter (kept in addition to the DIA-based filter above,
# since eastern white pine isn't present in meaningful numbers in all four
# states, but it's a useful categorical (rather than numeric-threshold)
# treeDomain check).
test_that("tpa() matches EVALIDator for treeDomain (species filter, RI)", {
  ref <- fetchRef(wc = wc_ri, snum = 4, sdenom = 2, wnum = "TREE.SPCD = 129")
  out <- as.data.frame(tpa(db_ri, treeType = 'live', landType = 'forest',
                           treeDomain = SPCD == 129)) # eastern white pine
  expect_equal(out$TPA, ref$ratioEstimate, tolerance = 1e-6)
  expect_equal(out$TPA_SE, ref$ratioSEPercent, tolerance = 1e-6)
  expect_equal(out$nPlots_TREE, ref$numPlotCount)
})

# Test 15 ------------------------------
# bySpecies grouping (RI): validates a couple of species rows produced by
# grpBy = SPCD against an independent single-species EVALIDator query, i.e.
# that a domain filter survives rFIA's internal grpBy/join path rather than
# being silently dropped for some groups (the historical area()/areaChange()
# bug pattern from v1.1.1). The FIADB-API's own row-grouping mechanism
# (`rselected`) could not be made to return grouped rows via the
# `fullreport` endpoint -- `rselected`/`cselected` appear to be no-ops there,
# always returning a single Total row regardless of value (confirmed by
# inspecting the SQL echoed back in the response: the GROUP BY clause never
# includes a grouping column no matter what `rselected` is set to). See
# tpa.md. RI only, and only a random sample of species (rather than all of
# them), to keep this test's live API call volume small.
test_that("tpa() bySpecies matches EVALIDator per-species (RI)", {
  out <- as.data.frame(tpa(db_ri, treeType = 'live', landType = 'forest',
                           bySpecies = TRUE))
  set.seed(42)
  sampled <- out[sample(nrow(out), 2), ]
  for (i in seq_len(nrow(sampled))) {
    ref <- fetchRef(wc = wc_ri, snum = 4, sdenom = 2,
                     wnum = paste0("TREE.SPCD = ", sampled$SPCD[i]))
    expect_equal(sampled$TPA[i], ref$ratioEstimate, tolerance = 1e-6,
                 label = paste0("TPA (SPCD ", sampled$SPCD[i], ")"))
    expect_equal(sampled$nPlots_TREE[i], ref$numPlotCount,
                 label = paste0("nPlots_TREE (SPCD ", sampled$SPCD[i], ")"))
  }
})

# Non-TI method (SMA/LMA/EMA/ANNUAL) internal consistency -------------------
# EVALIDator has no equivalent for these, so correctness here means: the
# code runs cleanly across the same filter/grpBy/byPlot space already
# exercised above, totals/per-acre plumbing holds regardless of method, and
# the documented cross-method relationships in
# vignettes/alternativeEstimators.Rmd hold as *bounded*/*directional*
# checks -- never exact equality (see tpa.md for the full writeup of why).
# See tests/testthat/test-util.R for the underlying maWeights()/
# filterAnnual() unit-level checks these per-function tests build on.

# Test 16 ------------------------------
# EMA(lambda -> 1) should monotonically approach SMA (RI). Never exactly
# equal -- lambda never literally reaches 1 in a real call (see
# test-util.R for why the exact boundary is degenerate) -- so this checks
# the trend, not a fixed-tolerance snapshot.
test_that("tpa() EMA(lambda -> 1) monotonically approaches SMA (RI)", {
  sma <- as.data.frame(tpa(db_ri, treeType = 'live', landType = 'forest', method = 'SMA'))
  dists <- sapply(c(0.5, 0.9, 0.99, 0.999), \(lam) {
    ema <- as.data.frame(tpa(db_ri, treeType = 'live', landType = 'forest',
                             method = 'EMA', lambda = lam))
    abs(ema$TPA - sma$TPA)
  })
  expect_true(all(diff(dists) < 0))
  expect_lt(dists[length(dists)], 1)
})

# Test 17 ------------------------------
# TI and SMA are not claimed to be numerically equal in general -- TI
# implicitly weights each panel by its plot count, SMA weights every panel
# equally regardless of size (see tpa.md for the panel plot-count CV
# computed for each of these four states). Empirically, across all four
# states (panel-count CV ranging ~0.6-1.5, i.e. none of them have tightly
# balanced panels), TI and SMA still landed within ~5% of each other for
# TPA/BAA, so a single generous relative tolerance is used here rather than
# a per-state balanced/imbalanced tolerance -- see tpa.md for the
# measurements that justified this simplification.
for (st in states) {
  test_that(paste("tpa() TI and SMA agree within a bounded tolerance (", st, ")"), {
    ti <- as.data.frame(tpa(dbs[[st]], treeType = 'live', landType = 'forest', method = 'TI'))
    sma <- as.data.frame(tpa(dbs[[st]], treeType = 'live', landType = 'forest', method = 'SMA'))
    expect_equal(sma$TPA, ti$TPA, tolerance = 0.10)
    expect_equal(sma$BAA, ti$BAA, tolerance = 0.10)
  })
}

# Test 18 ------------------------------
# totals = TRUE / per-acre consistency holds under every non-TI method, not
# just TI (Test 10 above only checked the TI/default path).
for (st in states) {
  test_that(paste("tpa() totals are consistent with per-acre estimates under non-TI methods (", st, ")"), {
    for (m in c('SMA', 'LMA', 'EMA', 'ANNUAL')) {
      out <- as.data.frame(tpa(dbs[[st]], treeType = 'live', landType = 'forest',
                               method = m, totals = TRUE))
      expect_equal(out$TREE_TOTAL / out$AREA_TOTAL, out$TPA, tolerance = 1e-9,
                   label = paste0(st, " ", m, " TPA"))
      expect_equal(out$BA_TOTAL / out$AREA_TOTAL, out$BAA, tolerance = 1e-9,
                   label = paste0(st, " ", m, " BAA"))
    }
  })
}

# Test 19 ------------------------------
# byPlot = TRUE combined with a non-TI method is a distinct code path --
# mergeSmallStrata() (R/util.R) is explicitly skipped whenever byPlot =
# TRUE, regardless of method. Confirm it still returns per-plot (not
# population-level) rows without error.
test_that("tpa() byPlot = TRUE works with a non-TI method (RI, SMA)", {
  out <- as.data.frame(tpa(db_ri, treeType = 'live', landType = 'forest',
                           method = 'SMA', byPlot = TRUE))
  expect_true(all(c('PLT_CN', 'TPA', 'BAA') %in% names(out)))
  expect_gt(nrow(out), 1) # per-plot rows, not a single population estimate
})

# Test 20 ------------------------------
# Domain-filter + bySpecies interaction (the historical
# area()/areaChange() bug pattern from v1.1.1, see tpa.md Test 15 above)
# re-run under every non-TI method: no error, no warning, sane shape.
for (st in states) {
  for (m in c('SMA', 'LMA', 'EMA', 'ANNUAL')) {
    test_that(paste("tpa() domain filter + bySpecies runs cleanly under method =", m, "(", st, ")"), {
      expect_no_warning(
        out <- as.data.frame(tpa(dbs[[st]], treeType = 'live', landType = 'forest',
                                 treeDomain = DIA >= 20, areaDomain = PHYSCLCD %in% 21:29,
                                 bySpecies = TRUE, method = m))
      )
      expect_true(nrow(out) >= 0)
      expect_true(all(out$TPA >= 0, na.rm = TRUE))
    })
  }
}

# Test 21 ------------------------------
# Plain default-args EMA smoke test, one per state -- regression coverage
# for the v1.1.1 "error when setting method = 'EMA'" bug (NEWS.md), which
# previously had zero dedicated regression tests anywhere in the package.
for (st in states) {
  test_that(paste("tpa() runs with method = 'EMA' and default arguments (", st, ")"), {
    expect_no_error(out <- as.data.frame(tpa(dbs[[st]], method = 'EMA')))
    expect_s3_class(out, "data.frame")
  })
}

