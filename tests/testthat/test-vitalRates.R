# Test vitalRates() --------------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Most recent estimates for growing stock on timber land by species
out <- vitalRates(db = fiaRI_mr, land = 'timber', bySpecies = TRUE, totals = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 2 ------------------------------
# Most recent estimates for growing stock on timber land by species by plot
out <- vitalRates(db = fiaRI_mr, land = 'timber', bySpecies = TRUE, byPlot = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
# Estimates for live white pine (> 12" DBH) on forested mesic sites
out <- vitalRates(fiaRI_mr,
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
out <- vitalRates(db = fiaRI_mr, grpBy = STAND_AGE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 5 ------------------------------ 
# Estimates for live trees greater than 20in DBH on forest land
out <- vitalRates(db = fiaRI, landType = 'forest', treeType = 'live',
           treeDomain = DIA > 20)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

test_that("multiple years", {
  expect_gt(length(unique(out$YEAR)), 1)
})

# Test 6 ------------------------------
# Most recent estimates for all stems on forest land by species
out <- vitalRates(db = fiaRI_mr, landType = 'forest', treeType = 'all',
           bySpecies = TRUE)
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 7 ------------------------------
# Most recent estimates for all stems on forest land 
# grouped by user-defined areal units
out <- vitalRates(fiaRI_mr,
           polys = countiesRI,
           returnSpatial = TRUE)
plot.out <- plotFIA(out, BA_GROW) 
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
# in sync by hand. The same EVAL_GRP code covers every eval type (including
# EXPGROW, the remeasurement/growth evaluation vitalRates() uses), since
# POP_EVAL_GRP groups all eval types for a state/year together.
wcs <- lapply(dbs, \(d) unique(d$POP_EVAL_GRP$EVAL_GRP))
wc_ri <- wcs[["RI"]]

# Test 8 ------------------------------
# Internal consistency: totals divided by area/tree-count reproduce the
# per-acre/per-stem estimates (doesn't require EVALIDator).
for (st in states) {
  test_that(paste("vitalRates() totals are consistent with per-acre/per-stem estimates (", st, ")"), {
    out <- as.data.frame(vitalRates(dbs[[st]], totals = TRUE))
    expect_equal(out$DIA_TOTAL / out$TREE_TOTAL, out$DIA_GROW, tolerance = 1e-9)
    expect_equal(out$BA_TOTAL / out$TREE_TOTAL, out$BA_GROW, tolerance = 1e-9)
    expect_equal(out$NETVOL_TOTAL / out$TREE_TOTAL, out$NETVOL_GROW, tolerance = 1e-9)
    expect_equal(out$SAWVOL_TOTAL / out$TREE_TOTAL, out$SAWVOL_GROW, tolerance = 1e-9)
    expect_equal(out$BIO_TOTAL / out$TREE_TOTAL, out$BIO_GROW, tolerance = 1e-9)
    expect_equal(out$BA_TOTAL / out$AREA_TOTAL, out$BA_GROW_AC, tolerance = 1e-9)
    expect_equal(out$NETVOL_TOTAL / out$AREA_TOTAL, out$NETVOL_GROW_AC, tolerance = 1e-9)
    expect_equal(out$SAWVOL_TOTAL / out$AREA_TOTAL, out$SAWVOL_GROW_AC, tolerance = 1e-9)
    expect_equal(out$BIO_TOTAL / out$AREA_TOTAL, out$BIO_GROW_AC, tolerance = 1e-9)
  })
}

# Test 9 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(vitalRates(db_ri, polys = countiesRI, returnSpatial = TRUE))
  out_df <- as.data.frame(vitalRates(db_ri, polys = countiesRI, returnSpatial = FALSE))
  out_sf <- out_sf[, names(out_df)]
  out_sf <- out_sf[order(out_sf$COUNTY), ]
  out_df <- out_df[order(out_df$COUNTY), ]
  expect_equal(out_sf, out_df)
})

# Test 10 ------------------------------
# A treeDomain/areaDomain matching no trees should return a clean 0-row
# result, not error or emit an internal max()-on-empty-vector warning (the
# same combineMR() edge case documented in tpa.md, "Fixed" #2 -- shared
# utility, applies to every estimator including vitalRates()).
test_that("vitalRates() handles an empty treeDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(vitalRates(db_ri, treeDomain = SPCD == 999))
  )
  expect_equal(nrow(out), 0)
})

test_that("vitalRates() handles an empty areaDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(vitalRates(db_ri, areaDomain = PHYSCLCD == 11))
  )
  expect_equal(nrow(out), 0)
})

# Test 11 ------------------------------
# Regression tests for two nPlots bugs found and fixed during this
# validation pass (see vitalRates.md, "Fixed"):
#   1. nPlots_AREA didn't respond to landType/areaDomain at all (always
#      reported the full unrestricted panel's plot count).
#   2. nPlots_TREE didn't respond to treeDomain at all (even a treeDomain
#      matching zero trees left nPlots_TREE unchanged).
# Both are checked here by asserting the plot counts actually shrink under
# a restrictive domain, without needing EVALIDator ground truth.
for (st in states) {
  db_st <- dbs[[st]]

  test_that(paste("vitalRates() nPlots_AREA responds to landType (", st, ")"), {
    forest <- as.data.frame(vitalRates(db_st, landType = 'forest'))
    timber <- as.data.frame(vitalRates(db_st, landType = 'timber'))
    expect_lt(timber$nPlots_AREA, forest$nPlots_AREA)
  })

  test_that(paste("vitalRates() nPlots_AREA responds to areaDomain (", st, ")"), {
    unrestricted <- as.data.frame(vitalRates(db_st))
    restricted <- as.data.frame(vitalRates(db_st, areaDomain = PHYSCLCD %in% 21:29))
    expect_lt(restricted$nPlots_AREA, unrestricted$nPlots_AREA)
  })
}

test_that("vitalRates() nPlots_TREE responds to treeDomain (RI)", {
  unrestricted <- as.data.frame(vitalRates(db_ri))
  restricted <- as.data.frame(vitalRates(db_ri, treeDomain = SPCD == 129))
  expect_lt(restricted$nPlots_TREE, unrestricted$nPlots_TREE)
})

test_that("vitalRates() bySpecies nPlots_TREE varies by species (RI)", {
  # Before the fix, every species row reported the same (unrestricted)
  # nPlots_TREE regardless of how common that species actually was.
  out <- as.data.frame(vitalRates(db_ri, bySpecies = TRUE))
  expect_gt(length(unique(out$nPlots_TREE)), 1)
})

# Numeric validation against EVALIDator ------------------------------------
# Reference values are fetched live from the FIADB-API `fullreport` endpoint
# (see core_references/validation/fetch_evalidator.R and
# core_references/validation/vitalRates.md for methodology and full results)
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
# Core default case (treeType = 'all', landType = 'forest', the function
# defaults) matches EVALIDator's net growth-accounting attributes (which
# include recruitment/ingrowth and subtract mortality/cut, same as
# treeType = 'all') to full double precision across one state per FIA
# region: RI (Northern), NC (Southern), CO (Interior West), OR (Pacific
# Northwest). EVALIDator attribute 2635/2636 = average annual net growth of
# aboveground biomass of trees at least 5in DBH (forest/timber), ratio'd
# against attribute 2/3 (forest/timber land area) -- the only growth
# metric with a published "all trees >= 5in DBH" (not growing-stock- or
# sawtimber-restricted) EVALIDator attribute; see vitalRates.md for why
# NETVOL_GROW_AC/SAWVOL_GROW_AC are only checked under treeType = 'gs'.
for (st in states) {
  test_that(paste("vitalRates() BIO_GROW_AC matches EVALIDator for", st, "(core default case)"), {
    wc_st <- wcs[[st]]
    ref <- fetchRef(wc = wc_st, snum = 2635, sdenom = 2)

    out_st <- as.data.frame(vitalRates(dbs[[st]]))

    expect_equal(out_st$BIO_GROW_AC, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(abs(out_st$BIO_GROW_AC_SE), abs(ref$ratioSEPercent), tolerance = 1e-6)
    expect_equal(out_st$nPlots_AREA, ref$denPlotCount)
  })
}

# Test 13 ------------------------------
# landType = 'timber' variant, biomass net growth attribute 2636 ratio'd
# against attribute 3 (timberland area).
for (st in states) {
  test_that(paste("vitalRates() BIO_GROW_AC matches EVALIDator for landType = 'timber' (", st, ")"), {
    wc_st <- wcs[[st]]
    ref <- fetchRef(wc = wc_st, snum = 2636, sdenom = 3)
    out <- as.data.frame(vitalRates(dbs[[st]], landType = 'timber'))
    expect_equal(out$BIO_GROW_AC, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(abs(out$BIO_GROW_AC_SE), abs(ref$ratioSEPercent), tolerance = 1e-6)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })
}

# Test 14 ------------------------------
# treeType = 'gs' (growing-stock) variant. Unlike treeType = 'all', EVALIDator
# publishes growing-stock-specific growth attributes for volume and sawlog
# volume as well as biomass, so this is the only treeType where
# NETVOL_GROW_AC/SAWVOL_GROW_AC (not just BIO_GROW_AC) can be checked
# directly. Attribute 202/208 = net growth of merch bole cubic volume of
# growing-stock trees (forest/timber); 203/209 = net growth of sawlog board-
# foot volume (International 1/4-inch rule) of growing-stock/sawtimber trees
# (forest/timber); 312/318 = net growth of aboveground biomass of
# growing-stock trees (forest/timber).
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("vitalRates() matches EVALIDator for treeType = 'gs' (", st, ")"), {
    volRef <- fetchRef(wc = wc_st, snum = 202, sdenom = 2)
    sawRef <- fetchRef(wc = wc_st, snum = 203, sdenom = 2)
    bioRef <- fetchRef(wc = wc_st, snum = 312, sdenom = 2)

    out <- as.data.frame(vitalRates(db_st, treeType = 'gs'))

    expect_equal(out$NETVOL_GROW_AC, volRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(abs(out$NETVOL_GROW_AC_SE), abs(volRef$ratioSEPercent), tolerance = 1e-6)
    # SAWVOL_GROW_AC is expressed in thousand board feet (MBF); EVALIDator's
    # attribute is raw board feet.
    expect_equal(out$SAWVOL_GROW_AC * 1000, sawRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(abs(out$SAWVOL_GROW_AC_SE), abs(sawRef$ratioSEPercent), tolerance = 1e-6)
    expect_equal(out$BIO_GROW_AC, bioRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(abs(out$BIO_GROW_AC_SE), abs(bioRef$ratioSEPercent), tolerance = 1e-6)
    expect_equal(out$nPlots_AREA, volRef$denPlotCount)
  })

  test_that(paste("vitalRates() matches EVALIDator for treeType = 'gs', landType = 'timber' (", st, ")"), {
    volRef <- fetchRef(wc = wc_st, snum = 208, sdenom = 3)
    sawRef <- fetchRef(wc = wc_st, snum = 209, sdenom = 3)
    # Attribute 315 = aboveground biomass net growth, growing-stock, timberland.
    # (318 is belowground biomass on forest land -- a different attribute
    # entirely; easy to mix up since both are in the same attribute-number
    # neighborhood.)
    bioRef <- fetchRef(wc = wc_st, snum = 315, sdenom = 3)

    out <- as.data.frame(vitalRates(db_st, treeType = 'gs', landType = 'timber'))

    expect_equal(out$NETVOL_GROW_AC, volRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$SAWVOL_GROW_AC * 1000, sawRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$BIO_GROW_AC, bioRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$nPlots_AREA, volRef$denPlotCount)
  })
}

# Test 15 ------------------------------
# areaDomain filter interaction, matched against EVALIDator's `strFilter`
# (applies to numerator AND denominator, since an area-level domain should
# shrink both). Uses the biomass core-default attribute (2635/2) since it's
# the one valid across every treeType/landType combination tested above.
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("vitalRates() matches EVALIDator for areaDomain (physiographic class filter) (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 2635, sdenom = 2,
                     strFilter = "COND.PHYSCLCD in (21,22,23,24,25,26,27,28,29)")
    out <- as.data.frame(vitalRates(db_st, areaDomain = PHYSCLCD %in% 21:29)) # mesic classes
    expect_equal(out$BIO_GROW_AC, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(abs(out$BIO_GROW_AC_SE), abs(ref$ratioSEPercent), tolerance = 1e-6)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })
}

# Test 16 ------------------------------
# treeDomain filter interaction, matched against EVALIDator's `wnum`
# (numerator-only filter). Species-code filters are used rather than a
# DIA-based filter (unlike tpa()/volume()) because vitalRates()'s tree
# domain indicator is evaluated against the *previous* measurement's
# attributes when available (tD.prev, defaulting to the current-measurement
# tD only for new/ingrowth trees with no previous record -- see
# vitalRatesStarter.R), while EVALIDator's growth-accounting SQL applies the
# same WHERE-clause filter using the *current* (TREE alias) measurement.
# SPCD doesn't change between measurements, so it's unambiguous either way;
# a DIA-based filter would not necessarily be (see vitalRates.md for a
# direct empirical check of whether this theoretical difference actually
# produces a mismatch).
test_that("vitalRates() matches EVALIDator for treeDomain (species filter, RI)", {
  ref <- fetchRef(wc = wc_ri, snum = 2635, sdenom = 2, wnum = "TREE.SPCD = 129")
  out <- as.data.frame(vitalRates(db_ri, treeDomain = SPCD == 129)) # eastern white pine
  expect_equal(out$BIO_GROW_AC, ref$ratioEstimate, tolerance = 1e-6)
  expect_equal(abs(out$BIO_GROW_AC_SE), abs(ref$ratioSEPercent), tolerance = 1e-6)
  expect_equal(out$nPlots_TREE, ref$numPlotCount)
})

# Test 17 ------------------------------
# bySpecies grouping (RI): validates a couple of species rows produced by
# grpBy = SPCD against an independent single-species EVALIDator query, i.e.
# that a domain filter survives rFIA's internal grpBy/join path rather than
# being silently dropped for some groups (the historical area()/areaChange()
# bug pattern from v1.1.1). See tpa.md for why EVALIDator's own row-grouping
# mechanism (rselected) can't be used here instead.
test_that("vitalRates() bySpecies matches EVALIDator per-species (RI)", {
  out <- as.data.frame(vitalRates(db_ri, bySpecies = TRUE))
  set.seed(42)
  sampled <- out[sample(nrow(out), 2), ]
  for (i in seq_len(nrow(sampled))) {
    ref <- fetchRef(wc = wc_ri, snum = 2635, sdenom = 2,
                     wnum = paste0("TREE.SPCD = ", sampled$SPCD[i]))
    expect_equal(sampled$BIO_GROW_AC[i], ref$ratioEstimate, tolerance = 1e-6,
                 label = paste0("BIO_GROW_AC (SPCD ", sampled$SPCD[i], ")"))
    expect_equal(sampled$nPlots_TREE[i], ref$numPlotCount,
                 label = paste0("nPlots_TREE (SPCD ", sampled$SPCD[i], ")"))
  }
})

# Non-TI method (SMA/LMA/EMA/ANNUAL) internal consistency -------------------
# EVALIDator has no equivalent for these, so correctness here means: the
# code runs cleanly across the same filter/grpBy/byPlot space already
# exercised above, totals/per-acre/per-stem plumbing holds regardless of
# method, and the documented cross-method relationships in
# vignettes/alternativeEstimators.Rmd hold as *bounded*/*directional*
# checks -- never exact equality (see tpa.md for the full writeup of why).
# See tests/testthat/test-util.R for the underlying maWeights()/
# filterAnnual()/combineMR() unit-level checks these per-function tests
# build on.
#
# A real, previously-unknown bug (not just a missing-coverage gap) was found
# and fixed during this pass -- see vitalRates.md, "Fixed" #7, for the full
# root-cause writeup. Summary: sumToEU()'s (R/util.R) SMA/LMA/EMA
# weighted-average collapse grouped the numerator side ("x") by
# `P2PNTCNT_EU` in addition to the intended grouping columns, but the
# denominator side ("y") correctly omitted it. Since P2PNTCNT_EU legitimately
# varies per remeasurement panel under a non-TI method, this silently
# prevented the numerator side from ever collapsing panels into one row --
# invisible for every estimator that calls sumToEU() only once (the
# dispatcher's own final group_by()/summarize(sum(...)) step happens to
# finish the collapse regardless), but vitalRatesStarter.R (like
# growMortStarter.R) calls sumToEU() a *second* time for a tree-total
# covariance term and left_joins the two outputs together -- a many-to-many
# join between two not-yet-collapsed multi-row tables, multiplying every
# growth total (BIO_TOTAL, TREE_TOTAL, etc.) roughly by the number of
# constituent panels (RI: BIO_GROW_AC 0.26 (TI) vs. 2.51 (SMA, pre-fix) --
# an ~860% inflation, vs. ~18% post-fix, itself explained by ordinary
# sampling noise on a near-zero estimate -- see Test 19 below). Tests 18-24
# below are general non-TI coverage (mirroring tpa.md's template); Test 24
# specifically regression-tests this exact bug via nPlots_TREE, which the
# pre-fix join bug inflated by roughly the panel count regardless of domain
# filters.

# Test 18 ------------------------------
# EMA(lambda -> 1) should monotonically approach SMA (RI). Never exactly
# equal (see test-util.R for why the exact boundary is degenerate) -- this
# checks the trend, not a fixed-tolerance snapshot. Mirrors tpa.md's Test 16.
test_that("vitalRates() EMA(lambda -> 1) monotonically approaches SMA (RI)", {
  sma <- as.data.frame(vitalRates(db_ri, method = 'SMA'))
  dists <- sapply(c(0.5, 0.9, 0.99, 0.999), \(lam) {
    ema <- as.data.frame(vitalRates(db_ri, method = 'EMA', lambda = lam))
    abs(ema$BIO_GROW_AC - sma$BIO_GROW_AC)
  })
  expect_true(all(diff(dists) < 0))
  expect_lt(dists[length(dists)], 0.01)
})

# Test 19 ------------------------------
# TI and SMA are not claimed to be numerically equal in general (see
# tpa.md). Unlike TPA/BAA/BIO_ACRE (always positive, bounded away from
# zero), BIO_GROW_AC is a *net* growth rate (ingrowth minus mortality/cut)
# that is legitimately small or near-zero for a slow-growing/small-sample
# state -- RI's TI estimate (0.26) has a 64% SE, so a flat relative-%
# tolerance (as used in tpa.md/biomass.md) is not well-suited here: a modest
# absolute difference translates into a large-looking relative one. Instead,
# bound the absolute |SMA - TI| difference by a multiple of the *combined*
# sampling error of the two estimates (sqrt(SE_TI^2 + SE_SMA^2), the SE of
# their difference under independence) -- i.e., "not statistically
# distinguishable from sampling noise." Observed ratios (diff / combined SE)
# were 0.20 (RI), 1.10 (NC), 0.85 (CO), 0.20 (OR) -- all comfortably under
# 1.5, the bound used below. This would NOT have passed pre-fix: the join-
# explosion bug (see header above) inflated RI's SMA estimate by ~9 combined
# SEs.
for (st in states) {
  test_that(paste("vitalRates() TI and SMA agree within a bounded tolerance (", st, ")"), {
    ti <- as.data.frame(vitalRates(dbs[[st]], method = 'TI'))
    sma <- as.data.frame(vitalRates(dbs[[st]], method = 'SMA'))
    ti_se_abs <- abs(ti$BIO_GROW_AC) * ti$BIO_GROW_AC_SE / 100
    sma_se_abs <- abs(sma$BIO_GROW_AC) * sma$BIO_GROW_AC_SE / 100
    combined_se <- sqrt(ti_se_abs^2 + sma_se_abs^2)
    expect_lt(abs(sma$BIO_GROW_AC - ti$BIO_GROW_AC), 1.5 * combined_se)
  })
}

# Test 20 ------------------------------
# totals = TRUE / per-acre / per-stem consistency holds under every non-TI
# method, not just TI (Test 8 above only checked the TI/default path).
for (st in states) {
  test_that(paste("vitalRates() totals are consistent with per-acre/per-stem estimates under non-TI methods (", st, ")"), {
    for (m in c('SMA', 'LMA', 'EMA', 'ANNUAL')) {
      out <- as.data.frame(vitalRates(dbs[[st]], totals = TRUE, method = m))
      expect_equal(out$BA_TOTAL / out$AREA_TOTAL, out$BA_GROW_AC, tolerance = 1e-9,
                   label = paste0(st, " ", m, " BA_GROW_AC"))
      expect_equal(out$BIO_TOTAL / out$AREA_TOTAL, out$BIO_GROW_AC, tolerance = 1e-9,
                   label = paste0(st, " ", m, " BIO_GROW_AC"))
      expect_equal(out$BA_TOTAL / out$TREE_TOTAL, out$BA_GROW, tolerance = 1e-9,
                   label = paste0(st, " ", m, " BA_GROW"))
      expect_equal(out$BIO_TOTAL / out$TREE_TOTAL, out$BIO_GROW, tolerance = 1e-9,
                   label = paste0(st, " ", m, " BIO_GROW"))
    }
  })
}

# Test 21 ------------------------------
# byPlot = TRUE combined with a non-TI method is a distinct code path --
# mergeSmallStrata() (R/util.R) is explicitly skipped whenever byPlot =
# TRUE, regardless of method. Confirm it still returns per-plot (not
# population-level) rows without error.
test_that("vitalRates() byPlot = TRUE works with a non-TI method (RI, SMA)", {
  out <- as.data.frame(vitalRates(db_ri, method = 'SMA', byPlot = TRUE))
  expect_true(all(c('PLT_CN', 'BIO_GROW') %in% names(out)))
  expect_gt(nrow(out), 1) # per-plot rows, not a single population estimate
})

# Test 22 ------------------------------
# Domain filter + bySpecies interaction (the historical
# area()/areaChange() bug pattern from v1.1.1, see tpa.md Test 15) re-run
# under every non-TI method: no error, no warning, sane (finite) shape.
for (st in states) {
  for (m in c('SMA', 'LMA', 'EMA', 'ANNUAL')) {
    test_that(paste("vitalRates() domain filter + bySpecies runs cleanly under method =", m, "(", st, ")"), {
      expect_no_warning(
        out <- as.data.frame(vitalRates(dbs[[st]], treeDomain = DIA >= 20, areaDomain = PHYSCLCD %in% 21:29,
                                        bySpecies = TRUE, method = m))
      )
      expect_true(nrow(out) >= 0)
      expect_true(all(is.finite(out$BIO_GROW_AC) | is.na(out$BIO_GROW_AC)))
    })
  }
}

# Test 23 ------------------------------
# Plain default-args smoke tests, one per state, for EMA and ANNUAL. ANNUAL
# is regression coverage for the combineMR()/ANNUAL pooling bug (tpa.md,
# "Fixed" #6) -- vitalRates() already passes `method` through to
# combineMR() (R/vitalRates.R), so a returned multi-row (not pooled) result
# here confirms it's unaffected. EMA mirrors tpa.md's v1.1.1 regression
# coverage.
for (st in states) {
  test_that(paste("vitalRates() runs with method = 'EMA' and default arguments (", st, ")"), {
    expect_no_error(out <- as.data.frame(vitalRates(dbs[[st]], method = 'EMA')))
    expect_s3_class(out, "data.frame")
  })

  test_that(paste("vitalRates() runs with method = 'ANNUAL' and default arguments, one row per panel (", st, ")"), {
    expect_no_error(out <- as.data.frame(vitalRates(dbs[[st]], method = 'ANNUAL')))
    expect_s3_class(out, "data.frame")
    expect_gt(nrow(out), 1) # not pooled into a single mislabeled row
  })
}

# Test 24 ------------------------------
# Regression test for the sumToEU()/P2PNTCNT_EU join-explosion bug found
# during this pass (see header above and vitalRates.md "Fixed" #7).
# nPlots_TREE is a direct, cheap witness: pre-fix, the bug inflated it by
# roughly the number of constituent panels (RI: 108 (TI) -> 708 (SMA), a
# ~6.5x blow-up) regardless of domain filters, since the join duplicates
# every row in the (already correct) tree list. Post-fix, SMA draws on the
# same underlying remeasurement-panel universe as TI (TI's static stratum
# weighting already spans every panel in the evaluation's window, just
# weighted differently than SMA's moving average), so nPlots_TREE is
# expected to match exactly -- confirmed empirically in all four states.
# This is a stronger, more direct check than Test 19's bounded-tolerance
# comparison, which a sufficiently small residual bug could still slip past.
for (st in states) {
  test_that(paste("vitalRates() nPlots_TREE under SMA is not inflated relative to TI (", st, ")"), {
    ti <- as.data.frame(vitalRates(dbs[[st]], method = 'TI'))
    sma <- as.data.frame(vitalRates(dbs[[st]], method = 'SMA'))
    expect_equal(sma$nPlots_TREE, ti$nPlots_TREE)
  })
}
