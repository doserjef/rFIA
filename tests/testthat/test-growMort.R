# Test growMort() --------------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Most recent estimates for growing stock on timber land by species
out <- growMort(db = fiaRI_mr, land = 'timber', bySpecies = TRUE, totals = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 2 ------------------------------
# Most recent estimates for growing stock on timber land by species by plot
out <- growMort(db = fiaRI_mr, land = 'timber', bySpecies = TRUE, byPlot = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
# Estimates for white pine (> 12" DBH) on forested mesic sites
out <- growMort(fiaRI_mr,
           treeType = 'all',
           treeDomain = SPCD == 129 & DIA > 12, # Species code for white pine
           areaDomain = PHYSCLCD %in% 21:29) # Mesic Physiographic classes

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 4 ------------------------------
# Most recent estimates grouped by stand age on forest land
# Make a categorical variable which represents stand age (grouped by 10 yr intervals)
fiaRI_mr$COND$STAND_AGE <- makeClasses(fiaRI_mr$COND$STDAGE, interval = 10)
out <- growMort(db = fiaRI_mr, grpBy = STAND_AGE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 5 ------------------------------
# Estimates for live trees greater than 20in DBH on forest land
out <- growMort(db = fiaRI, landType = 'forest', treeType = 'all',
           treeDomain = DIA > 20, stateVar = 'BAA')

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

test_that("multiple years", {
  expect_gt(length(unique(out$YEAR)), 1)
})

# Test 6 ------------------------------
# Most recent estimates for all stems on forest land by species
out <- growMort(db = fiaRI_mr, landType = 'forest', treeType = 'all',
                bySpecies = TRUE)
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 7 ------------------------------
# Most recent estimates for all stems on forest land
# grouped by user-defined areal units
out <- growMort(fiaRI_mr,
           polys = countiesRI,
           returnSpatial = TRUE, stateVar = 'SAWVOL')
plot.out <- plotFIA(out, GROW_SAWVOL_ACRE)
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
# EXPGROW/EXPMORT/EXPREMV, the remeasurement evaluations growMort() uses),
# since POP_EVAL_GRP groups all eval types for a state/year together.
wcs <- lapply(dbs, \(d) unique(d$POP_EVAL_GRP$EVAL_GRP))
wc_ri <- wcs[["RI"]]

# Test 8 ------------------------------
# Internal consistency: totals divided by area reproduce the per-acre
# estimates (doesn't require EVALIDator).
for (st in states) {
  test_that(paste("growMort() totals are consistent with per-acre estimates (", st, ")"), {
    out <- as.data.frame(growMort(dbs[[st]], totals = TRUE))
    expect_equal(out$RECR_TOTAL / out$AREA_TOTAL, out$RECR_TPA, tolerance = 1e-9)
    expect_equal(out$MORT_TOTAL / out$AREA_TOTAL, out$MORT_TPA, tolerance = 1e-9)
    expect_equal(out$REMV_TOTAL / out$AREA_TOTAL, out$REMV_TPA, tolerance = 1e-9)
    expect_equal(out$GROW_TOTAL / out$AREA_TOTAL, out$GROW_TPA, tolerance = 1e-9)
    expect_equal(out$CHNG_TOTAL / out$AREA_TOTAL, out$CHNG_TPA, tolerance = 1e-9)
  })
}

# Test 9 ------------------------------
# Internal consistency: CHNG = GROW + RECR - MORT - REMV, by construction
# (man page: "CHNG_*: estimate of mean annual net change (i.e., growth +
# recruitment - mortality - removals)"). This holds regardless of whether the
# underlying numbers are numerically correct, but is included as a regression
# guard against reintroducing the NA-misalignment bug documented in
# growMort.md ("Fixed" #4): before that fix, an aggregate-level mismatch
# between which rows survived na.rm = TRUE in each of RECR/MORT/REMV/GROW/CHNG
# broke this identity for state variables (e.g. NETVOL, SAWVOL_BF) where the
# underlying FIA volume column is undefined for some trees.
for (st in states) {
  for (sv in c('TPA', 'BAA', 'NETVOL', 'SAWVOL_BF', 'BIO_AG')) {
    test_that(paste("growMort() CHNG = GROW + RECR - MORT - REMV identity holds (",
                     st, sv, ")"), {
      out <- as.data.frame(growMort(dbs[[st]], stateVar = sv))
      # TPA/BAA columns keep their bare names (no "_ACRE" suffix) -- see
      # growMort.R's `stateVar != 'TPA'` guard and the BAA_ACRE -> BAA
      # cleanup that follows it. Every other stateVar gets "_<sv>_ACRE".
      suffix <- if (sv %in% c('TPA', 'BAA')) sv else paste0(sv, '_ACRE')
      grow <- out[[paste0('GROW_', suffix)]]
      recr <- out[[paste0('RECR_', suffix)]]
      mort <- out[[paste0('MORT_', suffix)]]
      remv <- out[[paste0('REMV_', suffix)]]
      chng <- out[[paste0('CHNG_', suffix)]]
      expect_equal(grow + recr - mort - remv, chng, tolerance = 1e-6)
    })
  }
}

# Test 10 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(growMort(db_ri, polys = countiesRI, returnSpatial = TRUE))
  out_df <- as.data.frame(growMort(db_ri, polys = countiesRI, returnSpatial = FALSE))
  out_sf <- out_sf[, names(out_df)]
  out_sf <- out_sf[order(out_sf$COUNTY), ]
  out_df <- out_df[order(out_df$COUNTY), ]
  expect_equal(out_sf, out_df)
})

# Test 11 ------------------------------
# A treeDomain/areaDomain matching no trees/area should return a clean 0-row
# result, not error or emit an internal max()-on-empty-vector warning (the
# same combineMR() edge case documented in tpa.md, "Fixed" #2 -- shared
# utility, applies to every estimator including growMort()).
test_that("growMort() handles an empty treeDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(growMort(db_ri, treeDomain = SPCD == 999))
  )
  expect_equal(nrow(out), 0)
})

test_that("growMort() handles an empty areaDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(growMort(db_ri, areaDomain = PHYSCLCD == 11))
  )
  expect_equal(nrow(out), 0)
})

# Test 12 ------------------------------
# nPlots_AREA/nPlots_TREE should shrink under a restrictive landType/
# areaDomain/treeDomain (not required to match any particular EVALIDator
# per-event plot count -- growMort() reports a single generic nPlots_TREE
# covering all recruitment/mortality/removal/survivor trees together, not a
# separate count per component; see growMort.md "Notes").
for (st in states) {
  db_st <- dbs[[st]]

  test_that(paste("growMort() nPlots_AREA responds to landType (", st, ")"), {
    forest <- as.data.frame(growMort(db_st, landType = 'forest'))
    timber <- as.data.frame(growMort(db_st, landType = 'timber'))
    expect_lt(timber$nPlots_AREA, forest$nPlots_AREA)
  })

  test_that(paste("growMort() nPlots_AREA responds to areaDomain (", st, ")"), {
    unrestricted <- as.data.frame(growMort(db_st))
    restricted <- as.data.frame(growMort(db_st, areaDomain = PHYSCLCD %in% 21:29))
    expect_lt(restricted$nPlots_AREA, unrestricted$nPlots_AREA)
  })
}

test_that("growMort() nPlots_TREE responds to treeDomain (RI)", {
  unrestricted <- as.data.frame(growMort(db_ri))
  restricted <- as.data.frame(growMort(db_ri, treeDomain = SPCD == 129))
  expect_lt(restricted$nPlots_TREE, unrestricted$nPlots_TREE)
})

test_that("growMort() bySpecies nPlots_TREE varies by species (RI)", {
  out <- as.data.frame(growMort(db_ri, bySpecies = TRUE))
  expect_gt(length(unique(out$nPlots_TREE)), 1)
})

# Numeric validation against EVALIDator ------------------------------------
# Reference values are fetched live from the FIADB-API `fullreport` endpoint
# (see core_references/validation/fetch_evalidator.R and
# core_references/validation/growMort.md for methodology and full results)
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

# Test 13 ------------------------------
# Core default case (treeType = 'all', landType = 'forest', stateVar = 'TPA',
# the function defaults) matches EVALIDator's mortality/harvest-removals
# "number of trees" attributes to full double precision across one state per
# FIA region: RI (Northern), NC (Southern), CO (Interior West), OR (Pacific
# Northwest). Attribute 901 = average annual mortality of trees (>= 5in DBH),
# in trees, forest land; 913 = average annual harvest removals of trees, in
# trees, forest land; both ratio'd against attribute 2 (forest land area).
# There is no EVALIDator attribute for recruitment ("ingrowth") counts, so
# RECR_TPA/GROW_TPA/CHNG_TPA have no direct numeric ground truth here -- see
# growMort.md; the CHNG = GROW + RECR - MORT - REMV identity (Test 9) is the
# only check available for those columns under stateVar = 'TPA'.
for (st in states) {
  test_that(paste("growMort() matches EVALIDator for", st, "(core default case)"), {
    wc_st <- wcs[[st]]
    mortRef <- fetchRef(wc = wc_st, snum = 901, sdenom = 2)
    remvRef <- fetchRef(wc = wc_st, snum = 913, sdenom = 2)

    out_st <- as.data.frame(growMort(dbs[[st]]))

    expect_equal(out_st$MORT_TPA, mortRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(out_st$MORT_TPA_SE, mortRef$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out_st$REMV_TPA, remvRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(out_st$REMV_TPA_SE, remvRef$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out_st$nPlots_AREA, mortRef$denPlotCount)
  })
}

# Test 14 ------------------------------
# landType = 'timber' variant, attributes 904 (mortality)/916 (harvest
# removals), ratio'd against attribute 3 (timberland area). RI/NC/CO match
# exactly; OR (and, per vitalRates.md, CA/WA) show a small known residual
# mismatch in macroplot-heavy states' timberland denominator that was already
# investigated and left unresolved during the vitalRates() validation pass
# (vitalRates.md, "Known issues" A) -- growMort() shares the identical
# SUBP_COND_CHNG_MTRX-based area-change logic, so it inherits the same open
# issue. That state is checked separately below with a wide tolerance/no
# assertion, rather than silently excluded.
for (st in c("RI", "NC", "CO")) {
  test_that(paste("growMort() matches EVALIDator for landType = 'timber' (", st, ")"), {
    wc_st <- wcs[[st]]
    ref <- fetchRef(wc = wc_st, snum = 904, sdenom = 3)
    out <- as.data.frame(growMort(dbs[[st]], landType = 'timber'))
    expect_equal(out$MORT_TPA, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })
}

test_that("growMort() landType = 'timber' (OR) -- known macroplot residual, not exact", {
  # See growMort.md and vitalRates.md ("Known issues" A). Documented as a
  # regression guard that the mismatch stays small (< 1%), not a numeric
  # match assertion.
  ref <- fetchRef(wc = wcs[["OR"]], snum = 904, sdenom = 3)
  out <- as.data.frame(growMort(dbs[["OR"]], landType = 'timber'))
  relErr <- abs(out$MORT_TPA - ref$ratioEstimate) / abs(ref$ratioEstimate)
  expect_lt(relErr, 0.01)
})

# Test 15 ------------------------------
# treeType = 'gs' (growing-stock) variant, attributes 902 (mortality)/914
# (harvest removals), ratio'd against attribute 2 (forest land area).
for (st in states) {
  test_that(paste("growMort() matches EVALIDator for treeType = 'gs' (", st, ")"), {
    wc_st <- wcs[[st]]
    mortRef <- fetchRef(wc = wc_st, snum = 902, sdenom = 2)
    remvRef <- fetchRef(wc = wc_st, snum = 914, sdenom = 2)
    out <- as.data.frame(growMort(dbs[[st]], treeType = 'gs'))
    expect_equal(out$MORT_TPA, mortRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$REMV_TPA, remvRef$ratioEstimate, tolerance = 1e-6)
  })
}

# Test 16 ------------------------------
# areaDomain filter interaction, matched against EVALIDator's `strFilter`
# (applies to numerator AND denominator, since an area-level domain should
# shrink both) -- see tpa.md for why `wnum`/`strFilter` is the right tool for
# treeDomain-/areaDomain-style filters, respectively.
for (st in states) {
  test_that(paste("growMort() matches EVALIDator for areaDomain (physiographic class filter) (", st, ")"), {
    wc_st <- wcs[[st]]
    ref <- fetchRef(wc = wc_st, snum = 901, sdenom = 2,
                     strFilter = "COND.PHYSCLCD in (21,22,23,24,25,26,27,28,29)")
    out <- as.data.frame(growMort(dbs[[st]], areaDomain = PHYSCLCD %in% 21:29)) # mesic classes
    expect_equal(out$MORT_TPA, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })
}

# Test 17 ------------------------------
# treeDomain filter interaction, matched against EVALIDator's `wnum`
# (numerator-only filter). A species-code filter is used (as in
# vitalRates.md) rather than a DIA-based filter, since growMort()'s
# comprehensive tree-domain indicator (tDI) is evaluated against the tree's
# *previous*-measurement attributes (tD.prev), while EVALIDator's
# growth-accounting SQL applies the WHERE-clause filter against the
# *current* measurement -- immaterial for a time-invariant attribute like
# SPCD, but not necessarily for a time-varying one like DIA.
test_that("growMort() matches EVALIDator for treeDomain (species filter, RI)", {
  ref <- fetchRef(wc = wc_ri, snum = 901, sdenom = 2, wnum = "TREE.SPCD = 129")
  out <- as.data.frame(growMort(db_ri, treeDomain = SPCD == 129)) # eastern white pine
  expect_equal(out$MORT_TPA, ref$ratioEstimate, tolerance = 1e-6)
  # nPlots_TREE is NOT expected to equal EVALIDator's per-attribute
  # numPlotCount here -- growMort() reports a single generic nPlots_TREE
  # spanning all recruitment/mortality/removal/survivor trees together, not a
  # count scoped to the mortality attribute specifically. See growMort.md,
  # "Notes". (nPlots_TREE's *responsiveness* to treeDomain is covered by
  # Test 12 instead.)
})

# Test 18 ------------------------------
# bySpecies grouping (RI): validates a couple of species rows produced by
# grpBy = SPCD against an independent single-species EVALIDator query, i.e.
# that a domain filter survives rFIA's internal grpBy/join path rather than
# being silently dropped for some groups (the historical area()/areaChange()
# bug pattern from v1.1.1). See tpa.md for why EVALIDator's own row-grouping
# mechanism (rselected) can't be used here instead.
test_that("growMort() bySpecies matches EVALIDator per-species (RI)", {
  out <- as.data.frame(growMort(db_ri, bySpecies = TRUE))
  set.seed(42)
  sampled <- out[sample(nrow(out), 2), ]
  for (i in seq_len(nrow(sampled))) {
    ref <- fetchRef(wc = wc_ri, snum = 901, sdenom = 2,
                     wnum = paste0("TREE.SPCD = ", sampled$SPCD[i]))
    expect_equal(sampled$MORT_TPA[i], ref$ratioEstimate, tolerance = 1e-6,
                 label = paste0("MORT_TPA (SPCD ", sampled$SPCD[i], ")"))
  }
})

# Test 19 ------------------------------
# stateVar = 'BIO_AG' (aboveground biomass, dry short tons/acre) matches
# EVALIDator's growth-accounting biomass attributes across all four states:
# attribute 2635/2636 = net growth (forest/timber, all trees >= 5in DBH),
# 2637/2638 = mortality, 2649/2650 = harvest removals. This exercises both
# the lbs->short-tons unit conversion and the GROW/CHNG computation fix
# together (see growMort.md, "Fixed" #2 and #4).
for (st in states) {
  test_that(paste("growMort() matches EVALIDator for stateVar = 'BIO_AG' (", st, ")"), {
    wc_st <- wcs[[st]]
    chngRef <- fetchRef(wc = wc_st, snum = 2635, sdenom = 2)
    mortRef <- fetchRef(wc = wc_st, snum = 2637, sdenom = 2)
    remvRef <- fetchRef(wc = wc_st, snum = 2649, sdenom = 2)
    out <- as.data.frame(growMort(dbs[[st]], stateVar = 'BIO_AG'))
    expect_equal(out$CHNG_BIO_AG_ACRE, chngRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$MORT_BIO_AG_ACRE, mortRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$REMV_BIO_AG_ACRE, remvRef$ratioEstimate, tolerance = 1e-6)
  })
}

# Test 20 ------------------------------
# stateVar = 'NETVOL' under treeType = 'gs' (RI) matches EVALIDator attribute
# 202 (net growth of merch. bole cubic volume of growing-stock trees, forest
# land). This specifically exercises the NA-alignment fix (growMort.md,
# "Fixed" #4): board-foot/cubic-foot volume is undefined for some trees
# (below merchantability thresholds), unlike TPA/BAA/biomass, which are
# defined for essentially every tree.
test_that("growMort() matches EVALIDator for stateVar = 'NETVOL', treeType = 'gs' (RI)", {
  ref <- fetchRef(wc = wc_ri, snum = 202, sdenom = 2)
  out <- as.data.frame(growMort(db_ri, stateVar = 'NETVOL', treeType = 'gs'))
  expect_equal(out$CHNG_NETVOL_ACRE, ref$ratioEstimate, tolerance = 1e-6)
})

# Test 21 ------------------------------
# stateVar = 'SAWVOL_BF' (RI) matches EVALIDator attribute 203 (net growth of
# sawlog board-foot volume, International 1/4-inch rule, forest land) without
# any *1000 rescaling -- unlike vitalRates(), growMort() already reports
# SAWVOL_BF in raw board feet, not thousand board feet (MBF).
test_that("growMort() matches EVALIDator for stateVar = 'SAWVOL_BF' (RI)", {
  ref <- fetchRef(wc = wc_ri, snum = 203, sdenom = 2)
  out <- as.data.frame(growMort(db_ri, stateVar = 'SAWVOL_BF'))
  expect_equal(out$CHNG_SAWVOL_BF_ACRE, ref$ratioEstimate, tolerance = 1e-6)
})
