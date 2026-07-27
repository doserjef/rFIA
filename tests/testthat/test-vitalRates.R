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
