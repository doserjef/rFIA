# Test biomass() ----------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Return aboveground biomass for all forestland by county and return spatial object
out <- biomass(db = fiaRI_mr, polys = countiesRI, returnSpatial = TRUE)
plot.out <- plotFIA(out, BIO_ACRE)
test_that("out is correct", {
  expect_s3_class(out, 'sf')
})
test_that('plot.out is a ggplot', {
  expect_s3_class(plot.out, 'gg')
})

# Test 2 ------------------------------
# Biomass by component for most recent survey on timberland for growing stock trees
out <- biomass(db = fiaRI_mr, byComponent = TRUE, landType = 'timber',
               treeType = 'gs')
test_that('out is correct', {
  expect_s3_class(out, 'tbl_df')
})

# Test 3 ------------------------------
# Biomass for multiple components for multiple species
out <- biomass(db = fiaRI_mr, bySpecies = TRUE, component = c('ROOT', 'FOLIAGE'))
test_that('out is correct', {
  expect_s3_class(out, 'tbl_df')
})

# Test 4 ------------------------------
# AG biomass (including foliage) on timberland for AGS species, by plot
out <- biomass(db = fiaRI_mr, landType = 'timber', treeType = 'gs', component = c('AG', 'FOLIAGE'),
               byPlot = TRUE)
test_that('out is correct', {
  expect_s3_class(out, 'tbl_df')
})

# Test 5 ------------------------------
# Belowground (i.e., coarse roots) and stump biomass only 
out <- biomass(db = fiaRI, component = c('ROOT', 'STUMP'))

test_that('out is correct', {
  expect_s3_class(out, 'tbl_df')
})

# Test 6 ------------------------------
# AGB by size-class
out <- biomass(db = fiaRI, bySizeClass = TRUE)

test_that('out is correct', {
  expect_s3_class(out, 'tbl_df')
})

# Internal consistency checks (no EVALIDator, no network needed) ----------
# These only require the local FIADB extract cache, not network access, so
# they still run when apps.fs.usda.gov is unreachable. See
# core_references/validation/biomass.md for full methodology/results.
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
  out_sf <- as.data.frame(biomass(db_ri, polys = countiesRI, returnSpatial = TRUE))
  out_df <- as.data.frame(biomass(db_ri, polys = countiesRI, returnSpatial = FALSE))
  out_sf <- out_sf[, names(out_df)]
  out_sf <- out_sf[order(out_sf$COUNTY), ]
  out_df <- out_df[order(out_df$COUNTY), ]
  expect_equal(out_sf, out_df)
})

# Test 8 ------------------------------
# component = 'TOTAL' reproduces the sum of the individual components it's
# documented to be composed of (ROOT, STEM, STEM_BARK, BRANCH, FOLIAGE) --
# regression test for the v1.1.3 component = 'TOTAL' double-counting bug.
for (st in states) {
  test_that(paste("biomass() component = 'TOTAL' equals sum of its components (", st, ")"), {
    outTotal <- as.data.frame(biomass(dbs[[st]], component = 'TOTAL'))
    outComp <- as.data.frame(biomass(dbs[[st]], byComponent = TRUE))
    compSum <- sum(outComp$BIO_ACRE[outComp$COMPONENT %in%
                                       c('ROOT', 'STEM', 'STEM_BARK', 'BRANCH', 'FOLIAGE')])
    expect_equal(outTotal$BIO_ACRE, compSum, tolerance = 1e-9)
  })
}

# Test 9 ------------------------------
# Internal consistency: totals divided by area reproduce the per-acre
# estimate (doesn't require EVALIDator).
for (st in states) {
  test_that(paste("biomass() totals are consistent with per-acre estimates (", st, ")"), {
    out <- as.data.frame(biomass(dbs[[st]], totals = TRUE))
    expect_equal(out$BIO_TOTAL / out$AREA_TOTAL, out$BIO_ACRE, tolerance = 1e-9)
  })
}

# Test 10 ------------------------------
# A treeDomain matching no trees should return a clean 0-row result, not
# error or emit an internal max()-on-empty-vector warning.
test_that("biomass() handles an empty treeDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(biomass(db_ri, treeDomain = SPCD == 999))
  )
  expect_equal(nrow(out), 0)
})

# Numeric validation against EVALIDator ------------------------------------
# Reference values are fetched live from the FIADB-API `fullreport` endpoint
# (see core_references/validation/fetch_evalidator.R and
# core_references/validation/biomass.md for methodology and full results)
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

# Test 11 ------------------------------
# Core default case (component = 'AG', treeType = 'live', landType =
# 'forest') matches EVALIDator to full double precision across one state per
# FIA region: RI (Northern), NC (Southern), CO (Interior West), OR (Pacific
# Northwest). EVALIDator attribute 10 = aboveground biomass of live trees on
# forest land, ratio'd against attribute 2 (forest land area). Note that
# rFIA's default component = 'AG' is read directly off FIADB's DRYBIO_AG
# column (not summed from sub-components), which is exactly what attribute
# 10 computes.
for (st in states) {
  test_that(paste("biomass() matches EVALIDator for", st, "(core default case)"), {
    wc_st <- wcs[[st]]
    ref <- fetchRef(wc = wc_st, snum = 10, sdenom = 2)
    out <- as.data.frame(biomass(dbs[[st]]))
    expect_equal(out$BIO_ACRE, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$BIO_ACRE_SE, ref$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out$nPlots_TREE, ref$numPlotCount)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })
}

# Test 12 ------------------------------
# landType/treeType/component variants across all four FIA regions. Matched
# against EVALIDator attributes 13 (timberland/live/AG), 11266 (forest
# land/standing dead (>= 1in)/AG), 59 (forest land/live/belowground, i.e.
# component = 'ROOT'), 11020 (forest land/live/foliage, i.e. component =
# 'FOLIAGE'), 11032 (forest land/live/branch (>= 1in), i.e. component =
# 'BRANCH'). All ratio'd against the matching area attribute (3 =
# timberland, 2 = forest land).
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("biomass() matches EVALIDator for landType = 'timber' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 13, sdenom = 3)
    out <- as.data.frame(biomass(db_st, landType = 'timber'))
    expect_equal(out$BIO_ACRE, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$BIO_ACRE_SE, ref$ratioSEPercent, tolerance = 1e-6)
  })

  test_that(paste("biomass() matches EVALIDator for treeType = 'dead' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 11266, sdenom = 2)
    out <- as.data.frame(biomass(db_st, treeType = 'dead'))
    expect_equal(out$BIO_ACRE, ref$ratioEstimate, tolerance = 1e-6)
  })

  test_that(paste("biomass() matches EVALIDator for component = 'ROOT' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 59, sdenom = 2)
    out <- as.data.frame(biomass(db_st, component = 'ROOT'))
    expect_equal(out$BIO_ACRE, ref$ratioEstimate, tolerance = 1e-6)
  })

  test_that(paste("biomass() matches EVALIDator for component = 'FOLIAGE' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 11020, sdenom = 2)
    out <- as.data.frame(biomass(db_st, component = 'FOLIAGE'))
    expect_equal(out$BIO_ACRE, ref$ratioEstimate, tolerance = 1e-6)
  })

  test_that(paste("biomass() matches EVALIDator for component = 'BRANCH' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 11032, sdenom = 2)
    out <- as.data.frame(biomass(db_st, component = 'BRANCH'))
    expect_equal(out$BIO_ACRE, ref$ratioEstimate, tolerance = 1e-6)
  })
}

# Test 13 ------------------------------
# Merchantable-bole/stump-scale components (STEM, STEM_BARK, STUMP_BARK,
# BOLE + BOLE_BARK) only have an EVALIDator equivalent for timber species >=
# 5in dbh, so treeDomain = DIA >= 5 is added on the rFIA side to make an
# apples-to-apples comparison (this doubles as a treeDomain interaction
# check). EVALIDator attributes 11016 (total-stem wood), 11017 (total-stem
# bark), 11018 (stump bark), 11019 (merch. bole bark); 11000 (merch. bole
# wood + bark combined) is checked against BOLE + BOLE_BARK summed.
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("biomass() matches EVALIDator for component = 'STEM', DIA >= 5 (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 11016, sdenom = 2)
    out <- as.data.frame(biomass(db_st, component = 'STEM', treeDomain = DIA >= 5))
    expect_equal(out$BIO_ACRE, ref$ratioEstimate, tolerance = 1e-6)
  })

  test_that(paste("biomass() matches EVALIDator for component = 'STEM_BARK', DIA >= 5 (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 11017, sdenom = 2)
    out <- as.data.frame(biomass(db_st, component = 'STEM_BARK', treeDomain = DIA >= 5))
    expect_equal(out$BIO_ACRE, ref$ratioEstimate, tolerance = 1e-6)
  })

  test_that(paste("biomass() matches EVALIDator for component = 'STUMP_BARK', DIA >= 5 (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 11018, sdenom = 2)
    out <- as.data.frame(biomass(db_st, component = 'STUMP_BARK', treeDomain = DIA >= 5))
    expect_equal(out$BIO_ACRE, ref$ratioEstimate, tolerance = 1e-6)
  })

  test_that(paste("biomass() matches EVALIDator for component = c('BOLE', 'BOLE_BARK'), DIA >= 5 (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 11000, sdenom = 2)
    outBole <- as.data.frame(biomass(db_st, component = 'BOLE', treeDomain = DIA >= 5))
    outBoleBark <- as.data.frame(biomass(db_st, component = 'BOLE_BARK', treeDomain = DIA >= 5))
    expect_equal(outBole$BIO_ACRE + outBoleBark$BIO_ACRE, ref$ratioEstimate, tolerance = 1e-6)
  })
}

# Test 14 ------------------------------
# Domain filter interactions across all four FIA regions. treeDomain matched
# against EVALIDator's `wnum` (numerator-only filter, since a tree-level
# domain should not change the area denominator); areaDomain matched against
# `strFilter` (applies to numerator AND denominator, since an area-level
# domain should shrink both). Both filters here (large-diameter trees, mesic
# physiographic classes) use nationally-defined codes so the same filter is
# meaningful in every region.
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("biomass() matches EVALIDator for treeDomain (DIA >= 20) (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 10, sdenom = 2, wnum = "TREE.DIA >= 20")
    out <- as.data.frame(biomass(db_st, treeDomain = DIA >= 20))
    expect_equal(out$BIO_ACRE, ref$ratioEstimate, tolerance = 1e-6)
  })

  test_that(paste("biomass() matches EVALIDator for areaDomain (physiographic class filter) (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 10, sdenom = 2,
                     strFilter = "COND.PHYSCLCD in (21,22,23,24,25,26,27,28,29)")
    out <- as.data.frame(biomass(db_st, areaDomain = PHYSCLCD %in% 21:29)) # mesic classes
    expect_equal(out$BIO_ACRE, ref$ratioEstimate, tolerance = 1e-6)
  })
}

# Test 15 ------------------------------
# bySpecies grouping (RI): validates a couple of species rows produced by
# grpBy = SPCD against an independent single-species EVALIDator query, i.e.
# that a domain filter survives rFIA's internal grpBy/join path rather than
# being silently dropped for some groups (the historical area()/areaChange()
# bug pattern from v1.1.1). RI only, and only a random sample of species
# (rather than all of them), to keep this test's live API call volume small.
test_that("biomass() bySpecies matches EVALIDator per-species (RI)", {
  out <- as.data.frame(biomass(db_ri, bySpecies = TRUE))
  set.seed(42)
  sampled <- out[sample(nrow(out), 2), ]
  for (i in seq_len(nrow(sampled))) {
    ref <- fetchRef(wc = wc_ri, snum = 10, sdenom = 2,
                     wnum = paste0("TREE.SPCD = ", sampled$SPCD[i]))
    expect_equal(sampled$BIO_ACRE[i], ref$ratioEstimate, tolerance = 1e-6,
                 label = paste0("BIO_ACRE (SPCD ", sampled$SPCD[i], ")"))
  }
})
