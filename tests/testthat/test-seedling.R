# Test seedling() --------------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Most recent estimates for growing stock on timber land by species
out <- seedling(db = fiaRI_mr, land = 'timber', bySpecies = TRUE, totals = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 2 ------------------------------
# Most recent estimates for growing stock on timber land by species by plot
out <- seedling(db = fiaRI_mr, land = 'timber', bySpecies = TRUE, byPlot = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
# Estimates for live white pine on forested mesic sites
out <- seedling(fiaRI_mr,
           treeDomain = SPCD == 129, # Species code for white pine
           areaDomain = PHYSCLCD %in% 21:29)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 4 ------------------------------
# Most recent estimates grouped by stand age on forest land
# Make a categorical variable which represents stand age (grouped by 10 yr intervals)
fiaRI_mr$COND$STAND_AGE <- makeClasses(fiaRI_mr$COND$STDAGE, interval = 10)
out <- seedling(db = fiaRI_mr, grpBy = STAND_AGE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 5 ------------------------------
# Estimates for seedlings on forest land for all available inventories (time-series)
out <- seedling(db = fiaRI, landType = 'forest')

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

test_that("multiple years", {
  expect_gt(length(unique(out$YEAR)), 1)
})

# Test 6 ------------------------------
# Most recent estimates on forest land by species
out <- seedling(db = fiaRI_mr, landType = 'forest', bySpecies = TRUE)
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 7 ------------------------------
# Most recent estimates for all stems on forest land
# grouped by user-defined areal units
out <- seedling(fiaRI_mr,
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
# (including the EVALIDator-comparison tests further down).
states <- c("RI", "NC", "CO", "OR")
dbs <- lapply(states, \(st) clipFIA(readFIA(validation_data_dir, states = st), mostRecent = TRUE))
names(dbs) <- states
db_ri <- dbs[["RI"]]
# EVAL_GRP encodes STATECD + 4-digit year; reading it off each clipped db
# mirrors exactly which evaluation `mostRecent = TRUE` selected (see tpa.md).
wcs <- lapply(dbs, \(d) unique(d$POP_EVAL_GRP$EVAL_GRP))
wc_ri <- wcs[["RI"]]

# Test 8 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(seedling(db_ri, polys = countiesRI, returnSpatial = TRUE))
  out_df <- as.data.frame(seedling(db_ri, polys = countiesRI, returnSpatial = FALSE))
  out_sf <- out_sf[, names(out_df)]
  out_sf <- out_sf[order(out_sf$COUNTY), ]
  out_df <- out_df[order(out_df$COUNTY), ]
  expect_equal(out_sf, out_df)
})

# Test 9 ------------------------------
# Internal consistency: totals divided by area reproduce the per-acre
# estimate (doesn't require EVALIDator).
for (st in states) {
  test_that(paste("seedling() totals are consistent with per-acre estimates (", st, ")"), {
    out <- as.data.frame(seedling(dbs[[st]], totals = TRUE))
    expect_equal(out$TREE_TOTAL / out$AREA_TOTAL, out$TPA, tolerance = 1e-9)
  })
}

# Test 10 ------------------------------
# A treeDomain matching no seedlings should return a clean 0-row result, not
# error or emit an internal max()-on-empty-vector warning (same combineMR()
# guard exercised by tpa.md, "Fixed" #2, and invasive.md, "Fixed" #3).
test_that("seedling() handles an empty treeDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(seedling(db_ri, treeDomain = SPCD == 999))
  )
  expect_equal(nrow(out), 0)
})

test_that("seedling() handles an empty areaDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(seedling(db_ri, areaDomain = STATECD == 999))
  )
  expect_equal(nrow(out), 0)
})

# Test 11 ------------------------------
# Regression test for the CONDID-omitted-from-distinct() undercount bug (see
# seedling.md, "Fixed" #3): a subplot that straddles two conditions can have
# separate SEEDLING rows for the same PLT_CN/SUBP/SPCD, one per CONDID.
# NC plot 1150115978290487 has red maple (SPCD 316) seedlings recorded on
# SUBP 3 under CONDID 1 (TPA_UNADJ 149.9306), SUBP 3 under CONDID 2
# (TPA_UNADJ 149.9306), and SUBP 4 under CONDID 2 (TPA_UNADJ 374.8264) --
# hand sum 674.6875. Before the fix, distinct(PLT_CN, SUBP, SPCD) collapsed
# the two CONDID 1/2 rows for SUBP 3 into one, silently dropping 149.9306.
test_that("seedling() byPlot TPA matches a hand calculation from raw data with split-condition subplots (NC)", {
  db_nc <- dbs[["NC"]]
  out <- as.data.frame(seedling(db_nc, byPlot = TRUE, bySpecies = TRUE))
  row <- out[!is.na(out$SPCD) & out$SPCD == 316 & out$pltID == "1_37_19_37", ]
  expect_equal(nrow(row), 1)
  expect_equal(row$TPA, 674.6875, tolerance = 1e-6)
})

# Numeric validation against EVALIDator ------------------------------------
# Reference values are fetched live from the FIADB-API `fullreport` endpoint
# (see core_references/validation/fetch_evalidator.R and
# core_references/validation/seedling.md for methodology and full results)
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
# attribute 45 = number of live seedlings on forest land, 46 = same on
# timberland, ratio'd against attribute 2 (forest land area) / 3
# (timberland area) respectively.
for (st in states) {
  test_that(paste("seedling() matches EVALIDator for", st, "(forest land)"), {
    wc_st <- wcs[[st]]
    ref <- fetchRef(wc = wc_st, snum = 45, sdenom = 2)
    out <- as.data.frame(seedling(dbs[[st]], landType = 'forest'))

    expect_equal(out$TPA, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$TPA_SE, ref$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out$nPlots_TREE, ref$numPlotCount)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })

  test_that(paste("seedling() matches EVALIDator for", st, "(timberland)"), {
    wc_st <- wcs[[st]]
    ref <- fetchRef(wc = wc_st, snum = 46, sdenom = 3)
    out <- as.data.frame(seedling(dbs[[st]], landType = 'timber'))

    expect_equal(out$TPA, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })
}

# Test 13 ------------------------------
# Domain filter interactions across all four FIA regions. treeDomain matched
# against EVALIDator's `wnum` (numerator-only filter, since a seedling-level
# domain should not change the area denominator); areaDomain matched against
# `strFilter` (applies to numerator AND denominator). See tpa.md for why
# `wnum` vs `strFilter` matters here -- the same distinction applies to
# seedling()'s treeDomain/areaDomain.
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("seedling() matches EVALIDator for areaDomain (physiographic class filter) (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 45, sdenom = 2,
                     strFilter = "COND.PHYSCLCD in (21,22,23,24,25,26,27,28,29)")
    out <- as.data.frame(seedling(db_st, areaDomain = PHYSCLCD %in% 21:29)) # mesic classes
    expect_equal(out$TPA, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$TPA_SE, ref$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out$nPlots_TREE, ref$numPlotCount)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })
}

# RI-specific species filter (white pine) and NC-specific species filter
# (loblolly pine, the dominant regenerating species in the Piedmont/Coastal
# Plain) -- kept as separate state-specific checks since neither species is
# a nationally meaningful filter for all four states.
test_that("seedling() matches EVALIDator for treeDomain (species filter, RI white pine)", {
  ref <- fetchRef(wc = wc_ri, snum = 45, sdenom = 2, wnum = "SEEDLING.SPCD = 129")
  out <- as.data.frame(seedling(db_ri, treeDomain = SPCD == 129))
  expect_equal(out$TPA, ref$ratioEstimate, tolerance = 1e-6)
  expect_equal(out$nPlots_TREE, ref$numPlotCount)
})

test_that("seedling() matches EVALIDator for treeDomain (species filter, NC loblolly pine)", {
  wc_nc <- wcs[["NC"]]
  ref <- fetchRef(wc = wc_nc, snum = 45, sdenom = 2, wnum = "SEEDLING.SPCD = 131")
  out <- as.data.frame(seedling(dbs[["NC"]], treeDomain = SPCD == 131))
  expect_equal(out$TPA, ref$ratioEstimate, tolerance = 1e-6)
  expect_equal(out$nPlots_TREE, ref$numPlotCount)
})

# Test 14 ------------------------------
# bySpecies grouping (RI): validates a random sample of individual species
# rows produced by grpBy = SPCD against an independent single-species
# EVALIDator query, i.e. that a domain filter survives rFIA's internal
# grpBy/join path rather than being silently dropped for some groups (the
# historical area()/areaChange() bug pattern from v1.1.1). See tpa.md for
# why EVALIDator's own row-grouping mechanism (`rselected`) can't be used
# directly via the `fullreport` endpoint.
test_that("seedling() bySpecies matches EVALIDator per-species (RI)", {
  out <- as.data.frame(seedling(db_ri, bySpecies = TRUE))
  set.seed(42)
  sampled <- out[sample(nrow(out), 3), ]
  for (i in seq_len(nrow(sampled))) {
    ref <- fetchRef(wc = wc_ri, snum = 45, sdenom = 2,
                     wnum = paste0("SEEDLING.SPCD = ", sampled$SPCD[i]))
    expect_equal(sampled$TPA[i], ref$ratioEstimate, tolerance = 1e-6,
                 label = paste0("TPA (SPCD ", sampled$SPCD[i], ")"))
    expect_equal(sampled$nPlots_TREE[i], ref$numPlotCount,
                 label = paste0("nPlots_TREE (SPCD ", sampled$SPCD[i], ")"))
  }
})
