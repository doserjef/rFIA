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

