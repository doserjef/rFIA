# Test invasive() -----------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Most recent estimates for invasives on forest land
out <- invasive(db = fiaRI_mr, landType = 'forest', totals = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 2 ------------------------------
# Most recent estimates by plot
out <- invasive(db = fiaRI_mr, land = 'forest', byPlot = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
# Most recent estimates grouped by stand age on forest land
# Make a categorical variable which represents stand age (grouped by 10 yr intervals)
fiaRI_mr$COND$STAND_AGE <- makeClasses(fiaRI_mr$COND$STDAGE, interval = 10)
out <- invasive(db = fiaRI_mr, grpBy = STAND_AGE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 4 ------------------------------ 
# Estimates on forested mesic sites
out <- invasive(db = fiaRI, landType = 'forest', 
                areaDomain = PHYSCLCD %in% 21:29)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

test_that("multiple years", {
  expect_gt(length(unique(out$YEAR)), 1)
})

# Test 5 ------------------------------
# Most recent estimates by county
out <- invasive(fiaRI_mr, polys = countiesRI, returnSpatial = TRUE)
test_that("out is of class sf", {
  expect_s3_class(out, "sf")
})

# Internal consistency checks (no EVALIDator, no network needed) ----------
# invasive() has no numeric ground truth available at all: EVALIDator's
# attribute library (EVALIDATOR_POP_ESTIMATE.csv) has no invasive-species
# attributes whatsoever (confirmed: 0 matches for "invasive", "P2VEG",
# "understory", "noxious", "nonnative", etc., and its EVAL_TYP list has no
# invasive-species-related evaluation type). All checks below are therefore
# either internal-consistency checks or cross-checks against nPlots_AREA
# from tpa() (already numerically validated against EVALIDator in
# tpa.md) for the same landType/areaDomain restriction -- valid because
# nPlots_AREA is fundamentally a property of the area denominator, not the
# specific numerator being estimated, and (per Test 8 below) turns out to be
# identical between the two functions in every state checked except Rhode
# Island. See core_references/validation/invasive.md for full
# methodology/results.
skip_on_cran()

validation_data_dir <- Sys.getenv("RFIA_VALIDATION_DATA", "~/Dropbox/data/fia")
skip_if_not(dir.exists(validation_data_dir),
            "Local FIADB validation data cache not found")

# One state per FIA region, read/clipped once and reused by every test below.
# OR (the Pacific Northwest state used by every prior validation pass) has no
# invasive-species sampling data at all in this local cache (its
# INVASIVE_SUBPLOT_SPP extract is header-only), nor does WA -- ID is
# substituted here as the nearest state with real invasive-species data.
states <- c("RI", "NC", "CO", "ID")
dbs <- lapply(states, \(st) clipFIA(readFIA(validation_data_dir, states = st), mostRecent = TRUE))
names(dbs) <- states
db_ri <- dbs[["RI"]]

# Test 6 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(invasive(db_ri, polys = countiesRI, returnSpatial = TRUE))
  out_df <- as.data.frame(invasive(db_ri, polys = countiesRI, returnSpatial = FALSE))
  out_sf <- out_sf[, names(out_df)]
  out_sf <- out_sf[order(out_sf$COUNTY, out_sf$SYMBOL), ]
  out_df <- out_df[order(out_df$COUNTY, out_df$SYMBOL), ]
  expect_equal(out_sf, out_df)
})

# Test 7 ------------------------------
# Internal consistency: totals divided by area reproduce the per-acre
# estimate (doesn't require EVALIDator).
for (st in states) {
  test_that(paste("invasive() totals are consistent with per-acre estimates (", st, ")"), {
    out <- as.data.frame(invasive(dbs[[st]], totals = TRUE))
    expect_equal(out$INV_AREA_TOTAL / out$AREA_TOTAL * 100, out$COVER_PCT, tolerance = 1e-9)
  })
}

# Test 8 ------------------------------
# nPlots_AREA cross-check against tpa() (already validated against
# EVALIDator; see tpa.md) for the same landType/areaDomain restriction --
# the best available numeric ground truth given EVALIDator has no
# invasive-species attributes at all. Exact in NC/CO/ID (every forest plot
# in these states was also sampled for invasives); RI's invasive-species
# sample is a much smaller subset of its forest plots (6 of 132) since
# INVASIVE_SAMPLING_STATUS_CD restricts the universe further there, so RI is
# checked only for internal monotonicity (areaDomain/timber restrictions
# should never increase the plot count) rather than an exact match. This
# doubles as the regression check for the nPlots_AREA phantom-row fix
# (invasiveStarter.R was missing the same `!is.na(CONDID)` guard already
# fixed in tpa()/biomass()/carbon()/volume()/dwm()).
for (st in c("NC", "CO", "ID")) {
  db_st <- dbs[[st]]
  test_that(paste("invasive() nPlots_AREA matches tpa() exactly (", st, ")"), {
    for (lt in c('forest', 'timber')) {
      inv <- as.data.frame(invasive(db_st, landType = lt))
      ref <- as.data.frame(tpa(db_st, landType = lt, treeType = 'live'))
      expect_equal(unique(inv$nPlots_AREA), ref$nPlots_AREA,
                   label = paste("landType =", lt))
    }
    inv_ad <- as.data.frame(invasive(db_st, areaDomain = PHYSCLCD %in% 21:29))
    ref_ad <- as.data.frame(tpa(db_st, areaDomain = PHYSCLCD %in% 21:29, treeType = 'live'))
    expect_equal(unique(inv_ad$nPlots_AREA), ref_ad$nPlots_AREA, label = "areaDomain")
  })
}

test_that("invasive() plot counts do not increase under landType/areaDomain restrictions (RI)", {
  outForest <- as.data.frame(invasive(db_ri, landType = 'forest'))
  outTimber <- as.data.frame(invasive(db_ri, landType = 'timber'))
  outAD <- as.data.frame(invasive(db_ri, areaDomain = PHYSCLCD %in% 21:29))
  expect_true(unique(outTimber$nPlots_AREA) <= unique(outForest$nPlots_AREA))
  expect_true(unique(outAD$nPlots_AREA) <= unique(outForest$nPlots_AREA))
})

# Test 9 ------------------------------
# An areaDomain matching no conditions should return a clean 0-row result,
# not error or emit an internal max()-on-empty-vector warning (regression
# test for a bug found this pass: invasiveStarter.R's population-estimation
# tree list was missing a `!is.na(SYMBOL)` filter that its byPlot branch
# already had, letting a phantom "no species detected" row survive when
# every species group was empty -- see invasive.md, "Fixed" #3).
test_that("invasive() handles an empty areaDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(invasive(db_ri, areaDomain = STATECD == 999))
  )
  expect_equal(nrow(out), 0)
})

# Test 10 ------------------------------
# Regression test for the REF_PLANT_DICTIONARY update (see invasive.md,
# "Fixed" #1): species identified only to genus (e.g. Ligustrum spp.,
# VEG_SPCD 'LIGUS2') were previously absent from the bundled dictionary,
# so invasive()'s final `drop_na(grpBy)` step (grpBy includes
# SCIENTIFIC_NAME/COMMON_NAME) silently discarded their entire row --
# including real, substantial COVER_PCT data (LIGUS2 alone affected 36% of
# North Carolina's raw invasive-species records). Confirms every genus-level
# code actually present in NC's raw data now survives to the final output
# with a real (positive) COVER_PCT.
test_that("invasive() no longer drops genus-level species codes (NC)", {
  db_nc <- dbs[["NC"]]
  rawCodes <- unique(db_nc$INVASIVE_SUBPLOT_SPP$VEG_SPCD)
  out <- as.data.frame(invasive(db_nc, landType = 'forest'))
  genusCodes <- c('ROSA5', 'LIGUS2', 'VINCA', 'HEDER', 'ELAEA', 'LONIC', 'LESPE', 'WISTE')
  genusCodes <- genusCodes[genusCodes %in% rawCodes]
  expect_true(length(genusCodes) > 0)
  expect_true(all(genusCodes %in% out$SYMBOL))
  expect_true(all(out$COVER_PCT[out$SYMBOL %in% genusCodes] > 0))
})

# Test 11 ------------------------------
# byPlot = TRUE cover values, spot-checked directly against raw
# INVASIVE_SUBPLOT_SPP/SUBP_COND/COND data (the best available ground truth,
# given no EVALIDator equivalent exists) -- regression test for the byPlot
# formula bug found this pass (see invasive.md, "Fixed" #2): the previous
# `mean(cover, na.rm = TRUE)` treated a subplot where the species wasn't
# recorded as a missing observation to exclude from the average, rather than
# a 0-cover observation to include, inflating cover by up to 16x for a
# species detected on fewer than all 4 subplots (the normal case for patchy
# invasive species). PROP_INV_COVER is deliberately left unadjusted by
# CONDPROP_UNADJ (that's reported separately as PROP_FOREST, the same split
# biomassStarter()'s byPlot branch uses for BIO_ACRE/PROP_FOREST), so the
# hand calculation is just cover * SUBPCOND_PROP / 4, with PROP_FOREST
# checked separately. Rosa multiflora (SYMBOL 'ROMU') on a specific RI plot
# is recorded on only 1 of 4 subplots (20% cover, subplot fully within a
# condition that is only 25% of the plot's area, i.e. PROP_FOREST = 0.25):
# by hand, 0.20 * 1.0 / 4 = 0.05.
test_that("invasive() byPlot cover matches a hand calculation from raw data (RI)", {
  out <- as.data.frame(invasive(db_ri, byPlot = TRUE))
  row <- out[!is.na(out$SYMBOL) & out$SYMBOL == 'ROMU' &
               !is.na(out$PROP_INV_COVER) & out$PROP_INV_COVER > 0, ]
  expect_equal(sort(row$PROP_INV_COVER), sort(c(0.0025, 0.0025, 0.05)),
               tolerance = 1e-6)
  expect_equal(row$PROP_FOREST[row$PROP_INV_COVER == 0.05], 0.25, tolerance = 1e-6)
})

