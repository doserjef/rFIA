# Test standStruct() --------------------------------------------------------

skip_on_cran()

# Testing data
data(fiaRI)
data(countiesRI)
# Get most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
out <- standStruct(fiaRI, polys = countiesRI, returnSpatial = TRUE, totals = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})
test_that("out is of class sf", {
  expect_s3_class(out, "sf")
})

# Test 2 ------------------------------
out <- standStruct(db = fiaRI_mr, landType = 'forest')
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
out <- standStruct(db = fiaRI_mr, landType = 'timber',
                 byPlot = TRUE)
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 4 ------------------------------
# Most recent estimates grouped by stand age on forest land.
fiaRI_mr$COND$STAND_AGE <- makeClasses(fiaRI_mr$COND$STDAGE, interval = 10)
out <- standStruct(db = fiaRI_mr, grpBy = STAND_AGE)
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})
out <- standStruct(db = fiaRI_mr, grpBy = OWNGRPCD)
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})


# Test 5 ------------------------------
# Estimates on forested mesic sites
# (all available inventories)
out <- standStruct(fiaRI,
                 areaDomain = PHYSCLCD %in% 21:29) # Mesic Physiographic classes
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})
test_that("multiple years", {
  expect_gt(length(unique(out$YEAR)), 1)
})

# Test 7 ------------------------------
# Most recent estimates on forestland in user-defined polygons
out <- standStruct(fiaRI_mr, landType = 'forest', polys = countiesRI,
                 returnSpatial = TRUE)
plot.out <- plotFIA(out, COVER_PCT)
test_that("out is of class sf", {
  expect_s3_class(out, "sf")
})
test_that('plot.out is a ggplot', {
  expect_s3_class(plot.out, 'gg')
})

# Numeric validation ---------------------------------------------------------
# standStruct() has no EVALIDator ground truth at all: stand structural stage
# is an rFIA-specific classification (Frelich & Lorimer 1991-style, substituting
# basal area for exposed crown area), not a standard FIADB/EVALIDator
# population attribute -- EVALIDATOR_POP_ESTIMATE.csv has no matches for
# "structural stage", "stand structure", or similar. Validation here is
# therefore internal consistency, cross-checks against tpa()'s nPlots_AREA/
# AREA_TOTAL (already validated against EVALIDator; see tpa.md) for the same
# landType/areaDomain/grpBy restriction, and hand calculations replicating
# structHelper()'s own classification formula independently from raw data.
# See core_references/validation/standStruct.md for full methodology/results.
skip_on_cran()

validation_data_dir <- Sys.getenv("RFIA_VALIDATION_DATA", "~/Dropbox/data/fia")
skip_if_not(dir.exists(validation_data_dir),
            "Local FIADB validation data cache not found")

# One state per FIA region, read/clipped once and reused by every test below.
states <- c("RI", "NC", "CO", "OR")
dbs <- lapply(states, \(st) clipFIA(readFIA(validation_data_dir, states = st), mostRecent = TRUE))
names(dbs) <- states
db_ri <- dbs[["RI"]]

# Test 8 ------------------------------
# COVER_PCT should sum to exactly 100% across STAGE categories -- the
# category list (pole/mature/late/mosaic) is exhaustive by construction
# (structHelper() always returns one of the four). This is also a
# regression test for the CONDID-omitted-from-distinct() bug (see
# standStruct.md, "Fixed" #2): before the fix, a plot with 2+ zero-tree
# forest conditions had all but one silently dropped from every STAGE
# category, so their combined COVER_PCT fell short of 100%.
for (st in states) {
  test_that(paste("standStruct() COVER_PCT sums to 100 (", st, ")"), {
    out <- as.data.frame(standStruct(dbs[[st]]))
    expect_equal(sum(out$COVER_PCT), 100, tolerance = 1e-6)
  })
}

# Test 9 ------------------------------
# Internal consistency: totals divided by area reproduce the per-acre
# estimate (doesn't require EVALIDator).
for (st in states) {
  test_that(paste("standStruct() totals are consistent with per-acre estimates (", st, ")"), {
    out <- as.data.frame(standStruct(dbs[[st]], totals = TRUE))
    expect_equal(out$STAGE_AREA_TOTAL / out$AREA_TOTAL * 100, out$COVER_PCT, tolerance = 1e-9)
  })
}

# Test 10 ------------------------------
# nPlots_AREA and AREA_TOTAL cross-checked against tpa() (already validated
# against EVALIDator; see tpa.md) for the same landType/areaDomain/grpBy
# restriction -- the best available numeric ground truth given EVALIDator
# has no structural-stage attribute at all. Regression test for the
# nPlots_AREA phantom-row bug (standStruct.md, "Fixed" #1): before the fix,
# nPlots_AREA didn't reflect landType = 'timber'/areaDomain restrictions.
for (st in states) {
  db_st <- dbs[[st]]
  test_that(paste("standStruct() nPlots_AREA/AREA_TOTAL match tpa() exactly (", st, ")"), {
    for (lt in c('forest', 'timber')) {
      ss <- as.data.frame(standStruct(db_st, landType = lt, totals = TRUE))
      ref <- as.data.frame(tpa(db_st, landType = lt, treeType = 'live', totals = TRUE))
      expect_equal(unique(ss$nPlots_AREA), ref$nPlots_AREA, label = paste("landType =", lt))
      expect_equal(unique(ss$AREA_TOTAL), ref$AREA_TOTAL, label = paste("landType =", lt))
    }
    ss_ad <- as.data.frame(standStruct(db_st, areaDomain = PHYSCLCD %in% 21:29, totals = TRUE))
    ref_ad <- as.data.frame(tpa(db_st, areaDomain = PHYSCLCD %in% 21:29, treeType = 'live', totals = TRUE))
    expect_equal(unique(ss_ad$nPlots_AREA), ref_ad$nPlots_AREA, label = "areaDomain")
    expect_equal(unique(ss_ad$AREA_TOTAL), ref_ad$AREA_TOTAL, label = "areaDomain")
  })
}

# Test 11 ------------------------------
# grpBy (OWNGRPCD, NC): each group's COVER_PCT should still sum to 100%, and
# each group's AREA_TOTAL should match tpa()'s grouped AREA_TOTAL exactly --
# validates that a grpBy join doesn't silently drop or misattribute area for
# some groups (the historical area()/areaChange() bug pattern from v1.1.1).
test_that("standStruct() grpBy = OWNGRPCD matches tpa() per group (NC)", {
  db_nc <- dbs[["NC"]]
  ss <- as.data.frame(standStruct(db_nc, grpBy = OWNGRPCD, totals = TRUE))
  ref <- as.data.frame(tpa(db_nc, grpBy = OWNGRPCD, treeType = 'live', totals = TRUE))
  byGrp <- stats::aggregate(COVER_PCT ~ OWNGRPCD, data = ss, FUN = sum)
  expect_equal(byGrp$COVER_PCT, rep(100, nrow(byGrp)), tolerance = 1e-6)

  areaByGrp <- unique(ss[, c("OWNGRPCD", "AREA_TOTAL")])
  areaByGrp <- areaByGrp[order(areaByGrp$OWNGRPCD), ]
  ref <- ref[order(ref$OWNGRPCD), ]
  expect_equal(areaByGrp$AREA_TOTAL, ref$AREA_TOTAL)
})

# Test 12 ------------------------------
# Hand calculation of structHelper()'s own basal-area-proportion formula
# from raw TREE/COND data, independent of the package code, for a specific
# plot (RI, pltID "1_44_3_233"): 25 live trees >= 1" DBH on the plot's one
# forested condition (CONDID 2, CONDPROP_UNADJ 0.326472), with crown class
# in {2,3,4} and DIA >= 5" (the only trees that count toward basal area).
# By hand: pole-class (5" <= DIA < 10.23622") BA share 0.6739, mature-class
# (10.23622" <= DIA < 18.11024") BA share 0.3261, large-class share 0 --
# pole + mature > 0.67 and pole > mature, so STAGE = 'pole' (per
# standStruct.Rd's classification rules), matching COVER_PCT thresholds
# (12.7-25.9cm pole / 26-45.9cm mature / 46+cm large, i.e. exactly 5/
# 10.23622/18.11024 inches).
test_that("standStruct() byPlot STAGE matches a hand calculation from raw data (RI)", {
  bp <- as.data.frame(standStruct(db_ri, byPlot = TRUE))
  row <- bp[bp$pltID == "1_44_3_233", ]
  expect_equal(nrow(row), 1)
  expect_equal(as.character(row$STAGE), "POLE")
  expect_equal(row$PROP_STAGE, 0.326472, tolerance = 1e-6)
  expect_equal(row$PROP_FOREST, 0.326472, tolerance = 1e-6)
})

# Test 13 ------------------------------
# Regression test for the CONDID-omitted-from-distinct() undercount bug
# (see standStruct.md, "Fixed" #2): NC plot 1150116756290487 has two
# zero-tree forest conditions (CONDID 2 and 3, CONDPROP_UNADJ 0.25 each).
# Before the fix, distinct(PLT_CN, SUBP, TREE) collapsed both conditions'
# phantom "no tree" rows (SUBP = NA, TREE = NA for both) into one, silently
# dropping one condition's area from every STAGE category. PROP_STAGE
# should equal the *combined* 0.5 (matching PROP_FOREST), not just 0.25.
test_that("standStruct() byPlot correctly combines multiple zero-tree conditions on one plot (NC)", {
  db_nc <- dbs[["NC"]]
  bp <- as.data.frame(standStruct(db_nc, byPlot = TRUE))
  row <- bp[bp$PLT_CN == "1150116756290487", ]
  expect_equal(nrow(row), 1)
  expect_equal(as.character(row$STAGE), "MOSAIC")
  expect_equal(row$PROP_STAGE, 0.5, tolerance = 1e-6)
  expect_equal(row$PROP_FOREST, 0.5, tolerance = 1e-6)
})

# Test 14 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(standStruct(db_ri, polys = countiesRI, returnSpatial = TRUE))
  out_df <- as.data.frame(standStruct(db_ri, polys = countiesRI, returnSpatial = FALSE))
  out_sf <- out_sf[, names(out_df)]
  out_sf <- out_sf[order(out_sf$COUNTY, out_sf$STAGE), ]
  out_df <- out_df[order(out_df$COUNTY, out_df$STAGE), ]
  expect_equal(out_sf, out_df)
})

# Test 15 ------------------------------
# An areaDomain matching no conditions should return a clean 0-row result,
# not error or emit an internal max()-on-empty-vector warning. Regression
# test for a bug found this pass (standStruct.md, "Fixed" #3):
# standStructStarter.R's population-estimation STAGE computation was
# missing the `!is.na(CONDID)` filter its condition list `a` already has,
# so a phantom "no condition" row still got classified as 'mosaic' via
# structHelper()'s NaN fallback, surviving as a spurious result instead of
# a genuinely empty one.
test_that("standStruct() handles an empty areaDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(standStruct(db_ri, areaDomain = STATECD == 999))
  )
  expect_equal(nrow(out), 0)
})
