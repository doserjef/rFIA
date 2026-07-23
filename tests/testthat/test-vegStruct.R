# Test vegStruct() --------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Most recent estimates for vegetation on forest land
out <- vegStruct(db = fiaRI_mr, landType = 'forest', totals = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 2 ------------------------------
# Most recent estimates by plot
out <- vegStruct(db = fiaRI_mr, land = 'forest', byPlot = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
# Most recent estimates grouped by stand age on forest land
# Make a categorical variable which represents stand age (grouped by 10 yr intervals)
fiaRI_mr$COND$STAND_AGE <- makeClasses(fiaRI_mr$COND$STDAGE, interval = 10)
out <- vegStruct(db = fiaRI_mr, grpBy = STAND_AGE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 4 ------------------------------
# Estimates on forested mesic sites
out <- vegStruct(db = fiaRI, landType = 'forest',
                 areaDomain = PHYSCLCD %in% 21:29)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

test_that("multiple years", {
  expect_gt(length(unique(out$YEAR)), 1)
})

# Test 5 ------------------------------
# Most recent estimates by county
out <- vegStruct(fiaRI_mr, polys = countiesRI, returnSpatial = TRUE)
test_that("out is of class sf", {
  expect_s3_class(out, "sf")
})

# Internal consistency checks (no EVALIDator, no network needed) ----------
# vegStruct() has no numeric ground truth available at all: EVALIDator's
# attribute library has no P2 vegetation structure attributes whatsoever
# (0 matches for "P2VEG", "vegetation structure", "growth habit"). All
# checks below are therefore either internal-consistency checks or
# cross-checks against nPlots_AREA from tpa() (already validated against
# EVALIDator; see tpa.md) for the same landType/areaDomain restriction --
# the same approach used for invasive() (see invasive.md), since both
# functions restrict to a P2 ancillary sampling protocol
# (P2VEG_SAMPLING_STATUS_CD here, INVASIVE_SAMPLING_STATUS_CD there) that
# may or may not further restrict the plot universe depending on the state.
# See core_references/validation/vegStruct.md for full methodology/results.
skip_on_cran()

validation_data_dir <- Sys.getenv("RFIA_VALIDATION_DATA", "~/Dropbox/data/fia")
skip_if_not(dir.exists(validation_data_dir),
            "Local FIADB validation data cache not found")

# One state per FIA region, read/clipped once and reused by every test below.
states <- c("RI", "NC", "CO", "OR")
dbs <- lapply(states, \(st) clipFIA(readFIA(validation_data_dir, states = st), mostRecent = TRUE))
names(dbs) <- states
db_ri <- dbs[["RI"]]

# Test 6 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(vegStruct(db_ri, polys = countiesRI, returnSpatial = TRUE))
  out_df <- as.data.frame(vegStruct(db_ri, polys = countiesRI, returnSpatial = FALSE))
  out_sf <- out_sf[, names(out_df)]
  out_sf <- out_sf[order(out_sf$COUNTY, out_sf$LAYER, out_sf$GROWTH_HABIT), ]
  out_df <- out_df[order(out_df$COUNTY, out_df$LAYER, out_df$GROWTH_HABIT), ]
  expect_equal(out_sf, out_df)
})

# Test 7 ------------------------------
# Internal consistency: totals divided by area reproduce the per-acre
# estimate (doesn't require EVALIDator).
for (st in states) {
  test_that(paste("vegStruct() totals are consistent with per-acre estimates (", st, ")"), {
    out <- as.data.frame(vegStruct(dbs[[st]], totals = TRUE))
    expect_equal(out$COVER_AREA_TOTAL / out$AREA_TOTAL * 100, out$COVER_PCT, tolerance = 1e-9)
  })
}

# Test 8 ------------------------------
# nPlots_AREA cross-check against tpa() (already validated against
# EVALIDator; see tpa.md) for the same landType/areaDomain restriction --
# the best available numeric ground truth given EVALIDator has no P2
# vegetation structure attributes at all. Exact in CO/OR (P2Veg sampling
# doesn't restrict the plot universe further there); NC's P2Veg sample is a
# much smaller, genuinely restricted subset of its forest plots (210 of
# 3561), so NC is checked only for internal monotonicity (a landType/
# areaDomain restriction should never increase the plot count) rather than
# an exact match. This doubles as the regression check for the
# nPlots_AREA phantom-row fix (vegStructStarter.R was missing the same
# `!is.na(CONDID)` guard already fixed in
# tpa()/biomass()/carbon()/volume()/dwm()/invasive()/seedling()/
# standStruct()/diversity()).
for (st in c("CO", "OR")) {
  db_st <- dbs[[st]]
  test_that(paste("vegStruct() nPlots_AREA matches tpa() exactly (", st, ")"), {
    for (lt in c('forest', 'timber')) {
      vs <- as.data.frame(vegStruct(db_st, landType = lt))
      ref <- as.data.frame(tpa(db_st, landType = lt, treeType = 'live'))
      expect_equal(unique(vs$nPlots_AREA), ref$nPlots_AREA,
                   label = paste("landType =", lt))
    }
    vs_ad <- as.data.frame(vegStruct(db_st, areaDomain = PHYSCLCD %in% 21:29))
    ref_ad <- as.data.frame(tpa(db_st, areaDomain = PHYSCLCD %in% 21:29, treeType = 'live'))
    expect_equal(unique(vs_ad$nPlots_AREA), ref_ad$nPlots_AREA, label = "areaDomain")
  })
}

for (st in c("RI", "NC")) {
  db_st <- dbs[[st]]
  test_that(paste("vegStruct() plot counts do not increase under landType/areaDomain restrictions (", st, ")"), {
    outForest <- as.data.frame(vegStruct(db_st, landType = 'forest'))
    outTimber <- as.data.frame(vegStruct(db_st, landType = 'timber'))
    outAD <- as.data.frame(vegStruct(db_st, areaDomain = PHYSCLCD %in% 21:29))
    expect_true(unique(outTimber$nPlots_AREA) <= unique(outForest$nPlots_AREA))
    expect_true(unique(outAD$nPlots_AREA) <= unique(outForest$nPlots_AREA))
  })
}

# Test 9 ------------------------------
# grpBy interaction (OWNGRPCD, CO -- a state where P2Veg sampling doesn't
# restrict the plot universe, so an exact match against tpa() is
# meaningful): each group's AREA_TOTAL should match tpa()'s grouped
# AREA_TOTAL exactly, validating that the grpBy join doesn't silently drop
# or misattribute area for some groups (the historical
# area()/areaChange() bug pattern from v1.1.1).
test_that("vegStruct() grpBy = OWNGRPCD matches tpa() per group (CO)", {
  db_co <- dbs[["CO"]]
  vs <- as.data.frame(vegStruct(db_co, grpBy = OWNGRPCD, totals = TRUE))
  ref <- as.data.frame(tpa(db_co, grpBy = OWNGRPCD, treeType = 'live', totals = TRUE))
  vs <- unique(vs[, c("OWNGRPCD", "AREA_TOTAL")])
  vs <- vs[order(vs$OWNGRPCD), ]
  ref <- ref[order(ref$OWNGRPCD), ]
  expect_equal(vs$AREA_TOTAL, ref$AREA_TOTAL)
})

# Test 10 ------------------------------
# Regression test for the byPlot cover-formula bug (see vegStruct.md,
# "Fixed" #2, the same class of bug already fixed in invasive()'s byPlot
# branch): a LAYER/GROWTH_HABIT combination not recorded on every subplot
# (the normal case for patchy vegetation) previously had its cover averaged
# over only the subplots where it *was* recorded (via
# mean(cover, na.rm = TRUE)), inflating PROP_COVER by up to 4x. NC plot
# 471569784489998 has Forbs (0-2ft layer) recorded on SUBP 3 (part of
# CONDID 2, COVER_PCT 5, SUBPCOND_PROP 0.922438) and SUBP 4 (CONDID 3,
# COVER_PCT 10, SUBPCOND_PROP 1.0) -- by hand, dividing by a fixed 4
# subplots: (0.05*0.922438 + 0.10*1.0) / 4 = 0.03653047.
test_that("vegStruct() byPlot PROP_COVER matches a hand calculation from raw data (NC)", {
  db_nc <- dbs[["NC"]]
  bp <- as.data.frame(vegStruct(db_nc, byPlot = TRUE))
  row <- bp[bp$PLT_CN == "471569784489998" & bp$LAYER == "0 to 2.0 feet" &
              bp$GROWTH_HABIT == "Forbs", ]
  expect_equal(nrow(row), 1)
  expect_equal(row$PROP_COVER, 0.03653047, tolerance = 1e-6)
})

# Test 11 ------------------------------
# Regression test for the incomplete GROWTH_HABIT_CD domain mapping (see
# vegStruct.md, "Fixed" #3): the region-specific codes DS (Interior West,
# RSCD 22: dead pinyon-species shrubs) and SS (PNWRS Fire Effects and
# Recovery Study plots: newly sprouted shrub cover after fire) were
# previously unmapped, so GROWTH_HABIT became NA and the entire record was
# silently dropped (GROWTH_HABIT is part of grpBy, and the final output
# step drops NA groups). CO has real DS-coded records; OR has real
# SS-coded records.
test_that("vegStruct() no longer drops region-specific GROWTH_HABIT_CD codes (CO, OR)", {
  db_co <- dbs[["CO"]]
  rawDS <- db_co$P2VEG_SUBP_STRUCTURE$GROWTH_HABIT_CD
  expect_true("DS" %in% rawDS)
  out_co <- as.data.frame(vegStruct(db_co))
  expect_true("Dead pinyon species shrubs" %in% out_co$GROWTH_HABIT)

  db_or <- dbs[["OR"]]
  rawSS <- db_or$P2VEG_SUBP_STRUCTURE$GROWTH_HABIT_CD
  expect_true("SS" %in% rawSS)
  out_or <- as.data.frame(vegStruct(db_or))
  expect_true("Newly sprouted shrub cover" %in% out_or$GROWTH_HABIT)
})

# Test 12 ------------------------------
# An areaDomain matching no conditions should return a clean 0-row result,
# not error or emit an internal max()-on-empty-vector warning (regression
# test for a bug found this pass: vegStructStarter.R's population-
# estimation tree list was missing a `!is.na(CONDID)` filter that its
# condition list `a` already has, letting a phantom "no condition" row
# survive with a real-looking cover = 0 -- see vegStruct.md, "Fixed" #1).
test_that("vegStruct() handles an empty areaDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(vegStruct(db_ri, areaDomain = STATECD == 999))
  )
  expect_equal(nrow(out), 0)
})
