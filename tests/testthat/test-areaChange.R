# Test tpa() --------------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Most recent estimates for timberland
out <- areaChange(db = fiaRI_mr, landType = 'timber')

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 2 ------------------------------
# Most recent estimates for forest land by plot
out <- areaChange(db = fiaRI_mr, landType = 'forest', byPlot = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
# Estimates for live white pine (> 12" DBH)
out <- areaChange(fiaRI_mr,
           treeDomain = SPCD == 129 & DIA > 22) # Species code for white pine

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 4 ------------------------------
# Most recent estimates grouped by stand age on forest land
# Make a categorical variable which represents stand age (grouped by 10 yr intervals)
fiaRI_mr$COND$STAND_AGE <- makeClasses(fiaRI_mr$COND$STDAGE, interval = 10)
out <- areaChange(db = fiaRI_mr, grpBy = STAND_AGE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 5 ------------------------------ 
# Estimates for areaChange with trees greater than 20 in DBH
out <- areaChange(db = fiaRI, landType = 'forest', treeDomain = DIA > 20)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

test_that("multiple years", {
  expect_gt(length(unique(out$YEAR)), 1)
})

# Test 7 ------------------------------
# Most recent estimates for all stems on forest land 
# grouped by user-defined areaChangel units
out <- areaChange(fiaRI_mr,
           polys = countiesRI,
           returnSpatial = TRUE, method = 'EMA')
plot.out <- plotFIA(out, AREA_CHNG) # Plot of TPA with color scale
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
# EVAL_GRP encodes STATECD + 4-digit year (e.g. 442024 = Rhode Island 2024);
# reading it off each clipped db mirrors exactly which evaluation
# `mostRecent` actually selected, so it never needs to be hard-coded.
wcs <- lapply(dbs, \(d) unique(d$POP_EVAL_GRP$EVAL_GRP))
wc_ri <- wcs[["RI"]]

# Test 8 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(areaChange(db_ri, polys = countiesRI, landType = 'forest',
                                     returnSpatial = TRUE))
  out_df <- as.data.frame(areaChange(db_ri, polys = countiesRI, landType = 'forest',
                                     returnSpatial = FALSE))
  common <- intersect(names(out_sf), names(out_df))
  out_sf <- out_sf[order(out_sf$polyID), common]
  out_df <- out_df[order(out_df$polyID), common]
  expect_equal(out_sf, out_df)
})

# Test 9 ------------------------------
# chngType = 'net' is defined as the net result of diversion and reversion
# processes (see man/areaChange.Rd, "Estimation Details"): net AREA_CHNG must
# equal (reversion AREA_CHNG - diversion AREA_CHNG) from the chngType =
# 'component' breakdown, for both landType = 'forest' and 'timber', across
# all four FIA regions.
for (st in states) {
  for (lt in c('forest', 'timber')) {
    label1 <- if (lt == 'forest') 'Forest' else 'Timber'
    label2 <- if (lt == 'forest') 'Non-forest' else 'Non-timber'

    test_that(paste("areaChange() net AREA_CHNG equals reversion - diversion (",
                     st, ",", lt, ")"), {
      db_st <- dbs[[st]]
      net <- as.data.frame(areaChange(db_st, landType = lt, chngType = 'net'))
      comp <- as.data.frame(areaChange(db_st, landType = lt, chngType = 'component'))

      diversion <- comp$AREA_CHNG[comp$STATUS1 == label1 & comp$STATUS2 == label2]
      reversion <- comp$AREA_CHNG[comp$STATUS1 == label2 & comp$STATUS2 == label1]

      expect_equal(net$AREA_CHNG, reversion - diversion, tolerance = 1e-6)
    })
  }
}

# Test 10 ------------------------------
# A treeDomain matching no trees should return a clean 0-row result, not
# error or emit an internal max()-on-empty-vector warning (combineMR() is
# shared with tpa()/area(); see tpa.md/area.md, "Fixed").
test_that("areaChange() handles an empty treeDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(areaChange(db_ri, landType = 'forest', treeDomain = SPCD == 999))
  )
  expect_equal(nrow(out), 0)
})

# Numeric validation against EVALIDator ------------------------------------
# Reference values are fetched live from the FIADB-API `fullreport` endpoint
# (see core_references/validation/fetch_evalidator.R and
# core_references/validation/areaChange.md for methodology and full results)
# rather than hard-coded, so these tests can never drift from what
# EVALIDator currently reports. They require network access to
# apps.fs.usda.gov (on top of the local data cache already required above),
# so they're skipped (not failed) when it's unavailable.
skip_if_not_installed("curl")
skip_if_not_installed("jsonlite")
source(test_path("..", "..", "core_references", "validation", "fetch_evalidator.R"))

network_ok <- tryCatch({
  fetch_evalidator(wc = wc_ri, snum = 127)
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
# EVALIDator's EXPCHNG-tagged attributes for area change (126-139) are not
# signed net-change deltas -- they are base-population area totals computed
# from SUBP_COND_CHNG_MTRX proportions, categorized by whether *both* or
# *either* measurement was forest/timberland (see areaChange.md,
# "Methodological note"). Attribute 127/129 = area of conditions that were
# forest/timberland at BOTH measurements -- this matches
# areaChange(chngType = 'component')'s "STATUS1 == STATUS2" (no-change) row
# exactly. Attribute 128/130 = area of conditions that were forest/timberland
# at EITHER measurement -- this matches the sum of PREV_AREA across all three
# component categories (no-change + diversion + reversion), which is exactly
# the population that this bug (nonsampled conditions misclassified as a
# genuine land-use change) previously inflated -- see areaChange.md, "Fixed".
for (st in states) {
  wc_st <- wcs[[st]]

  for (spec in list(list(lt = 'forest', label = 'Forest', snum_both = 127, snum_either = 128),
                    list(lt = 'timber', label = 'Timber', snum_both = 129, snum_either = 130))) {

    test_that(paste("areaChange() matches EVALIDator for landType = '", spec$lt,
                     "', 'both' population (", st, ")"), {
      ref <- fetchRef(wc = wc_st, snum = spec$snum_both)
      out <- as.data.frame(areaChange(dbs[[st]], landType = spec$lt, chngType = 'component'))
      stable <- out[out$STATUS1 == spec$label & out$STATUS2 == spec$label, ]
      expect_equal(stable$PREV_AREA, ref$estimate, tolerance = 1e-6)
      expect_equal(stable$PREV_AREA_SE, ref$sePercent, tolerance = 1e-6)
      expect_equal(stable$nPlots_AREA, ref$plotCount)
    })

    test_that(paste("areaChange() matches EVALIDator for landType = '", spec$lt,
                     "', 'either' population (", st, ")"), {
      ref <- fetchRef(wc = wc_st, snum = spec$snum_either)
      out <- as.data.frame(areaChange(dbs[[st]], landType = spec$lt, chngType = 'component'))
      expect_equal(sum(out$PREV_AREA), ref$estimate, tolerance = 1e-6)
    })
  }
}
