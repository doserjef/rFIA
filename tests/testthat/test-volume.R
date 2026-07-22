# Test volume() -----------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Most recent estimates for growing stock on timber land by species
out <- volume(db = fiaRI_mr, land = 'timber', bySpecies = TRUE, totals = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 2 ------------------------------
# Most recent estimates for growing stock on timber land by species by plot
out <- volume(db = fiaRI_mr, land = 'timber', bySpecies = TRUE, byPlot = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
# Estimates for live white pine (> 12" DBH) on forested mesic sites
out <- volume(fiaRI_mr,
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
out <- volume(db = fiaRI_mr, grpBy = STAND_AGE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 5 ------------------------------ 
# Estimates for snags greater than 20 in DBH on forestland for all
#  available inventories (time-series)
out <- volume(db = fiaRI, landType = 'forest', treeType = 'dead',
              treeDomain = DIA > 20)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

test_that("multiple years", {
  expect_gt(length(unique(out$YEAR)), 1)
})

# Test 6 ------------------------------
# Most recent estimates for all stems on forest land by species
out <- volume(db = fiaRI_mr, landType = 'forest', treeType = 'all',
              bySpecies = TRUE)
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 7 ------------------------------
# Most recent estimates for all stems on forest land
# grouped by user-defined areal units
out <- volume(fiaRI_mr,
              polys = countiesRI,
              returnSpatial = TRUE)
plot.out <- plotFIA(out, BOLE_CF_ACRE) # Plot of bole volume as color scale
test_that("out is of class sf", {
  expect_s3_class(out, "sf")
})
test_that('plot.out is a ggplot', {
  expect_s3_class(plot.out, 'gg')
})

# Internal consistency checks (no EVALIDator, no network needed) ----------
# These only require the local FIADB extract cache, not network access, so
# they still run when apps.fs.usda.gov is unreachable. See
# core_references/validation/volume.md for full methodology/results.
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

# Test 8 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(volume(db_ri, polys = countiesRI, returnSpatial = TRUE))
  out_df <- as.data.frame(volume(db_ri, polys = countiesRI, returnSpatial = FALSE))
  out_sf <- out_sf[, names(out_df)]
  out_sf <- out_sf[order(out_sf$COUNTY), ]
  out_df <- out_df[order(out_df$COUNTY), ]
  expect_equal(out_sf, out_df)
})

# Test 9 ------------------------------
# Internal consistency: totals divided by area reproduce the per-acre
# estimate (doesn't require EVALIDator).
for (st in states) {
  test_that(paste("volume() totals are consistent with per-acre estimates (", st, ")"), {
    out <- as.data.frame(volume(dbs[[st]], totals = TRUE))
    expect_equal(out$BOLE_CF_TOTAL / out$AREA_TOTAL, out$BOLE_CF_ACRE, tolerance = 1e-9)
    expect_equal(out$SAW_CF_TOTAL / out$AREA_TOTAL, out$SAW_CF_ACRE, tolerance = 1e-9)
    expect_equal(out$SAW_MBF_TOTAL / out$AREA_TOTAL, out$SAW_MBF_ACRE, tolerance = 1e-9)
  })
}

# Test 10 ------------------------------
# A treeDomain matching no trees should return a clean 0-row result, not
# error or emit an internal max()-on-empty-vector warning.
test_that("volume() handles an empty treeDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(volume(db_ri, treeDomain = SPCD == 999))
  )
  expect_equal(nrow(out), 0)
})

# Numeric validation against EVALIDator ------------------------------------
# Reference values are fetched live from the FIADB-API `fullreport` endpoint
# (see core_references/validation/fetch_evalidator.R and
# core_references/validation/volume.md for methodology and full results)
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
# Core default case (treeType = 'live', landType = 'forest', volType =
# 'NET') matches EVALIDator to full double precision across one state per
# FIA region: RI (Northern), NC (Southern), CO (Interior West), OR (Pacific
# Northwest). EVALIDator attribute 574171 = net merchantable bole cubic-foot
# volume of live trees (timber species >= 5in dbh) on forest land, ratio'd
# against attribute 2 (forest land area); attribute 16 = net sawlog
# cubic-foot volume of sawtimber trees on forest land. Attribute 20 = net
# sawlog board-foot volume (International 1/4-inch rule) on forest land --
# rFIA's SAW_MBF_ACRE is expressed in *thousand* board feet (hence the
# `/1000` in volumeStarter.R), so it's multiplied back up by 1000 here to
# compare against EVALIDator's raw board-foot attribute. Empirically
# confirmed (by comparing against the Scribner/Doyle board-foot attributes
# too) that VOLBFNET is the International 1/4-inch rule -- FIADB's
# standard/default board-foot volume equation.
for (st in states) {
  test_that(paste("volume() matches EVALIDator for", st, "(core default case)"), {
    wc_st <- wcs[[st]]
    boleRef <- fetchRef(wc = wc_st, snum = 574171, sdenom = 2)
    sawCfRef <- fetchRef(wc = wc_st, snum = 16, sdenom = 2)
    sawMbfRef <- fetchRef(wc = wc_st, snum = 20, sdenom = 2)

    out_st <- as.data.frame(volume(dbs[[st]], treeType = 'live', landType = 'forest'))

    expect_equal(out_st$BOLE_CF_ACRE, boleRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(out_st$SAW_CF_ACRE, sawCfRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(out_st$SAW_MBF_ACRE * 1000, sawMbfRef$ratioEstimate, tolerance = 1e-6)
    expect_equal(out_st$BOLE_CF_ACRE_SE, boleRef$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out_st$nPlots_TREE, boleRef$numPlotCount)
    expect_equal(out_st$nPlots_AREA, boleRef$denPlotCount)
  })
}

# Test 12 ------------------------------
# landType/treeType variants across all four FIA regions, matched against
# EVALIDator attributes 574172 (timberland/live bole cf), 15/18 (forest/
# timberland growing-stock bole cf -- restricted to TREECLCD = 2, i.e.
# excludes cull/rough trees that 'live' still includes), and 11252/11253
# (forest/timberland standing dead bole cf). Each ratio'd against the
# matching area attribute (3 = timberland, 2 = forest land). landType =
# 'timber' also doubles as the primary regression check for the nPlots_AREA
# phantom-row fix (see volume.md, "Fixed" #1): volumeStarter.R was missing
# the `!is.na(CONDID)` guard that tpa()/biomass()/carbon() already have.
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("volume() matches EVALIDator for landType = 'timber' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 574172, sdenom = 3)
    out <- as.data.frame(volume(db_st, treeType = 'live', landType = 'timber'))
    expect_equal(out$BOLE_CF_ACRE, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$BOLE_CF_ACRE_SE, ref$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out$nPlots_TREE, ref$numPlotCount)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })

  test_that(paste("volume() matches EVALIDator for treeType = 'gs' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 15, sdenom = 2)
    out <- as.data.frame(volume(db_st, treeType = 'gs', landType = 'forest'))
    expect_equal(out$BOLE_CF_ACRE, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$BOLE_CF_ACRE_SE, ref$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out$nPlots_TREE, ref$numPlotCount)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })

  test_that(paste("volume() matches EVALIDator for treeType = 'dead' (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 11252, sdenom = 2)
    out <- as.data.frame(volume(db_st, treeType = 'dead', landType = 'forest'))
    expect_equal(out$BOLE_CF_ACRE, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$BOLE_CF_ACRE_SE, ref$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out$nPlots_TREE, ref$numPlotCount)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })
}

# Test 13 ------------------------------
# Domain filter interactions across all four FIA regions. treeDomain matched
# against EVALIDator's `wnum` (numerator-only filter, since a tree-level
# domain should not change the area denominator, so only nPlots_TREE is
# checked); areaDomain matched against `strFilter` (applies to numerator AND
# denominator, so both plot counts are checked) -- this is the second
# regression check for the nPlots_AREA phantom-row fix, exercising it with a
# filter that excludes some, but not all, plots (landType = 'timber' in
# Test 12 exercises a different restriction of the same fix). Both filters
# here (large-diameter trees, mesic physiographic classes) use
# nationally-defined codes so the same filter is meaningful in every region.
for (st in states) {
  db_st <- dbs[[st]]
  wc_st <- wcs[[st]]

  test_that(paste("volume() matches EVALIDator for treeDomain (DIA >= 20) (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 574171, sdenom = 2, wnum = "TREE.DIA >= 20")
    out <- as.data.frame(volume(db_st, treeType = 'live', landType = 'forest',
                                treeDomain = DIA >= 20))
    expect_equal(out$BOLE_CF_ACRE, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$BOLE_CF_ACRE_SE, ref$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out$nPlots_TREE, ref$numPlotCount)
  })

  test_that(paste("volume() matches EVALIDator for areaDomain (physiographic class filter) (", st, ")"), {
    ref <- fetchRef(wc = wc_st, snum = 574171, sdenom = 2,
                     strFilter = "COND.PHYSCLCD in (21,22,23,24,25,26,27,28,29)")
    out <- as.data.frame(volume(db_st, treeType = 'live', landType = 'forest',
                                areaDomain = PHYSCLCD %in% 21:29)) # mesic classes
    expect_equal(out$BOLE_CF_ACRE, ref$ratioEstimate, tolerance = 1e-6)
    expect_equal(out$BOLE_CF_ACRE_SE, ref$ratioSEPercent, tolerance = 1e-6)
    expect_equal(out$nPlots_TREE, ref$numPlotCount)
    expect_equal(out$nPlots_AREA, ref$denPlotCount)
  })
}

# Test 14 ------------------------------
# bySpecies grouping (RI): validates a couple of species rows produced by
# grpBy = SPCD against an independent single-species EVALIDator query, i.e.
# that a domain filter survives rFIA's internal grpBy/join path rather than
# being silently dropped for some groups (the historical area()/areaChange()
# bug pattern from v1.1.1). RI only, and only a random sample of species
# (rather than all of them), to keep this test's live API call volume small.
test_that("volume() bySpecies matches EVALIDator per-species (RI)", {
  out <- as.data.frame(volume(db_ri, treeType = 'live', landType = 'forest',
                              bySpecies = TRUE))
  set.seed(42)
  sampled <- out[sample(nrow(out), 2), ]
  for (i in seq_len(nrow(sampled))) {
    ref <- fetchRef(wc = wc_ri, snum = 574171, sdenom = 2,
                     wnum = paste0("TREE.SPCD = ", sampled$SPCD[i]))
    expect_equal(sampled$BOLE_CF_ACRE[i], ref$ratioEstimate, tolerance = 1e-6,
                 label = paste0("BOLE_CF_ACRE (SPCD ", sampled$SPCD[i], ")"))
    expect_equal(sampled$nPlots_TREE[i], ref$numPlotCount,
                 label = paste0("nPlots_TREE (SPCD ", sampled$SPCD[i], ")"))
  }
})

