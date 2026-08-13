# Test customPSE() ---------------------------------------------------------
# customPSE() has no EVALIDator equivalent (it estimates user-defined
# variables, not standard FIA attributes), so it's validated internally
# instead: feeding a tree-/condition-list produced by tpa(), area(), or
# volume() (via their treeList/condList = TRUE argument) back into
# customPSE() should exactly reproduce that same function's own
# population-level point estimates, SEs, and plot counts. See
# core_references/validation/customPSE.md for the full write-up, including a
# genuine nPlots_x/nPlots_y bug found and fixed this way.

skip_on_cran()

data(fiaRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
# Basic smoke test: tree-area ratio (TPA, BAA per acre of forest land),
# matching the example in ?customPSE.
tree.list <- tpa(fiaRI_mr, treeList = TRUE)
out <- customPSE(db = fiaRI_mr,
                  x = dplyr::select(tree.list, -c(AREA_BASIS)),
                  xVars = c(TPA, BAA),
                  y = dplyr::select(tree.list, -c(TREE_BASIS)),
                  yVars = PROP_FOREST)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 2 ------------------------------
# xGrpBy smoke test (species groups)
tree.list.sp <- tpa(fiaRI_mr, treeList = TRUE, bySpecies = TRUE)
out <- customPSE(db = fiaRI_mr,
                  x = dplyr::select(tree.list.sp, -c(AREA_BASIS)),
                  xVars = TPA,
                  xGrpBy = SPCD,
                  y = dplyr::select(tree.list.sp, -c(TREE_BASIS, SPCD, COMMON_NAME, SCIENTIFIC_NAME)),
                  yVars = PROP_FOREST)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Internal consistency checks (no EVALIDator, no network needed) ----------
# These only require the local FIADB extract cache, not network access.
skip_on_cran()

validation_data_dir <- Sys.getenv("RFIA_VALIDATION_DATA", "~/Dropbox/data/fia")
skip_if_not(dir.exists(validation_data_dir),
            "Local FIADB validation data cache not found")

# One state per FIA region, read/clipped once and reused by every test below,
# since clipping a full state extract (NC/CO/OR) takes several seconds and
# multiple tests need each state.
states <- c("RI", "NC", "CO", "OR")
dbs <- lapply(states, \(st) clipFIA(readFIA(validation_data_dir, states = st), mostRecent = TRUE))
names(dbs) <- states
db_ri <- dbs[["RI"]]

# Test 3 ------------------------------
# tpa(treeList = TRUE) fed into customPSE() (tree-area ratio: TPA, BAA per
# acre of forest land) should exactly reproduce tpa()'s own population-level
# point estimates, SEs, and plot counts -- the default (unrestricted) case.
# Prior to the fix documented in customPSE.md, nPlots_x/nPlots_y were
# inflated to the full forested-plot count even here (not just under a
# restrictive treeDomain), because every forested condition appears in the
# treeList regardless of whether it has a qualifying tree.
for (st in states) {
  test_that(paste("customPSE() matches tpa() (default case) (", st, ")"), {
    db_st <- dbs[[st]]
    tl <- tpa(db_st, treeList = TRUE)
    pop <- as.data.frame(tpa(db_st))

    out <- as.data.frame(customPSE(
      db = db_st,
      x = dplyr::select(tl, -c(AREA_BASIS)), xVars = c(TPA, BAA),
      y = dplyr::select(tl, -c(TREE_BASIS)), yVars = PROP_FOREST
    ))

    expect_equal(out$TPA_RATIO, pop$TPA, tolerance = 1e-9)
    expect_equal(out$BAA_RATIO, pop$BAA, tolerance = 1e-9)
    expect_equal(out$TPA_RATIO_SE, pop$TPA_SE, tolerance = 1e-9)
    expect_equal(out$BAA_RATIO_SE, pop$BAA_SE, tolerance = 1e-9)
    expect_equal(out$nPlots_x, pop$nPlots_TREE)
    expect_equal(out$nPlots_y, pop$nPlots_AREA)
  })
}

# Test 4 ------------------------------
# Same check under a restrictive treeDomain (large trees only), which
# produces many forested conditions with zero qualifying trees -- the case
# that originally exposed the nPlots_x/nPlots_y bug most clearly (RI:
# nPlots_TREE 58 vs. nPlots_AREA 127, i.e. customPSE() previously reported
# nPlots_x = 127 instead of 58).
for (st in states) {
  test_that(paste("customPSE() matches tpa() (treeDomain = DIA >= 20) (", st, ")"), {
    db_st <- dbs[[st]]
    tl <- tpa(db_st, treeList = TRUE, treeDomain = DIA >= 20)
    pop <- as.data.frame(tpa(db_st, treeDomain = DIA >= 20))

    out <- as.data.frame(customPSE(
      db = db_st,
      x = dplyr::select(tl, -c(AREA_BASIS)), xVars = c(TPA, BAA),
      y = dplyr::select(tl, -c(TREE_BASIS)), yVars = PROP_FOREST
    ))

    expect_equal(out$TPA_RATIO, pop$TPA, tolerance = 1e-9)
    expect_equal(out$BAA_RATIO, pop$BAA, tolerance = 1e-9)
    expect_equal(out$TPA_RATIO_SE, pop$TPA_SE, tolerance = 1e-9)
    expect_equal(out$BAA_RATIO_SE, pop$BAA_SE, tolerance = 1e-9)
    expect_equal(out$nPlots_x, pop$nPlots_TREE)
    expect_equal(out$nPlots_y, pop$nPlots_AREA)
  })
}

# Test 5 ------------------------------
# volume(treeList = TRUE) fed into customPSE() should exactly reproduce
# volume()'s own population-level estimates, under the same restrictive
# treeDomain used above (shares the tpaStarter.R/volumeStarter.R treeList
# construction pattern, so is subject to the same fix).
for (st in states) {
  test_that(paste("customPSE() matches volume() (treeDomain = DIA >= 20) (", st, ")"), {
    db_st <- dbs[[st]]
    tl <- volume(db_st, treeList = TRUE, treeDomain = DIA >= 20)
    pop <- as.data.frame(volume(db_st, treeDomain = DIA >= 20))

    out <- as.data.frame(customPSE(
      db = db_st,
      x = dplyr::select(tl, -c(AREA_BASIS)), xVars = BOLE_CF_ACRE,
      y = dplyr::select(tl, -c(TREE_BASIS)), yVars = PROP_FOREST
    ))

    expect_equal(out$BOLE_CF_ACRE_RATIO, pop$BOLE_CF_ACRE, tolerance = 1e-9)
    expect_equal(out$BOLE_CF_ACRE_RATIO_SE, pop$BOLE_CF_ACRE_SE, tolerance = 1e-9)
    expect_equal(out$nPlots_x, pop$nPlots_TREE)
    expect_equal(out$nPlots_y, pop$nPlots_AREA)
  })
}

# Test 6 ------------------------------
# area(condList = TRUE) fed into customPSE() as a self-ratio (numerator =
# denominator = the same restricted condList) should exactly reproduce
# area()'s own nPlots_AREA_NUM under the same areaDomain. Condition-only
# inputs never carry a TREE_BASIS-style phantom-row problem (AREA_BASIS is
# never NA in a condList), so this is a confirmatory check that the fix in
# sumToPlot() didn't disturb the already-correct area-only path.
for (st in states) {
  test_that(paste("customPSE() matches area() nPlots (areaDomain, mesic classes) (", st, ")"), {
    db_st <- dbs[[st]]
    cl <- area(db_st, condList = TRUE, areaDomain = PHYSCLCD %in% 21:29)
    pop <- as.data.frame(area(db_st, areaDomain = PHYSCLCD %in% 21:29))

    out <- as.data.frame(customPSE(
      db = db_st,
      x = cl, xVars = c(NUM = PROP_FOREST),
      y = cl, yVars = PROP_FOREST
    ))

    expect_equal(out$nPlots_x, pop$nPlots_AREA_NUM)
    expect_equal(out$nPlots_y, pop$nPlots_AREA_NUM)
  })
}

# Test 7 ------------------------------
# xGrpBy = SPCD: customPSE() should reproduce tpa(bySpecies = TRUE)'s
# per-species point estimates and nPlots_x exactly -- this validates that a
# domain filter survives customPSE()'s own grpBy/join path rather than being
# silently dropped for some groups (the historical area()/areaChange() bug
# pattern from v1.1.1), and that the nPlots fix holds per-group, not just in
# the ungrouped case. RI only, to keep this test lightweight.
test_that("customPSE() xGrpBy = SPCD matches tpa(bySpecies = TRUE) per species (RI)", {
  tl <- tpa(db_ri, treeList = TRUE, bySpecies = TRUE)
  pop <- as.data.frame(tpa(db_ri, bySpecies = TRUE))

  out <- as.data.frame(customPSE(
    db = db_ri,
    x = dplyr::select(tl, -c(AREA_BASIS)), xVars = TPA, xGrpBy = SPCD,
    y = dplyr::select(tl, -c(TREE_BASIS, SPCD, COMMON_NAME, SCIENTIFIC_NAME)),
    yVars = PROP_FOREST
  ))

  merged <- merge(out[, c("SPCD", "TPA_RATIO", "TPA_RATIO_SE", "nPlots_x", "nPlots_y")],
                   pop[, c("SPCD", "TPA", "TPA_SE", "nPlots_TREE", "nPlots_AREA")],
                   by = "SPCD")
  expect_equal(nrow(merged), nrow(pop))
  expect_equal(merged$TPA_RATIO, merged$TPA, tolerance = 1e-9)
  expect_equal(merged$TPA_RATIO_SE, merged$TPA_SE, tolerance = 1e-9)
  expect_equal(merged$nPlots_x, merged$nPlots_TREE)
  # yGrpBy is unset (denominator is total forest land, not per-species), so
  # nPlots_y is constant across species and equals the ungrouped forest-land
  # plot count.
  expect_true(all(merged$nPlots_y == merged$nPlots_AREA[1]))
})

# Test 8 ------------------------------
# Internal consistency: TOTAL / denominator TOTAL reproduces the ratio
# (doesn't require comparison to another function).
for (st in states) {
  test_that(paste("customPSE() totals are consistent with ratio estimates (", st, ")"), {
    db_st <- dbs[[st]]
    tl <- tpa(db_st, treeList = TRUE)
    out <- as.data.frame(customPSE(
      db = db_st,
      x = dplyr::select(tl, -c(AREA_BASIS)), xVars = c(TPA, BAA),
      y = dplyr::select(tl, -c(TREE_BASIS)), yVars = PROP_FOREST
    ))
    expect_equal(out$TPA_TOTAL / out$PROP_FOREST_TOTAL, out$TPA_RATIO, tolerance = 1e-9)
    expect_equal(out$BAA_TOTAL / out$PROP_FOREST_TOTAL, out$BAA_RATIO, tolerance = 1e-9)
  })
}

# Test 9 ------------------------------
# A treeDomain matching no trees should return a clean 0-row result (every
# condition's TREE_BASIS is NA, so sumToPlot() filters everything out), not
# error or warn.
test_that("customPSE() handles an empty treeDomain without warning", {
  tl <- tpa(db_ri, treeList = TRUE, treeDomain = SPCD == 999)
  expect_no_warning(
    out <- as.data.frame(customPSE(
      db = db_ri,
      x = dplyr::select(tl, -c(AREA_BASIS)), xVars = TPA,
      y = dplyr::select(tl, -c(TREE_BASIS)), yVars = PROP_FOREST
    ))
  )
  expect_equal(nrow(out), 0)
})

# Test 10 -----------------------------
# Regression test for GitHub issue #47: a spatial mask spanning multiple
# states whose `mostRecent` evaluations fall in different years (here KY
# 2023 vs. OH/WV 2024) should collapse to a single combined estimate, not
# one row per state/year -- customPSE() previously always treated `mr` as
# FALSE (see customPSE.md, "Fixed" #2), because it checked `mostRecent` on
# `db` *after* readRemoteHelper() had already pared `db` down to only the
# named FIA tables, silently dropping the `mostRecent` marker clipFIA()
# attaches to the top-level list. Requires sf and a mask spanning WV/OH/KY
# (available in the same local FIADB cache used above), so this is skipped
# separately if sf isn't installed or those states aren't cached.
test_that("customPSE() combines a multi-state mostRecent mask into one row (issue #47)", {
  skip_if_not_installed("sf")
  skip_if_not(all(file.exists(file.path(validation_data_dir, c("WV_PLOT.csv", "OH_PLOT.csv", "KY_PLOT.csv")))),
              "WV/OH/KY not found in local FIADB validation data cache")

  poly <- sf::st_sfc(sf::st_polygon(list(cbind(c(-83, -82, -82, -83, -83),
                                                c(38, 38, 39, 39, 38)))),
                      crs = 4326) %>% sf::st_sf()

  fiaMulti <- readFIA(validation_data_dir, states = c("WV", "OH", "KY"), inMemory = TRUE)
  fiaMulti <- clipFIA(fiaMulti, mostRecent = TRUE, mask = poly)

  # Sanity check: the mask really does span plots in different states with
  # different mostRecent years, or this test wouldn't exercise the bug.
  statesInMask <- unique(fiaMulti$PLOT$STATECD)
  yearsInMask <- unique(fiaMulti$PLOT$MEASYEAR[fiaMulti$PLOT$prev == 0])
  skip_if_not(length(statesInMask) > 1 && length(yearsInMask) > 1,
              "Mask no longer spans multiple states/years with this data cache")

  cl <- area(fiaMulti, landType = "forest", condList = TRUE)
  out <- as.data.frame(customPSE(db = fiaMulti, x = cl, xVars = c(NUM = PROP_FOREST),
                                  y = cl, yVars = PROP_FOREST))
  pop <- as.data.frame(area(fiaMulti, landType = "forest"))

  expect_equal(nrow(out), 1)
  expect_equal(out$YEAR, max(pop$YEAR))
  expect_equal(out$NUM_TOTAL, pop$AREA_TOTAL, tolerance = 1e-9)
  expect_equal(out$nPlots_x, pop$nPlots_AREA_NUM)
})
