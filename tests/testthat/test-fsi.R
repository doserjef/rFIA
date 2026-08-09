# Test tpa() --------------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
out <- fsi(db = fiaRI_mr, scaleBy = FORTYPCD)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 2 ------------------------------
out <- fsi(db = fiaRI_mr, scaleBy = FORTYPCD,
           byPlot = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
results <- fsi(db = fiaRI_mr,
               scaleBy = FORTYPCD,
               returnBetas = TRUE)

test_that("results is a tbl_df", {
  expect_s3_class(results$results, "tbl_df")
})

test_that("results has two things", {
  expect_equal(length(results), 2)
})

# Test 4 ------------------------------
out <- fsi(fiaRI_mr,
           scaleBy = SITECLCD,
           treeType = 'live',
           treeDomain = SPCD == 129 & DIA > 12,
           areaDomain = PHYSCLCD %in% 21:29)
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Internal consistency checks (no EVALIDator equivalent exists for fsi() --
# see core_references/validation/fsi.md for methodology) --------------------
skip_on_cran()

validation_data_dir <- Sys.getenv("RFIA_VALIDATION_DATA", "~/Dropbox/data/fia")
skip_if_not(dir.exists(validation_data_dir),
            "Local FIADB validation data cache not found")

# One state per FIA region, read/clipped once and reused by every test below.
states <- c("RI", "NC", "CO", "OR")
dbs <- lapply(states, \(st) clipFIA(readFIA(validation_data_dir, states = st), mostRecent = TRUE))
names(dbs) <- states
db_ri <- dbs[["RI"]]

# betas = alpha=1, rate=0 collapses Nmax(S) (eq. 1, Stanke et al. 2021) to 1,
# so rd == TPA_UNADJ exactly. This isolates eqs. 2-4 from the stochastic JAGS
# curve fit, and lets CURR_RD/PREV_RD be cross-checked against tpa()
# (already validated), since they reduce to a plain live-tree TPA estimate.
trivialBetas <- data.frame(grps = 1, alpha = 1, rate = 0, n = 1)

# Test 5 ------------------------------
# Formula fidelity: byPlot output must satisfy eqs. 2-4 exactly.
for (st in states) {
  test_that(paste("fsi() byPlot output satisfies eqs. 2-4 exactly (", st, ")"), {
    out <- as.data.frame(fsi(dbs[[st]], byPlot = TRUE, betas = trivialBetas))
    expect_equal(out$CURR_RD, out$CURR_TPA, tolerance = 1e-9)
    expect_equal(out$PREV_RD, out$PREV_TPA, tolerance = 1e-9)
    expect_equal(out$FSI, (out$CURR_RD - out$PREV_RD) / out$REMPER, tolerance = 1e-9)
    expect_equal(out$PERC_FSI, out$FSI / out$PREV_RD * 100, tolerance = 1e-9)
  })
}

# Test 6 ------------------------------
# Regression test for the REMPER * tAdj bug (see fsi.md, "Fixed" #1): before
# the fix, fsi()'s population-level per-acre CURR_RD was ~50-65% below a
# domain-matched tpa() cross-check, and nPlots exceeded tpa()'s
# nPlots_AREA -- which should never happen, since fsi()'s growth-eligible
# population is a subset of tpa()'s full current-forest population. Both are
# checked directly here.
for (st in states) {
  test_that(paste("fsi() population CURR_RD is consistent with tpa() (", st, ")"), {
    pop <- as.data.frame(fsi(dbs[[st]], betas = trivialBetas))
    tpa_curr <- as.data.frame(tpa(dbs[[st]], landType = 'forest', treeType = 'live'))
    expect_lt(abs(pop$CURR_RD - tpa_curr$TPA) / tpa_curr$TPA, 0.15)
    expect_true(pop$nPlots <= tpa_curr$nPlots_AREA)
  })
}

# Test 7 ------------------------------
# scaleBy group-specific curves regression test (v1.1.3 fix, see NEWS.md):
# group-specific alpha/rate must actually be applied, not the overall mean.
test_that("fsi() applies scaleBy group-specific betas, not the overall mean (RI)", {
  customBetas <- data.frame(grps = c("103", "401"), alpha = c(1, 1e6), rate = 0, n = 1)
  out <- as.data.frame(fsi(db_ri, grpBy = FORTYPCD, scaleBy = FORTYPCD,
                            byPlot = TRUE, betas = customBetas))
  out <- out[out$FORTYPCD %in% c(103, 401), ]
  ratio <- out$CURR_RD / out$CURR_TPA
  expect_equal(ratio[out$FORTYPCD == 103], rep(1, sum(out$FORTYPCD == 103)), tolerance = 1e-6)
  expect_equal(ratio[out$FORTYPCD == 401], rep(1e-6, sum(out$FORTYPCD == 401)), tolerance = 1e-6)
})

# Test 8 ------------------------------
# Domain filters must actually restrict the plot population (the historically
# buggy pattern -- treeDomain/areaDomain silently ignored -- see project plan).
for (st in states) {
  db_st <- dbs[[st]]

  test_that(paste("fsi() nPlots responds to areaDomain (", st, ")"), {
    unrestricted <- as.data.frame(fsi(db_st, betas = trivialBetas))
    restricted <- as.data.frame(fsi(db_st, betas = trivialBetas,
                                     areaDomain = PHYSCLCD %in% 21:29))
    expect_lt(restricted$nPlots, unrestricted$nPlots)
  })
}

test_that("fsi() nPlots responds to treeDomain (RI)", {
  unrestricted <- as.data.frame(fsi(db_ri, betas = trivialBetas))
  restricted <- as.data.frame(fsi(db_ri, betas = trivialBetas, treeDomain = SPCD == 129))
  expect_lt(restricted$nPlots, unrestricted$nPlots)
})

# Test 9 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("fsi() returnSpatial does not change numeric estimates (RI, byPlot)", {
  out_sf <- as.data.frame(fsi(db_ri, betas = trivialBetas, byPlot = TRUE, returnSpatial = TRUE))
  out_df <- as.data.frame(fsi(db_ri, betas = trivialBetas, byPlot = TRUE, returnSpatial = FALSE))
  common <- setdiff(intersect(names(out_sf), names(out_df)), "geometry")
  out_sf <- out_sf[order(out_sf$PLT_CN), common]
  out_df <- out_df[order(out_df$PLT_CN), common]
  expect_equal(out_sf, out_df)
})

# Test 10 ------------------------------
# Regression test for fsi.md "Known issues" D: the totals argument had no
# effect on output and has been removed entirely.
test_that("fsi() no longer has a totals argument", {
  expect_false("totals" %in% names(formals(fsi)))
})

# Test 11 ------------------------------
# Regression test for fsi.md "Known issues" E: an areaDomain matching no
# plots used to crash with "replacement has 1 row, data has 0" (from
# assigning a scalar `grps` column onto a zero-row curve-fit data frame).
# byPlot = FALSE should now return a clean empty result; byPlot = TRUE
# should keep all plot rows with degenerate (0/NA) values, matching how
# tpa()/vitalRates() handle an empty areaDomain.
test_that("fsi() handles an empty areaDomain without erroring (RI)", {
  expect_no_error(
    out <- as.data.frame(fsi(db_ri, betas = trivialBetas, areaDomain = PHYSCLCD == 11))
  )
  expect_equal(nrow(out), 0)

  expect_no_error(
    out_bp <- as.data.frame(fsi(db_ri, betas = trivialBetas, byPlot = TRUE,
                                 areaDomain = PHYSCLCD == 11))
  )
  expect_gt(nrow(out_bp), 0)
  expect_true(all(out_bp$FSI == 0 | is.na(out_bp$FSI)))
})

test_that("fsi() handles an empty areaDomain without erroring, default betas (RI)", {
  expect_no_error(
    out <- as.data.frame(fsi(db_ri, areaDomain = PHYSCLCD == 11))
  )
  expect_equal(nrow(out), 0)
})

# Test 12 ------------------------------
# Regression test for fsi.md "Known issues" F: a treeDomain matching no
# trees used to return a spurious 1-row NaN-filled result instead of a
# clean 0-row result (byPlot = FALSE); byPlot = TRUE keeps all plot rows,
# consistent with tpa()/vitalRates().
test_that("fsi() handles an empty treeDomain without erroring (RI)", {
  out <- as.data.frame(fsi(db_ri, betas = trivialBetas, treeDomain = SPCD == 999))
  expect_equal(nrow(out), 0)

  out_bp <- as.data.frame(fsi(db_ri, betas = trivialBetas, byPlot = TRUE,
                               treeDomain = SPCD == 999))
  expect_gt(nrow(out_bp), 0)
  expect_true(all(out_bp$FSI == 0 | is.na(out_bp$FSI)))
})

# Test 13 ------------------------------
# Regression test for fsi.md "Known issues" G: method = 'ANNUAL' used to
# trigger a dplyr across() deprecation warning.
test_that("fsi() method = 'ANNUAL' does not emit a deprecation warning (RI)", {
  expect_no_warning(
    out <- as.data.frame(fsi(db_ri, betas = trivialBetas, method = "ANNUAL"))
  )
})
