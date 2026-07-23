# Test diversity() --------------------------------------------------------

skip_on_cran()

# Testing data
data(fiaRI)
data(countiesRI)
# Get most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Test 1 ------------------------------
out <- diversity(fiaRI, polys = countiesRI, returnSpatial = TRUE)

test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})
test_that("out is of class sf", {
  expect_s3_class(out, "sf")
})

# Test 2 ------------------------------
out <- diversity(db = fiaRI_mr, landType = 'forest', treeType = 'live')
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 3 ------------------------------
out <- diversity(db = fiaRI_mr, landType = 'forest', treeType = 'live',
                 byPlot = TRUE)
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 4 ------------------------------
# Most recent estimates grouped by stand age on forest land.
fiaRI_mr$COND$STAND_AGE <- makeClasses(fiaRI_mr$COND$STDAGE, interval = 10)
out <- diversity(db = fiaRI_mr, grpBy = STAND_AGE)
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})


# Test 5 ------------------------------
# Estimates for live white pine (> 12" DBH) on forested mesic sites
# (all available inventories)
out <- diversity(fiaRI, treeType = 'live', treeDomain = DIA > 12,
                 areaDomain = PHYSCLCD %in% 21:29) # Mesic Physiographic classes
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})
test_that("multiple years", {
  expect_gt(length(unique(out$YEAR)), 1)
})

# Test 6 ------------------------------
# Most recent estimates for growing stock on timberland by size class
out <- diversity(fiaRI_mr, treeType = 'gs', landType = 'timber',
                 bySizeClass = TRUE)
test_that("out is of class tbl_df", {
  expect_s3_class(out, "tbl_df")
})

# Test 7 ------------------------------
# Most recent estimates on forestland in user-defined polygons
out <- diversity(fiaRI_mr, landType = 'forest', polys = countiesRI,
                 returnSpatial = TRUE)
plot.out <- plotFIA(out, H_a)
test_that("out is of class sf", {
  expect_s3_class(out, "sf")
})
test_that('plot.out is a ggplot', {
  expect_s3_class(plot.out, 'gg')
})

# Numeric validation ---------------------------------------------------------
# diversity() has no EVALIDator ground truth at all: Shannon/richness/
# equitability indices are not FIADB/EVALIDator population attributes.
# Validation here is therefore internal consistency, cross-checks against
# tpa()'s nPlots_AREA/AREA_TOTAL (already validated against EVALIDator; see
# tpa.md) for the same landType/areaDomain/grpBy restriction, and hand
# calculations replicating divIndex()'s own Shannon/richness formula
# independently from raw data. See core_references/validation/diversity.md
# for full methodology/results.
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
# nPlots_AREA/AREA_TOTAL cross-checked against tpa() (already validated
# against EVALIDator; see tpa.md) for the same landType/areaDomain
# restriction. Regression test for the nPlots_AREA phantom-row bug
# (diversity.md, "Fixed" #1): before the fix, nPlots_AREA didn't reflect
# landType = 'timber'/areaDomain restrictions.
for (st in states) {
  db_st <- dbs[[st]]
  test_that(paste("diversity() nPlots_AREA/AREA_TOTAL match tpa() exactly (", st, ")"), {
    for (lt in c('forest', 'timber')) {
      dv <- as.data.frame(diversity(db_st, landType = lt, totals = TRUE))
      ref <- as.data.frame(tpa(db_st, landType = lt, treeType = 'live', totals = TRUE))
      expect_equal(unique(dv$nPlots_AREA), ref$nPlots_AREA, label = paste("landType =", lt))
      expect_equal(unique(dv$AREA_TOTAL), ref$AREA_TOTAL, label = paste("landType =", lt))
    }
    dv_ad <- as.data.frame(diversity(db_st, areaDomain = PHYSCLCD %in% 21:29, totals = TRUE))
    ref_ad <- as.data.frame(tpa(db_st, areaDomain = PHYSCLCD %in% 21:29, treeType = 'live', totals = TRUE))
    expect_equal(unique(dv_ad$nPlots_AREA), ref_ad$nPlots_AREA, label = "areaDomain")
    expect_equal(unique(dv_ad$AREA_TOTAL), ref_ad$AREA_TOTAL, label = "areaDomain")
  })
}

# Test 9 ------------------------------
# grpBy = OWNGRPCD (a COND-table variable, constant per condition) should
# match tpa()'s grouped AREA_TOTAL/nPlots_AREA exactly -- confirms the
# aGrpBy fix (see "Fixed" #2 below) didn't disturb the already-correct
# COND-level grpBy case.
test_that("diversity() grpBy = OWNGRPCD matches tpa() per group (RI)", {
  # grpByToChar() (shared by diversity() and tpa()) emits a pre-existing,
  # unrelated many-to-many join warning on this state's full grpBy join
  # (already observed in test-area.R); suppressed on both sides since it
  # isn't part of what's being tested here.
  dv <- suppressWarnings(as.data.frame(diversity(db_ri, grpBy = OWNGRPCD, totals = TRUE)))
  ref <- suppressWarnings(as.data.frame(tpa(db_ri, grpBy = OWNGRPCD, treeType = 'live', totals = TRUE)))
  dv <- dv[order(dv$OWNGRPCD), ]
  ref <- ref[order(ref$OWNGRPCD), ]
  expect_equal(dv$AREA_TOTAL, ref$AREA_TOTAL)
  expect_equal(dv$nPlots_AREA, ref$nPlots_AREA)
})

# Test 10 ------------------------------
# Regression test for the aGrpBy bug (see diversity.md, "Fixed" #2):
# grpBy = SPGRPCD (a TREE-table variable, NOT constant per condition) must
# report the SAME AREA_TOTAL (the full landType domain's area) for every
# group -- matching tpa(bySpecies = TRUE)'s already-validated convention,
# where aGrpBy also excludes the TREE-level SPCD column. Before the fix,
# each condition's area was collapsed into whichever one SPGRPCD bin
# happened to come first per (PLT_CN, CONDID), so summing AREA_TOTAL across
# bins came out close to (instead of far exceeding) the true total, and
# nPlots_AREA/AREA_TOTAL varied incorrectly by group.
test_that("diversity() grpBy = SPGRPCD reports a consistent AREA_TOTAL across groups (RI)", {
  # See Test 9 above re: the suppressed grpByToChar() warning.
  dv <- suppressWarnings(as.data.frame(diversity(db_ri, grpBy = SPGRPCD, totals = TRUE)))
  ref <- suppressWarnings(as.data.frame(tpa(db_ri, treeType = 'live', totals = TRUE)))
  expect_true(length(unique(dv$AREA_TOTAL)) == 1)
  expect_equal(unique(dv$AREA_TOTAL), ref$AREA_TOTAL)
  expect_true(length(unique(dv$nPlots_AREA)) == 1)
  expect_equal(unique(dv$nPlots_AREA), ref$nPlots_AREA)
})

# Test 11 ------------------------------
# Regression test for the same aGrpBy bug via bySizeClass = TRUE: Eh
# (Shannon's Equitability, per divIndex()'s H/S formula -- not the more
# common H/ln(S) Pielou's evenness, see diversity.md Notes) is
# mathematically bounded within [0, 1/e] (~0.368) at the per-condition
# level, and since alpha-level Eh_a is an area-weighted average of bounded
# per-condition values, it must stay within that same bound. Before the
# fix, a corrupted (too-small) AREA_TOTAL denominator for some sizeClass
# bins pushed Eh_a above 1 in this exact case.
test_that("diversity() Eh_a stays within its mathematical bound with bySizeClass (RI)", {
  out <- as.data.frame(diversity(db_ri, bySizeClass = TRUE))
  expect_true(all(out$Eh_a <= (1 / exp(1)) + 1e-9, na.rm = TRUE))
})

# Test 12 ------------------------------
# Hand calculation of the Shannon index (H), richness (S), and Shannon's
# equitability (Eh) formulas from raw TREE data, independent of the
# package code, for a specific plot (RI, pltID "1_44_1_91"): 21 live trees
# across 3 species (SPCD 316, 129, 833), each with equal TPA_UNADJ
# (6.018046). By hand: p = (14/21, 1/21, 2/21), H = -sum(p*log(p)) =
# 0.4851045, S = 3, Eh = H/S = 0.1617015.
test_that("diversity() byPlot H/S/Eh match a hand calculation from raw data (RI)", {
  bp <- as.data.frame(diversity(db_ri, byPlot = TRUE))
  row <- bp[bp$pltID == "1_44_1_91", ]
  expect_equal(nrow(row), 1)
  expect_equal(row$H, 0.4851045, tolerance = 1e-6)
  expect_equal(row$S, 3)
  expect_equal(row$Eh, 0.1617015, tolerance = 1e-6)
})

# Test 13 ------------------------------
# Hand calculation of gamma diversity (H_g/S_g): pooling every live tree on
# forest land statewide (RI, the exact plot set from `pops$PLT_CN`, i.e.
# the plots actually used in the current TI evaluation) by species,
# independent of the package code.
test_that("diversity() H_g/S_g match a hand calculation pooling all of RI's live forest trees", {
  mr <- checkMR(db_ri, 0)
  pops <- handlePops(db_ri, evalType = c('VOL'), method = 'TI', mr)
  cond <- db_ri$COND[db_ri$COND$COND_STATUS_CD == 1 & db_ri$COND$PLT_CN %in% pops$PLT_CN,
                     c("PLT_CN", "CONDID")]
  tree <- merge(db_ri$TREE, cond, by = c("PLT_CN", "CONDID"))
  tree <- tree[!is.na(tree$DIA) & tree$TPA_UNADJ > 0 & tree$STATUSCD == 1, ]
  tpaBySp <- tapply(tree$TPA_UNADJ, tree$SPCD, sum)
  p <- tpaBySp / sum(tpaBySp)
  handH <- -sum(p * log(p))
  handS <- length(tpaBySp)

  out <- as.data.frame(diversity(db_ri))
  expect_equal(out$H_g, handH, tolerance = 1e-6)
  expect_equal(out$S_g, handS)
})

# Test 14 ------------------------------
# Regression test for the CONDID-omitted-from-distinct() bug (see
# diversity.md, "Fixed" #3): NC plot 1150116756290487 has two zero-tree
# forest conditions (CONDID 2 and 3, CONDPROP_UNADJ 0.25 each). Before the
# fix, distinct(PLT_CN, SUBP, TREE) collapsed both conditions' phantom
# "no tree" rows into one, so condList = TRUE reported CONDID 3 as
# H = S = Eh = NA (dropped entirely from the join) instead of the correct
# H = S = 0 (a real, meaningful "no species present" value).
test_that("diversity() condList correctly reports both zero-tree conditions on one plot (NC)", {
  db_nc <- dbs[["NC"]]
  cl <- as.data.frame(diversity(db_nc, condList = TRUE))
  row <- cl[cl$PLT_CN == "1150116756290487", ]
  expect_equal(nrow(row), 2)
  expect_equal(sort(row$CONDID), c(2, 3))
  expect_true(all(row$H == 0))
  expect_true(all(row$S == 0))
  expect_true(all(row$PROP_FOREST == 0.25))
})

# Test 15 ------------------------------
# treeDomain interaction (DIA > 12, RI): H_g/S_g matches a hand calculation
# applying the same filter directly to raw data (validates that the
# treeDomain filter survives the divIndex()/gamma-diversity computation
# path, the historical area()/areaChange() bug pattern from v1.1.1).
test_that("diversity() matches a hand calculation with treeDomain (DIA > 12) (RI)", {
  mr <- checkMR(db_ri, 0)
  pops <- handlePops(db_ri, evalType = c('VOL'), method = 'TI', mr)
  cond <- db_ri$COND[db_ri$COND$COND_STATUS_CD == 1 & db_ri$COND$PLT_CN %in% pops$PLT_CN,
                     c("PLT_CN", "CONDID")]
  tree <- merge(db_ri$TREE, cond, by = c("PLT_CN", "CONDID"))
  tree <- tree[!is.na(tree$DIA) & tree$TPA_UNADJ > 0 & tree$STATUSCD == 1 & tree$DIA > 12, ]
  tpaBySp <- tapply(tree$TPA_UNADJ, tree$SPCD, sum)
  p <- tpaBySp / sum(tpaBySp)
  handH <- -sum(p * log(p))
  handS <- length(tpaBySp)

  out <- as.data.frame(diversity(db_ri, treeDomain = DIA > 12))
  expect_equal(out$H_g, handH, tolerance = 1e-6)
  expect_equal(out$S_g, handS)
})

# Test 16 ------------------------------
# returnSpatial should only add geometry, not change any numeric estimate.
test_that("returnSpatial does not change numeric estimates (RI, by county)", {
  out_sf <- as.data.frame(diversity(db_ri, polys = countiesRI, returnSpatial = TRUE))
  out_df <- as.data.frame(diversity(db_ri, polys = countiesRI, returnSpatial = FALSE))
  out_sf <- out_sf[, names(out_df)]
  out_sf <- out_sf[order(out_sf$COUNTY), ]
  out_df <- out_df[order(out_df$COUNTY), ]
  expect_equal(out_sf, out_df)
})

# Test 17 ------------------------------
# An areaDomain matching no conditions should return a clean 0-row result,
# not error or emit an internal max()-on-empty-vector warning. Regression
# test for a bug found this pass (diversity.md, "Fixed" #3): the STAGE-
# equivalent tree list was missing the `!is.na(CONDID)` filter its
# condition list `a` already has, so a phantom "no condition" row still got
# an H = S = 0 classification via divIndex()'s empty-species fallback,
# surviving as a spurious result instead of a genuinely empty one.
test_that("diversity() handles an empty areaDomain without warning", {
  expect_no_warning(
    out <- as.data.frame(diversity(db_ri, areaDomain = STATECD == 999))
  )
  expect_equal(nrow(out), 0)
})

# Test 18 ------------------------------
# An empty treeDomain (matching no trees at all, but leaving the area
# domain unrestricted) is a genuinely different case from an empty
# areaDomain: every forest condition still exists and still contributes
# real area, there just aren't any qualifying trees to compute diversity
# from. H_a/S_a = 0 (a real, meaningful "no species present" value, not a
# phantom artifact) is the correct result here, and H_g/S_g/H_b/Eh_b/S_b
# are NA (the "full" pooled tree list is empty, so the gamma-diversity join
# finds no match) -- both are expected, checked as a non-regression pin.
test_that("diversity() handles an empty treeDomain with a real zero (not empty) result", {
  out <- as.data.frame(diversity(db_ri, treeDomain = SPCD == 999))
  expect_equal(nrow(out), 1)
  expect_equal(out$H_a, 0)
  expect_equal(out$S_a, 0)
  expect_equal(out$Eh_a, 0)
  expect_true(is.na(out$H_g))
  expect_true(is.na(out$S_g))
})
