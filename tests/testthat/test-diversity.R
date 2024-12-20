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



