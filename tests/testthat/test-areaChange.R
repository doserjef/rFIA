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

