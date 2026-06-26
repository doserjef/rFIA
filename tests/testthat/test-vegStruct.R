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


