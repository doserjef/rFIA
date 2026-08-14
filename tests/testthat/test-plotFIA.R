# Test plotFIA() --------------------------------------------------------

skip_on_cran()

data(fiaRI)
data(countiesRI)

# Most recent subset
fiaRI_mr <- clipFIA(fiaRI)

# Precompute the summaries plotFIA() draws from, reused across tests below
tpaTS <- tpa(fiaRI, landType = 'forest', treeType = 'live')
tpaSp <- tpa(fiaRI, landType = 'forest', treeType = 'live', bySpecies = TRUE)
tpaPoly <- tpa(fiaRI, polys = countiesRI, returnSpatial = TRUE)
tpaPolyMR <- tpa(fiaRI_mr, polys = countiesRI, returnSpatial = TRUE)
tpaPts <- tpa(fiaRI_mr, byPlot = TRUE, returnSpatial = TRUE)
tpaSizeClass <- tpa(fiaRI_mr, bySpecies = TRUE, bySizeClass = TRUE)

# Test 1 ------------------------------
# FIA.Database input maps plot locations
test_that("plotFIA() maps plot locations from an FIA.Database", {
  out <- plotFIA(fiaRI)
  expect_s3_class(out, "gg")
})

# Test 2 ------------------------------
# Spatial choropleth (polygons), most recent subset
test_that("plotFIA() produces a choropleth from a spatial (sf) summary", {
  out <- plotFIA(tpaPolyMR, TPA)
  expect_s3_class(out, "gg")
})

# Test 3 ------------------------------
# Spatial choropleth, faceted by year (multi-year summary)
test_that("plotFIA() facets a spatial summary by YEAR", {
  out <- plotFIA(tpaPoly, TPA, facet = TRUE)
  expect_s3_class(out, "gg")
})

# Test 4 ------------------------------
# Spatial points (byPlot + returnSpatial), ungrouped
test_that("plotFIA() plots ungrouped spatial points", {
  out <- plotFIA(tpaPts, TPA)
  expect_s3_class(out, "gg")
})

# Test 5 ------------------------------
# Spatial points, grouped by a categorical variable
test_that("plotFIA() plots grouped spatial points", {
  ptsGrp <- tpaPts
  ptsGrp$tpaCat <- makeClasses(ptsGrp$TPA, interval = 50)
  out <- plotFIA(ptsGrp, TPA, grp = tpaCat)
  expect_s3_class(out, "gg")
})

# Test 6 ------------------------------
# Simple (ungrouped) time series
test_that("plotFIA() produces a time series when x is unspecified", {
  out <- plotFIA(tpaTS, TPA)
  expect_s3_class(out, "gg")
})

# Test 7 ------------------------------
# Simple time series with 95% CI error bars
test_that("plotFIA() adds error bar layers when se = TRUE", {
  out <- plotFIA(tpaTS, TPA, se = TRUE)
  expect_s3_class(out, "gg")
  expect_true(all(c("ymin", "ymax") %in% names(out$data)))
  expect_gte(length(out$layers), 2)
})

# Test 8 ------------------------------
# Grouped time series (bySpecies), restricted to top n.max groups
test_that("plotFIA() restricts to n.max groups, chosen by mean y", {
  out <- plotFIA(tpaSp, TPA, grp = COMMON_NAME, n.max = 3)
  expect_s3_class(out, "gg")
  expect_equal(length(unique(out$data$grpVar)), 3)
})

# Test 9 ------------------------------
# Grouped time series with error bars
test_that("plotFIA() combines grp and se = TRUE", {
  out <- plotFIA(tpaSp, TPA, grp = COMMON_NAME, n.max = 3, se = TRUE)
  expect_s3_class(out, "gg")
})

# Test 10 ------------------------------
# Non-time-series x-axis (documented example): BAA by size class, grouped by species
test_that("plotFIA() supports a non-time-series x-axis (x = sizeClass)", {
  out <- plotFIA(tpaSizeClass, y = BAA, grp = COMMON_NAME, x = sizeClass, n.max = 4)
  expect_s3_class(out, "gg")
})

# Test 11 ------------------------------
# plot.title / y.lab / x.lab / legend.title are passed through to the plot
test_that("plotFIA() applies user-supplied labels", {
  out <- plotFIA(tpaTS, TPA, plot.title = "My Title", y.lab = "Y", x.lab = "X")
  expect_equal(out$labels$title, "My Title")
  expect_equal(out$labels$y, "Y")
  expect_equal(out$labels$x, "X")
})

# Test 12 ------------------------------
# y is required unless data is an FIA.Database
test_that("plotFIA() errors when y is missing for non-FIA.Database data", {
  expect_error(plotFIA(as.data.frame(tpaTS)))
})

# Test 13 ------------------------------
# savePath/fileName must be specified together
test_that("plotFIA() warns when only one of savePath/fileName is given", {
  expect_warning(plotFIA(tpaPolyMR, TPA, savePath = tempdir()))
  expect_warning(plotFIA(tpaPolyMR, TPA, fileName = "x.png"))
})

# Test 14 ------------------------------
# Saving a static plot to disk
test_that("plotFIA() saves a static plot when savePath and fileName are given", {
  tmp <- tempfile("plotFIA_")
  dir.create(tmp)
  on.exit(unlink(tmp, recursive = TRUE), add = TRUE)
  out <- plotFIA(tpaPolyMR, TPA, savePath = tmp, fileName = "test_plot.png")
  expect_s3_class(out, "gg")
  expect_true(file.exists(file.path(tmp, "test_plot.png")))
})

# Regression tests: animate = TRUE ------------------------------------------
# gganimate is Suggests-only and never attached by library(), so plotFIA()
# must fully qualify every gganimate:: call it makes. These reproduce two
# bugs found during validation where that wasn't the case: transition_manual()
# was unqualified in the non-spatial branch, and anim_save() was unqualified
# in the shared save step used by both branches.
skip_if_not_installed("gganimate")

# Test 15 ------------------------------
# animate = TRUE on a non-spatial (time series) summary
test_that("plotFIA() animates a non-spatial time series", {
  out <- plotFIA(tpaSp, TPA, grp = COMMON_NAME, n.max = 3, animate = TRUE)
  expect_s3_class(out, "gganim")
})

# Test 16 ------------------------------
# animate = TRUE on a spatial (choropleth) summary
test_that("plotFIA() animates a spatial summary", {
  out <- plotFIA(tpaPoly, TPA, animate = TRUE)
  expect_s3_class(out, "gganim")
})

# Test 17 ------------------------------
# Saving an animation (spatial and non-spatial) to disk
test_that("plotFIA() saves an animated .gif to disk", {
  tmp <- tempfile("plotFIA_anim_")
  dir.create(tmp)
  on.exit(unlink(tmp, recursive = TRUE), add = TRUE)
  plotFIA(tpaPoly, TPA, animate = TRUE, savePath = tmp, fileName = "test_anim.gif")
  expect_true(file.exists(file.path(tmp, "test_anim.gif")))
})

# Test 18 ------------------------------
# min.year restricts which years are included in an animation, but has no
# effect on a static (animate = FALSE) plot.
test_that("min.year filters animation frames but not static plots", {
  yrs <- sort(unique(tpaPoly$YEAR))
  skip_if(length(yrs) < 2, "tpaPoly needs at least two distinct YEARs")
  cutoff <- yrs[2]

  animFull <- plotFIA(tpaPoly, TPA, animate = TRUE, min.year = min(yrs))
  animTrim <- plotFIA(tpaPoly, TPA, animate = TRUE, min.year = cutoff)
  expect_true(all(unique(animTrim$data$YEAR) >= cutoff))
  expect_lt(length(unique(animTrim$data$YEAR)), length(unique(animFull$data$YEAR)))

  staticFull <- plotFIA(tpaPoly, TPA, min.year = min(yrs))
  staticTrim <- plotFIA(tpaPoly, TPA, min.year = cutoff)
  expect_equal(sort(unique(staticFull$data$YEAR)), sort(unique(staticTrim$data$YEAR)))
})
