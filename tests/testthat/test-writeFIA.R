# Test writeFIA() ---------------------------------------------------------

# FIADB CNs can have 16 significant digits, but data.table::fwrite() writes
# doubles with at most 15, so a readFIA() -> writeFIA() round trip used to
# silently round them (e.g., 1097572188290487 -> 1097572188290490) and break
# joins between tables.
test_that("writeFIA() writes 16-digit CNs without loss of precision", {
  cond <- data.frame(CN = c(1097572188290488, 815129906290488),
                     PLT_CN = c(1097572188290487, 815129906290487),
                     PREV_PLT_CN = c(NA, 815129906290486),
                     STATECD = 44, CONDID = 1,
                     P2PNTCNT_EU = 12)
  db <- structure(list(COND = cond), class = 'FIA.Database')
  dir <- file.path(tempdir(), 'writeFIA-test')
  on.exit(unlink(dir, recursive = TRUE))

  suppressMessages(writeFIA(db, dir = dir, byState = TRUE))
  out <- data.table::fread(file.path(dir, 'RI_COND.csv'), integer64 = 'double')
  expect_identical(out$CN, cond$CN)
  expect_identical(out$PLT_CN, cond$PLT_CN)
  expect_identical(out$PREV_PLT_CN, cond$PREV_PLT_CN)
  expect_equal(out$P2PNTCNT_EU, cond$P2PNTCNT_EU)

  suppressMessages(writeFIA(db, dir = dir, byState = FALSE))
  out <- data.table::fread(file.path(dir, 'COND.csv'), integer64 = 'double')
  expect_identical(out$PLT_CN, cond$PLT_CN)
  expect_identical(out$PREV_PLT_CN, cond$PREV_PLT_CN)
})
