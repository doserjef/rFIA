customPSE <- function(db, x, xVars, xGrpBy = NULL, xTransform = NULL, 
                      y = NULL, yVars = NULL, yGrpBy = NULL, 
                      yTransform = NULL, method = 'TI', lambda = 0.5, 
                      totals = TRUE) {

  # Warnings upfront ----------------------------------------------------------
  # Must have an FIA.Database or a remote one
  if (!c(class(db) %in% c('FIA.Database', 'Remote.FIA.Database'))) {
    stop('Must provide an `FIA.Database` or `Remote.FIA.Database`. See `readFIA` and/or `getFIA` to read and load your FIA data.')
  }

  # PLT_CN must be present in both x and y
  if (!c('PLT_CN' %in% names(x))) {
    stop('`PLT_CN` must be included in `x`. See our website for an example use case of `customPSE()`.')
  }
  if (!is.null(y)) {
    if (!c('PLT_CN' %in% names(y))) {
      stop('`PLT_CN` must be included in `y`. See our website for an example use case of `customPSE()`.')
    }
  }

  # EVAL_TYP must be present in x
  if (!c('EVAL_TYP' %in% names(x))) {
    stop('`EVAL_TYP` must be included in `x`. See our website for an example use case of `customPSE`.')
  }


  # Only one of TREE_BASIS or AREA_BASIS may exist in each dataset
  if (all(c('TREE_BASIS', 'AREA_BASIS') %in% names(x))) {
    stop('Both `TREE_BASIS` and `AREA_BASIS` found in `x`, but only one is allowed. See our website for an example use case of `customPSE()`.')
  } else if (sum(c('TREE_BASIS', 'AREA_BASIS') %in% names(x)) != 1 ) {
    stop('Neither `TREE_BASIS` or `AREA_BASIS` found in `x`, but one is required. See our website for an example use case of `customPSE()`.')
  }
  if (!is.null(y)) {
    if (all(c('TREE_BASIS', 'AREA_BASIS') %in% names(y))) {
      stop('Both `TREE_BASIS` and `AREA_BASIS` found in `y`, but only one is allowed. See our website for an example use case of `customPSE()`.')
    } else if (sum(c('TREE_BASIS', 'AREA_BASIS') %in% names(y)) != 1) {
      stop('Neither `TREE_BASIS` or `AREA_BASIS` found in `y`, but one is required. See our website for an example use case of `customPSE()`.')
    }
  }

  # If using tree variables, SUBP and TREE are required
  # CONDID required for condition variables
  if ('TREE_BASIS' %in% names(x) & !all(c('SUBP', 'TREE') %in% names(x) )) {
    stop('Both `SUBP` and `TREE` required for estimation of tree variables, but are missing from `x`. See our website for an example use case of `customPSE()`.')
  } else if ( 'AREA_BASIS' %in% names(x) & !c('CONDID' %in% names(x))) {
    stop('`CONDID` required for estimation of condition variables, but is missing from `x`. See our website for an example use case of `customPSE()`.')
  }
  if (!is.null(y)) {
    if ('TREE_BASIS' %in% names(y) & !all(c('SUBP', 'TREE') %in% names(y) )) {
      stop('Both `SUBP` and `TREE` required for estimation of tree variables, but are missing from `y`. See our website for an example use case of `customPSE()`.')
    } else if ( 'AREA_BASIS' %in% names(y) & !c('CONDID' %in% names(y))) {
      stop('`CONDID` required for estimation of condition variables, but is missing from `y`. See our website for an example use case of `customPSE()`.')
    }
  }

  # Pull design info from db --------------------------------------------------
  # Make the sure the necessary tables are present in db
  req.tables <- c('PLOT', 'POP_EVAL', 'POP_EVAL_TYP', 'POP_ESTN_UNIT', 'POP_STRATUM', 'POP_PLOT_STRATUM_ASSGN')


  # If remote, read in state by state. Otherwise, drop all unnecessary tables
  remote <- ifelse(class(db) == 'Remote.FIA.Database', 1, 0)
  db <- readRemoteHelper(db$states, db, remote, req.tables, nCores = 1)

  # Pull the appropriate population tables
  mr <- checkMR(db, remote = ifelse(class(db) == 'Remote.FIA.Database', 1, 0))
  pops <- handlePops(db, evalType = x$EVAL_TYP[[1]], method, mr)

  # Prep tables ---------------------------------------------------------------
  # Make sure all variables specified exist in their respective dataset, and we
  # have no duplicates
  if ('TREE_BASIS' %in% names(x)) {
    x.id.vars <- c('SUBP', 'TREE')
  } else {
    x.id.vars <- 'CONDID'
  }
  xVars <- rlang::enquos(xVars)
  xGrpBy <- rlang::enquos(xGrpBy)
  x <- dplyr::select(x, PLT_CN,
                     dplyr::all_of(x.id.vars),
                     dplyr::any_of(c('TREE_BASIS', 'AREA_BASIS')),
                     ## Extras for vitalRates
                     dplyr::any_of(c('ONEORTWO')),
                     !!!xVars, !!!xGrpBy) %>%
    dplyr::distinct()


  if (!is.null(y)) {
    if ('TREE_BASIS' %in% names(y)) {
      y.id.vars <- c('SUBP', 'TREE')
    } else {
      y.id.vars <- 'CONDID'
    }
    yVars <- rlang::enquo(yVars)
    yGrpBy <- rlang::enquos(yGrpBy)
    y <- dplyr::select(y, PLT_CN,
                       dplyr::all_of(y.id.vars),
                       dplyr::any_of(c('TREE_BASIS', 'AREA_BASIS')),
                       !!yVars, !!!yGrpBy) %>%
      dplyr::distinct()
  }


  # Sum variable(s) up to plot-level ------------------------------------------

  # convert to character
  xGrpBy <- names(dplyr::select(x, !!!xGrpBy))
  if (!is.null(y)) {
    yGrpBy <- names(dplyr::select(y, !!!yGrpBy))
    # Make sure all denominator groups are a subset of numerator groups
    if (!all(yGrpBy %in% xGrpBy)) {
      stop('All grouping variables listed in `yGrpBy` must be included in `xGrpBy`. More specifically, `yGrpBy` must be equal to or a subset of `xGrpBy`. ')
    }
  }

  # Sum to plot
  xPlt <- sumToPlot(x, pops, xGrpBy)
  if (!is.null(y)) yPlt <- sumToPlot(y, pops, yGrpBy)

  # Apply transformations to plot-level summaries ------------------------------
  if (!is.null(xTransform)) {
    xPlt <- dplyr::mutate(xPlt,
                        dplyr::across(
                          .cols = c(-ESTN_UNIT_CN, -STRATUM_CN,
                                    -PLT_CN, - {{xGrpBy}}),
                          .fns = xTransform
                        ))
  }
  if (!is.null(y) & !is.null(yTransform)) {
    yPlt <- dplyr::mutate(yPlt,
                          dplyr::across(
                            .cols = c(-ESTN_UNIT_CN, -STRATUM_CN,
                                      -PLT_CN, - {{yGrpBy}}),
                            .fns = yTransform
                          ))
  }

  # Sum up to estimation unit --------------------------------------------------

  if (!is.null(y)) {
    # Adding YEAR to groups
    xGrpBy <- c('YEAR', xGrpBy)
    yGrpBy <- c('YEAR', yGrpBy)

    # Sum variable(s) up to strata then estimation unit level
    eu.sums <- sumToEU(db, xPlt, yPlt, pops, xGrpBy, yGrpBy, method = method, lambda)
    xEst <- eu.sums$x
    yEst <- eu.sums$y
  } else {
    # Adding YEAR to groups
    xGrpBy <- c('YEAR', xGrpBy)

    # Sum variable(s) up to strata then estimation unit level
    eu.sums <- sumToEU(db,
                       x = xPlt,
                       y = NULL,
                       pops = pops,
                       x.grpBy = xGrpBy,
                       y.grpBy = NULL,
                       method = method, 
                       lambda = lambda)
    xEst <- eu.sums$x
  }

  # List of names for each output variable


  # Combine most-recent population estimates across states with potentially
  # different reporting schedules, i.e., if 2016 is most recent in MI and 2017 is
  # most recent in WI, combine them and label as 2017
  if (mr) {
    xEst <- combineMR(xEst)
    if (!is.null(y)) yEst <- combineMR(yEst)
  }



  # Totals and ratios -------------------------------------------------------

  if (!is.null(y)) {

    # For variable scoping w/ dplyr
    xGrpSyms <- dplyr::syms(xGrpBy)
    yGrpSyms <- dplyr::syms(yGrpBy)
    xTotalSyms <- dplyr::syms(names(xEst)[stringr::str_sub(names(xEst), -5, -1) == '_mean'])
    xVarSyms <- dplyr::syms(names(xEst)[stringr::str_sub(names(xEst), -4, -1) == '_var'])
    xCovSyms <- dplyr::syms(names(xEst)[stringr::str_sub(names(xEst), -3, -1) == '_cv'])
    yTotalSyms <- dplyr::sym(names(yEst)[stringr::str_sub(names(yEst), -5, -1) == '_mean'])
    yVarSyms <- dplyr::sym(names(yEst)[stringr::str_sub(names(yEst), -4, -1) == '_var'])
    ratioSyms <- dplyr::syms(stringr::str_c(names(xEst)[stringr::str_sub(names(xEst), -5, -1) == '_mean'],  '_RATIO'))
    ratioVarSyms <- dplyr::syms(stringr::str_c(names(xEst)[stringr::str_sub(names(xEst), -5, -1) == '_mean'],  '_RATIO_VAR'))


    # Sum over estimation units
    xEst <- xEst %>%
      dplyr::select(-c(ESTN_UNIT_CN, AREA_USED)) %>%
      dplyr::group_by(!!!xGrpSyms) %>%
      dplyr::summarize(dplyr::across(dplyr::everything(), sum, na.rm = TRUE))

    yEst <- yEst %>%
      dplyr::select(-c(ESTN_UNIT_CN, AREA_USED, P2PNTCNT_EU)) %>%
      dplyr::group_by( !!!yGrpSyms) %>%
      dplyr::summarize(dplyr::across(dplyr::everything(), sum, na.rm = TRUE))


    # Join numerator/denominator, compute ratios
    out <- left_join(xEst, yEst, by = yGrpBy) %>%
      # Compute ratio point estimates
      dplyr::mutate(dplyr::across(c(!!!xTotalSyms),
                                  .fns = ~ .x / !!yTotalSyms,
                                  .names = "{.col}_RATIO")) %>%
      # Now ratio variances
      dplyr::mutate(dplyr::across(c(!!!xTotalSyms),
                                  .fns = ~ (1 / ((!!yTotalSyms)^2)) * (get(stringr::str_c(stringr::str_sub(dplyr::cur_column(), 1, -6), '_var')) + ((.x/!!yTotalSyms)^2 * !!yVarSyms) - (2 * (.x/!!yTotalSyms) * get(stringr::str_c(stringr::str_sub(dplyr::cur_column(), 1, -6), '_cv'))) ),
                                  .names = "{.col}_RATIO_VAR")) %>%
      # When we only have one non-zero plot, we'll sometimes get extremely
      # small negative values from rounding errors. Make these zero
      dplyr::mutate(dplyr::across(c(!!!ratioVarSyms),
                                  .fns = ~ case_when(.x < 0 ~ 0,
                                                   TRUE ~ .x))) %>%
      # Sampling error for all variables
      dplyr::mutate(dplyr::across(c(!!!xTotalSyms),
                                  .fns = ~ sqrt(get(stringr::str_c(stringr::str_sub(dplyr::cur_column(), 1, -6), '_var'))) / abs(.x) * 100,
                                  .names = "{.col}_SE")) %>%
      dplyr::mutate(dplyr::across(c(!!!ratioSyms),
                                  .fns = ~ sqrt(get(stringr::str_c(dplyr::cur_column(), '_VAR'))) / abs(.x) * 100,
                                  .names = "{.col}_SE"))

    # Format output logically
    out <- formatNames(out, xGrpBy)

  } else {

    xGrpSyms <- dplyr::syms(xGrpBy)
    out <- xEst %>%
      dplyr::select(-c(ESTN_UNIT_CN, AREA_USED)) %>%
      dplyr::group_by(!!!xGrpSyms) %>%
      dplyr::summarize(dplyr::across(dplyr::everything(), sum, na.rm = TRUE))
    out <- formatNames(out, xGrpBy)

  }

  # Drop totals unless told not to
  if (!totals) {
    out <- out[,!stringr::str_detect(names(out), '_TOTAL')]
  }

  # Remove variance columns to avoid confusion
  tEst <- tEst[,!stringr::str_detect(names(tEst), '_VAR')]

  # Pretty output
  out <- out %>%
    dplyr::ungroup() %>%
    dplyr::mutate_if(is.factor, as.character) %>%
    tidyr::drop_na(!!!xGrpSyms) %>%
    dplyr::arrange(YEAR) %>%
    as_tibble()

  return(out)
}



formatNames <- function(x, grpBy) {


  # x is the dataframe that we want to format the names of, linked to sumToEU
  nms <- names(x)

  # totals
  total.slots <- stringr::str_sub(nms, -5, -1) == '_mean'
  names(x)[total.slots] <-  stringr::str_c(stringr::str_sub(nms[total.slots], 1, -6), '_TOTAL')
  total.syms <- dplyr::syms(names(x)[total.slots])

  # total variances
  total.var.slots <- stringr::str_sub(nms, -4, -1) == '_var'
  names(x)[total.var.slots] <- stringr::str_c(stringr::str_sub(nms[total.var.slots], 1, -5), '_TOTAL_VAR')
  total.var.syms <- dplyr::syms(names(x)[total.var.slots])

  # total sampling errors
  total.se.slots <- stringr::str_sub(nms, -8, -1) == '_mean_SE'
  names(x)[total.se.slots] <- stringr::str_remove(nms[total.se.slots], '_mean')
  total.se.syms <- dplyr::syms(names(x)[total.se.slots])

  # ratios
  ratio.slots <- stringr::str_sub(nms, -11, -1) == '_mean_RATIO'
  names(x)[ratio.slots] <- stringr::str_c(stringr::str_sub(nms[ratio.slots], 1, -12), '_RATIO')
  ratio.syms <- dplyr::syms(names(x)[ratio.slots])

  # ratio variances
  ratio.var.slots <- stringr::str_sub(nms, -15, -1) == '_mean_RATIO_VAR'
  names(x)[ratio.var.slots] <- stringr::str_c(stringr::str_sub(nms[ratio.var.slots], 1, -16), '_RATIO_VAR')
  ratio.var.syms <- dplyr::syms(names(x)[ratio.var.slots])

  # ratio sampling errors
  ratio.se.slots <- stringr::str_sub(nms, -14, -1) == '_mean_RATIO_SE'
  names(x)[ratio.se.slots] <- stringr::str_remove(nms[ratio.se.slots], '_mean')
  ratio.se.syms <- dplyr::syms(names(x)[ratio.se.slots])


  # Design information
  names(x)[names(x) == 'P2PNTCNT_EU'] <- 'N'
  names(x)[names(x) == 'nPlots.x'] <- 'nPlots_x'
  names(x)[names(x) == 'nPlots.y'] <- 'nPlots_y'


  # Select and order
  grpSyms <- dplyr::syms(grpBy)
  x <- x %>%
    dplyr::select(!!!grpSyms,
                  !!!ratio.syms,
                  !!!total.syms,
                  !!!ratio.se.syms,
                  !!!total.se.syms,
                  !!!ratio.var.syms,
                  !!!total.var.syms,
                  dplyr::any_of(c('nPlots_x', 'nPlots_y')))

  return(x)
}

