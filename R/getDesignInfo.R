# Pull strata and estimation units weights for a given inventory
getDesignInfo <- function(db,
                          type = c('ALL','CURR','VOL','GROW','MORT',
                                   'REMV','CHNG','DWM','REGEN'),
                          mostRecent = TRUE,
                          evalid = NULL) {

  # Must have an FIA.Database or a remote one
  if (!c(class(db) %in% c('FIA.Database', 'Remote.FIA.Database'))) {
    stop('Must provide an FIA.Database or Remote.FIA.Database. See readFIA() and/or getFIA() to read and load your FIA data.')
  }

  # Type must exist
  if (all(!c(class(type) == 'character'))) {
    stop("'type' must be a character vector. Please choose one of (or a combination of): 'ALL', 'CURR', 'VOL', 'GROW', 'MORT', 'REMV', 'CHNG', 'DWM', 'REGEN'.")
  }
  type <- unique(stringr::str_to_upper(type))
  if (sum(type %in% c('ALL','CURR','VOL','GROW','MORT', 'REMV','CHNG','DWM','REGEN')) < length(type)) {
    bad.type = type[!c(type %in% c('ALL','CURR','VOL','GROW','MORT', 'REMV','CHNG','DWM','REGEN'))]
    stop(paste0("Don't recognize 'type': ", paste(as.character(bad.type), collapse = ', '), 
         ". Please choose one of (or a combination of): 'ALL', 'CURR', 'VOL', 'GROW', 'MORT', 'REMV', 'CHNG', 'DWM',' REGEN'."))
  }

  # Most recent must be logical
  if (!c(mostRecent %in% 0:1)) {
    stop("'mostRecent' must be logical, i.e., TRUE or FALSE.")
  }

  # Make the sure the necessary tables are present in db
  req.tables <- c('PLOT', 'POP_EVAL', 'POP_EVAL_TYP', 'POP_ESTN_UNIT', 'POP_STRATUM', 'POP_PLOT_STRATUM_ASSGN')
  if (is(db, 'FIA.Database')) {
    if (sum(req.tables %in% names(db)) < length(req.tables)) {
      missing.tables <- req.tables[!c(req.tables %in% names(db))]
      stop(paste(paste (as.character(missing.tables), collapse = ', '), 'tables not found in object db.'))
    }
  } else {
    # Read the tables we need, readFIA will throw a warning if they are missing
    db <- readFIA(dir = db$dir,
                  con = db$con,
                  schema = db$schema,
                  common = db$common,
                  tables =  c('PLOT', 'POP_EVAL', 'POP_EVAL_TYP', 
                              'POP_ESTN_UNIT', 'POP_STRATUM', 'POP_PLOT_STRATUM_ASSGN'),
                  states = db$states)
  }

  # Use clipFIA to handle the most recent subset if desired
  if (mostRecent) {
    db <- clipFIA(db)
  }

  # Fix TX problems with incomplete labeling of E v. W TX
  db <- handleTX(db)

  # WY and NM list early FHM inventories, but they don't work, so dropping
  if (any(c(35, 56) %in% unique(db$POP_EVAL$STATECD))) {
    db$POP_EVAL <- db$POP_EVAL %>%
      dplyr::mutate(cut.these = dplyr::case_when(STATECD %in% c(35, 56) & END_INVYR < 2001 ~ 1,
                                                 TRUE ~ 0)) %>%
      dplyr::filter(cut.these == 0) %>%
      dplyr::select(-c(cut.these))
  }

  # Pull together info for all evals listed in db
  evals <- db$POP_EVAL %>%
    # Slim it down
    dplyr::select(EVAL_CN = CN, STATECD, YEAR = END_INVYR, EVALID, ESTN_METHOD) %>%
    # Join eval type to get the actual evaluation type
    dplyr::left_join(dplyr::select(db$POP_EVAL_TYP, EVAL_CN, EVAL_TYP), by = 'EVAL_CN') %>%
    dplyr::filter(!is.na(EVAL_TYP))

  # If EVALID given, make sure it doesn't conflict w/ type
  if ( !is.null(evalid) ) {

    # Does the EVALID exist?
    if (!c(evalid %in% evals$EVALID)) {
      if (mostRecent) {
        stop(paste0('Specified `evalid` (', evalid, ') not found in `db`. Are you sure you want the most recent inventory (i.e., mostRecent = TRUE)?'))
      } else {
        stop(paste0('Specified `evalid` (', evalid, ') not found in `db`.'))
      }
    }

    # Subset evals
    evals <- dplyr::filter(evals, EVALID %in% evalid)
    implied.type <- stringr::str_sub(evals$EVAL_TYP, 4, -1)
    # Make sure the supplied evalids are in agreement with the specified evaluation type.
    if (!c(implied.type %in% type)) {
      stop(paste0('Specified `evalid` (', evalid, ') implies `type` ', implied.type, ', which conflicts with specified `type`: ', paste(as.character(type), collapse = ', '), '.' ))
    }

    # If EVALID not given, then subset by type. EVALID does this automatically
  } else {

    # Check that the type is available for all states
    states <- unique(evals$STATECD)
    for (i in states) {
      check.states <- evals %>%
        dplyr::filter(STATECD == i) %>%
        dplyr::left_join(intData$stateNames, by = 'STATECD')
      state.types <- stringr::str_sub(unique(check.states$EVAL_TYP), 4, -1)

      if (sum(type %in% state.types) < length(type) & length(type) < 9) {
        bad.type = type[!c(type %in% state.types)]
        warning(paste0(check.states$STATEAB[1], " doesn't include `type`(s): ", paste(as.character(bad.type), collapse = ', '), "."))
      }
    }

    # Subset evals
    evals <- dplyr::filter(evals, EVAL_TYP %in% paste0('EXP', type))

  }

  # Get remaining design info
  strata <- evals %>%
    # Drop all periodic inventories
    dplyr::filter(YEAR >= 2003) %>%
    # Join estimation unit
    dplyr::left_join(dplyr::select(db$POP_ESTN_UNIT, ESTN_UNIT_CN = CN,
                                   P1PNTCNT_EU, AREA_USED, EVAL_CN), by = 'EVAL_CN') %>%
    # Join stratum
    dplyr::left_join(dplyr::select(db$POP_STRATUM, ESTN_UNIT_CN,
                                   STRATUM_CN = CN, P1POINTCNT,
                                   P2POINTCNT, ADJ_FACTOR_MICR,
                                   ADJ_FACTOR_SUBP, ADJ_FACTOR_MACR),
                     by = c('ESTN_UNIT_CN')) %>%
    # Proportionate size of strata w/in estimation units
    dplyr::mutate(STRATUM_WGT = P1POINTCNT / P1PNTCNT_EU) %>%
    # Join plots to stratum
    dplyr::left_join(dplyr::select(db$POP_PLOT_STRATUM_ASSGN, PLT_CN, STRATUM_CN,
                                   UNITCD, COUNTYCD, PLOT), by = 'STRATUM_CN') %>%
    dplyr::mutate(pltID = stringr::str_c(UNITCD, STATECD, COUNTYCD, PLOT, sep = "_")) %>%
    dplyr::select(STATECD, YEAR, EVAL_TYP, EVALID, EVAL_TYP, ESTN_METHOD,
                  ESTN_UNIT_CN, AREA_USED,
                  STRATUM_CN, P2POINTCNT:ADJ_FACTOR_MACR, STRATUM_WGT,
                  pltID, PLT_CN) %>%
    dplyr::distinct()


  # If a CHNG inventory, then add GROWTH_ACCT
  if (any(paste0('EXP', type) %in% c('EXPMORT', 'EXPREMV', 'EXPGROW'))) {
    strata <- strata %>%
      dplyr::left_join(dplyr::select(db$POP_EVAL, EVALID, GROWTH_ACCT), by = 'EVALID') %>%
      dplyr::relocate(GROWTH_ACCT, .after = EVALID)
  }

  # Sum up number of plots per estimation unit
  p2eu <- strata %>%
    dplyr::distinct(ESTN_UNIT_CN, STRATUM_CN, P2POINTCNT) %>%
    dplyr::group_by(ESTN_UNIT_CN) %>%
    dplyr::summarise(P2PNTCNT_EU = sum(P2POINTCNT, na.rm = TRUE)) %>%
    dplyr::ungroup()

  # Add to original table
  strata <- strata %>%
    dplyr::left_join(p2eu, by = 'ESTN_UNIT_CN') %>%
    dplyr::relocate(P2PNTCNT_EU, .after = ESTN_UNIT_CN) %>%
    as_tibble()

  return(strata)

}
