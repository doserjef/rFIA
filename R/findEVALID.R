# TODO: needs assessment
#### THIS MAY NEED WORK. NOT ALL EVALIDs follow the same coding scheme (ex, CT 2005 --> 95322)
# Look up EVALID codes
findEVALID <- function(db = NULL,
                       mostRecent = FALSE,
                       state = NULL,
                       year = NULL,
                       type = NULL){

  #### REWRITING FOR SIMPLICITY #####
  # Joing w/ evaltype code
  ids <- db$POP_EVAL %>%
    dplyr::left_join(dplyr::select(db$POP_EVAL_TYP, c('EVAL_GRP_CN', 'EVAL_TYP')), by = 'EVAL_GRP_CN') %>%
    dplyr::mutate(place = stringr::str_to_upper(LOCATION_NM))

  if (!is.null(state)){
    state <- stringr::str_to_upper(state)
    evalGrp <- intData$EVAL_GRP %>%
      dplyr::select(STATECD, STATE) %>%
      dplyr::mutate(STATECD = as.numeric(STATECD))
    ## Join state abbs with state codes in popeval
    ids <- dplyr::left_join(ids, evalGrp, by = 'STATECD')
    # Check if any specified are missing from db
    if (any(unique(state) %in% unique(evalGrp$STATE) == FALSE)){
      missStates <- state[state %in% unique(evalGrp$STATE) == FALSE]
      fancyName <- unique(intData$EVAL_GRP$STATE[intData$EVAL_GRP$STATECD %in% missStates])
      stop(paste('States: ', toString(fancyName) , 'not found in db.', sep = ''))
    }
    ids <- dplyr::filter(ids, STATE %in% state)
  }
  if (!is.null(year)){
    #year <- ifelse(str_length(year) == 2, year, str_sub(year, -2,-1))
    ids <- dplyr::filter(ids, END_INVYR %in% year)
  }
  if (!is.null(type)){
    ids <- dplyr::filter(ids, EVAL_TYP %in% paste0('EXP', type))
  }
  if (mostRecent) {

    ## Grouped filter wasn't working as intended, use filtering join
    maxYear <- ids %>%
      ## Remove TX, do it seperately
      dplyr::filter(!(STATECD %in% 48)) %>%
      dplyr::mutate(place = stringr::str_to_upper(LOCATION_NM)) %>%
      dplyr::group_by(place, EVAL_TYP) %>%
      dplyr::summarize(END_INVYR = max(END_INVYR, na.rm = TRUE),
                       LOCATION_NM = dplyr::first(LOCATION_NM))

    ## Texas coding standards are very bad
    ## Name two different inventory units with 5 different names
    ## Due to that, only use inventories for the ENTIRE state, sorry
    if (any(ids$STATECD %in% 48)){
      # evalType <- c('EXP_ALL', 'EXP_VOL', '')
      # evalCode <- c('00', '01', '03', '07', '09', '29')
      #
      # txIDS <- ids %>%
      #   filter(STATECD %in% 48) %>%
      #   # ## Removing any inventory that references east or west, sorry
      #   # filter(stringr::str_detect(stringr::str_to_upper(EVAL_DESCR), 'EAST', negate = TRUE) &
      #   #          stringr::str_detect(stringr::str_to_upper(EVAL_DESCR), 'WEST', negate = TRUE)) %>%
      #   mutate(typeCode = str_sub(str_trim(EVALID), -2, -1))
      #
      #   mutate(place = stringr::str_to_upper(LOCATION_NM)) %>%
      #   group_by(place, EVAL_TYP) %>%
      #   summarize(END_INVYR = max(END_INVYR, na.rm = TRUE),
      #             LOCATION_NM = first(LOCATION_NM))

      ## Will require manual updates, fix your shit texas
      txIDS <- ids %>%
        dplyr::filter(STATECD %in% 48) %>%
        dplyr::filter(END_INVYR < 2017) %>%
        dplyr::filter(END_INVYR > 2006) %>%
        ## Removing any inventory that references east or west, sorry
        dplyr::filter(stringr::str_detect(stringr::str_to_upper(EVAL_DESCR), 'EAST', negate = TRUE) &
                        stringr::str_detect(stringr::str_to_upper(EVAL_DESCR), 'WEST', negate = TRUE)) %>%
        dplyr::mutate(place = stringr::str_to_upper(LOCATION_NM)) %>%
        dplyr::group_by(place, EVAL_TYP) %>%
        dplyr::summarize(END_INVYR = max(END_INVYR, na.rm = TRUE),
                         LOCATION_NM = dplyr::first(LOCATION_NM))

      maxYear <- dplyr::bind_rows(maxYear, txIDS)
    }

    # ids <- ids %>%
    #   mutate(place = stringr::str_to_upper(LOCATION_NM)) %>%
    #   ### TEXAS IS REALLY ANNOYING LIKE THIS
    #   ### FOR NOW, ONLY THE ENTIRE STATE
    #   filter(place %in% c('TEXAS(EAST)', 'TEXAS(WEST)') == FALSE)


    ids <- dplyr::left_join(maxYear, dplyr::select(ids, c('place', 'EVAL_TYP', 'END_INVYR', 'EVALID')), by = c('place', 'EVAL_TYP', 'END_INVYR'))
  }

  # Output as vector
  ID <- unique(ids$EVALID)

  return(ID)
}

