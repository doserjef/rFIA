# FIA.Database Class ------------------------------------------------------
summary.FIA.Database <- function(object, ...){
  cat('---- FIA Database Object -----', '\n')
  # Years available
  if (!is.null(object$POP_EVAL$END_INVYR)){
    cat('Reporting Years: ',
        unique(object$POP_EVAL$END_INVYR[order(object$POP_EVAL$END_INVYR)]), '\n')
  }
  # States Covered
  if (!is.null(object$PLOT$STATECD)){
    states <- unique(ifelse(stringr::str_length(object$PLOT$STATECD) < 2, paste(0, object$PLOT$STATECD, sep = ''), object$PLOT$STATECD))
    cat('States:          ',
        as.character(unique(intData$EVAL_GRP$STATE[intData$EVAL_GRP$STATECD %in% states])), '\n')
  }
  # Number of Plots
  if (!is.null(object$POP_STRATUM)){
    eval <- rFIA::findEVALID(object, mostRecent = TRUE, type = 'CURR')
    nPlots <- object$POP_STRATUM %>%
      dplyr::filter(EVALID %in% eval) %>%
      dplyr::group_by(ESTN_UNIT_CN, CN) %>%
      dplyr::summarise(n = first(P2POINTCNT)) %>%
      dplyr::summarise(n = sum(n))
    cat('Total Plots:     ', sum(nPlots$n), '\n')
  }

  ## Memory Allocated
  mem <- object.size(object)
  cat('Memory Used:     ', format(mem, units = 'MB'), '\n')

  ## Tables included

  cat('Tables:          ', names(object), '\n')

}

print.FIA.Database <- function(x, ...){
  cat('---- FIA Database Object -----', '\n')
  # Years available
  if (!is.null(x$POP_EVAL$END_INVYR)){
    cat('Reporting Years: ',
        unique(x$POP_EVAL$END_INVYR[order(x$POP_EVAL$END_INVYR)]), '\n')
  }
  # States Covered
  if (!is.null(x$PLOT$STATECD)){
    states <- unique(ifelse(stringr::str_length(x$PLOT$STATECD) < 2, paste(0, x$PLOT$STATECD, sep = ''), x$PLOT$STATECD))
    cat('States:          ',
        as.character(unique(intData$EVAL_GRP$STATE[intData$EVAL_GRP$STATECD %in% states])), '\n')
  }
  # Number of Plots
  if (!is.null(x$POP_STRATUM)){
    eval <- rFIA::findEVALID(x, mostRecent = TRUE, type = 'CURR')
    nPlots <- x$POP_STRATUM %>%
      dplyr::filter(EVALID %in% eval) %>%
      dplyr::group_by(ESTN_UNIT_CN, CN) %>%
      dplyr::summarise(n = dplyr::first(P2POINTCNT)) %>%
      dplyr::summarise(n = sum(n))
    cat('Total Plots:     ', sum(nPlots$n), '\n')
  }

  ## Memory Allocated
  mem <- object.size(x)
  cat('Memory Used:     ', format(mem, units = 'MB'), '\n')

  ## Tables included
  cat('Tables:          ', names(x), '\n', '\n')

  if (length(x) > 1){
    print(sapply(x, as_tibble))
  } else {
    print(as_tibble(x[1]))
  }
  cat('\n')
}

str.FIA.Database <- function(object, ...) {
  cat(paste('FIA.Database', "\n"))
}

summary.Remote.FIA.Database <- function(object, ...){
  cat('---- Remote FIA Database Object -----', '\n')

  # States Covered
  if (!is.null(object$states)){
    cat('States:          ',
        as.character(object$states), '\n')
  }

  ## Memory Allocated
  mem <- object.size(object)
  cat('Memory Used:     ', format(mem, units = 'MB'), '\n')


}

print.Remote.FIA.Database <- function(x, ...){
  cat('---- Remote FIA Database Object -----', '\n')

  # States Covered
  if (!is.null(x$states)){
    cat('States:          ',
        as.character(x$states), '\n')
  }

  ## Memory Allocated
  mem <- object.size(x)
  cat('Memory Used:     ', format(mem, units = 'MB'), '\n')

}

str.Remote.FIA.Database <- function(object, ...) {
  cat(paste('Remote.FIA.Database', "\n"))
}

