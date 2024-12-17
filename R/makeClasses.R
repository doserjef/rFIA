makeClasses <- function(x, interval = NULL, lower = NULL, upper = NULL, brks = NULL, numLabs = FALSE){
  if(is.null(brks)){
    # If min & max isn't specified, then use the data to compute
    low <- ifelse(is.null(lower), min(x, na.rm = TRUE), lower)
    up <- ifelse(is.null(upper), max(x, na.rm = TRUE), upper)
    # Compute class intervals
    brks = c(low)
    while (as.numeric(tail(brks,1)) < as.numeric(up)){
      brks <- c(brks, tail(brks,1) + interval)
    }
  } else {
    low = lower
  }

  # Apply classes to data
  classes <- cut(x, breaks = brks, include.lowest = TRUE, right = FALSE)

  # Convert to numeric (lowest value of interval)
  if (numLabs){
    classes <- as.numeric(classes) * interval -interval + low
  }

  return(classes)
}

