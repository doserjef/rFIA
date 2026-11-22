# TODO: in development and non-functional. 
carbonSAE <- function(polys, years, pool = "aboveground live", 
                      version, fun = "mean") {
  
  # Initial checks --------------------------------------------------------
  if (missing(polys)) {
    stop("polys (polygons representing the areas of interest for SAE) must be specified")
  }
  if (missing(years)) {
    stop("years for which carbon estimates are desired must be specified")
  }
  # Check if years is a valid year
  # NOTE: the currently valid years are hardcoded in and must be updated 
  #       anytime the underlying SAE models are run. 
  if (sum(!(years %in% 2001:2025)) > 0) {
    stop("currently valid prediction years are 2001-2025")
  }
  if (!(tolower(pool) %in% c("aboveground live", "belowground live", "dead wood", "litter", 
                    "soil organic"))) {
    stop("the specified pool is not valid. Supported pools are: 'aboveground live', 'belowground live', 'dead wood', 'litter', 'soil organic'")
  }
 
  # Set up GDAL config ----------------------------------------------------
  # This must happen before the COGs are opened with terra::rast(), otherwise 
  # GDAL probes the bucket for sidecar files (.aux.xml, .ovr, etc.), which 
  # produces extra HTTP requests and 403 warnings.

  # Get current GDAL config settings and store for later after function is done.
  old_config <- terra::getGDALconfig(c("GDAL_DISABLE_READDIR_ON_OPEN", 
                                       "CPL_VSIL_CURL_ALLOWED_EXTENSIONS"))
  
  # Set GDAL settings to behave well for remote COGs, and then restore on exit
  terra::setGDALconfig(c("GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR", 
                         "CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif"))
  on.exit(terra::setGDALconfig(names(old_config), old_config), add = TRUE)

  # Establish path and subroutine for referencing correct COGs. -----------  
  base_url <- "https://ncsu-sefs-lab-fia-carbon.s3.amazonaws.com/cog/"
  carbon_path <- function(pool, year, version) {
    # TODO: version is currently a no-op. 
    # TODO: will need to be updated over time. 
    paste0("/vsicurl/", base_url, "NY_S3_",  year, "_mean.tif")
  }
  # Reference the correct COGs to be partially read in later
  r <- terra::rast(vapply(years, function(y) carbon_path(pool, y, version), ""))
  # TODO: this will change depending on what actual values are extracted from the COG. 
  # TODO: hardcoded
  n_bands <- 3 
  n_years <- length(years)
  # Layer names are "<year>__<band>" so they can be split apart after extraction
  tmp <- paste(rep(years, each = n_bands), names(r), sep = "__")
  names(r) <- tmp
 
  # Prep polygons ---------------------------------------------------------

  # Project user-supplied polygons to the COG CRS.  
  # The if statement protects against problems that can occur depending on how
  # the polygons are input into the function, whether they are input as sf 
  # objects, or if they are input as terra objects. 
  if (any(class(polys) == "SpatVector")) {
    polys <- terra::project(polys, crs(r))
  } else {
    polys <- terra::project(terra::vect(polys), crs(r))
  }
 
  # Crop the raster to the current polygons. This will only fetch the tiles that 
  # overlap the given box. 
  r <- terra::crop(r, polys) 
  # Mask the raster
  r <- terra::mask(r, polys)
 
  # Extract the desired summary statistic within the polys of interest.  
  polys <- sf::st_as_sf(polys)
  out_small <- exactextractr::exact_extract(r, polys, fun = fun) 

  # Convert to long format: one row per polygon and year, with a column for 
  # each band (stage). Extracted column names are "<fun>.<year>__<band>". 
  out_small$polyID <- seq_len(nrow(out_small))
  out_small <- out_small %>%
    tidyr::pivot_longer(cols = -polyID, 
                        names_to = c("fun", "YEAR", "band"),
                        names_pattern = "^(.+?)\\.(\\d{4})__(.+)$") %>%
    dplyr::mutate(YEAR = as.integer(YEAR))
  # Only prefix band names with the summary function when there is more than one
  if (length(fun) > 1) {
    out_small <- dplyr::mutate(out_small, band = paste(fun, band, sep = "_"))
  }
  out_small <- out_small %>%
    dplyr::select(-fun) %>%
    tidyr::pivot_wider(names_from = band, values_from = value) %>%
    dplyr::arrange(polyID, YEAR)
  out <- cbind(polys[out_small$polyID, ], dplyr::select(out_small, -polyID))
  rownames(out) <- NULL
  
  return(out)
 }




