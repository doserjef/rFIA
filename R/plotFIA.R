# TODO: needs assessment
plotFIA <- function(data, y = NULL, grp = NULL, x = NULL, animate = FALSE, facet = FALSE,
                    se = FALSE, n.max = NULL, plot.title = NULL, y.lab = NULL, x.lab = NULL,
                    legend.title = NULL, legend.labs = waiver(), limits = c(NA, NA),
                    color.option = 'viridis', line.color = "gray30", line.width =1,
                    min.year = 2005, direction = 1, alpha = .9, transform = "identity",
                    text.size = 1, text.font = '', lab.width = 1, legend.height = 1,
                    legend.width = 1, device = 'png', savePath = NULL, fileName = NULL) {


  ## Requires gganimate when animate = TRUE
  if (animate == TRUE & !c('gganimate' %in% row.names(installed.packages()))) {
    stop('plotFIA requires package "gganimate" to produce animations. Please install with "install.packages(gganimate)" and try again.')
  }


  # Need to quote all variables for NSE
  y_quo = enquo(y)
  x_quo = enquo(x)
  grp_quo = enquo(grp)

  ## Some dummy checks
  if (!is.null(savePath) & is.null(fileName)){
    warning('Must specify both fileName & savePath to save a plot.')
  }
  if (is.null(savePath) & !is.null(fileName)){
    warning('Must specify both fileName & savePath to save a plot.')
  }
  # if (class(y) != 'character'){
  #   stop(cat('y must be of class character. Specify name of variable in data you would like to visualize. \n'))
  # }
  if (animate & !is.null(savePath) & !is.null(fileName)){
    message('Saving as .gif file. \n')
  }
  ## IF data is not an FIA.Database, y is required
  if (any(class(data) %in% c('FIA.Database') == FALSE) & quo_name(y_quo) == "NULL"){
    ## If it's remote, just read in the plot table
    if (is(data, 'Remote.FIA.Database')) {
      data <- readFIA(data$dir, tables = 'PLOT')
    } else {
      stop('Argument "y" required unless plotting an FIA.Database object.')
    }
  } #else if (quo_name(y_quo) != "NULL") {
    #stop(cat('Argument "y" required unless plotting an FIA.Database object.'))
  #}

  ## Plot plot locations in a database
  if (any(class(data) == 'FIA.Database')){
    pltSF <- data$PLOT %>%
      drop_na(LON, LAT) %>%
      st_as_sf(coords = c('LON', 'LAT'))
    st_crs(pltSF) <- 4326
    # Make the spatial map
    map <- pltSF %>%
      ggplot() +
      #geom_point(colour = line.color, size = line.width) +
      geom_sf(aes(geometry = geometry), colour = line.color, size = line.width) +
      #labs(color = ifelse(is.null(legend.title), str_wrap(quo_name(y_quo), width = 10 * lab.width), str_wrap(legend.title, width = 10 * lab.width))) +
      #scale_colour_viridis_c(alpha = alpha, option = color.option, direction = direction, trans = transform) +
      #scale_fill_viridis_c(alpha = alpha, option = color.option, direction = direction, trans = transform) +
      theme_minimal() +
      ggtitle(plot.title)+
      theme(#axis.text = element_blank(),
        legend.title = element_text(size = 14 * text.size, face = 'bold.italic', family = text.font),
        legend.text = element_text(size = 11 * text.size, face = 'italic', family = text.font),
        plot.title = element_text(size = 17 * text.size, face = 'bold', family = text.font),
        legend.key.height = unit(2.2 * legend.height, "cm"),
        legend.key.width  = unit(1 * legend.width, "cm"))
    return(map)
  }




  ## If a modifier was given to a variable, handle it (ex. y = TPA_PERC / 100)
  data <- data %>%
    mutate(xVar = !!x_quo,
           grpVar = !!grp_quo,
           yVar = !!y_quo)


  ## If they want a subset of the groups
  if (!is.null(n.max) & quo_name(grp_quo) != 'NULL'){
    d <- data %>%
      group_by(grpVar) %>%
      summarize(m = mean(yVar, na.rm = TRUE)) %>%
      top_n(n.max, m)
    grpStr <- d[,1, drop = TRUE]

    data <- data %>%
      filter(grpVar %in% grpStr)
  }

  ###### SPATIAL & SPATIOTEMPORAL PLOTS ######
  if ('sf' %in% class(data)){
    ## Plotting spatial points
    if (any(str_detect(st_geometry_type(data), 'POINT'))){
      if (quo_name(grp_quo) == 'NULL'){
        # Make the spatial map
        map <- data %>%
          ggplot() +
          geom_sf(aes(colour = yVar)) +
          labs(colour = ifelse(is.null(legend.title), str_wrap(quo_name(y_quo), width = 10 * lab.width), str_wrap(legend.title, width = 10 * lab.width))) +
          scale_colour_viridis_c(alpha = alpha, option = color.option, direction = direction, trans = transform, limits = limits) +
          #scale_fill_viridis_c(alpha = alpha, option = color.option, direction = direction, trans = transform) +
          theme_minimal() +
          ggtitle(plot.title)+
          theme(#axis.text = element_blank(),
            legend.title = element_text(size = 14 * text.size, face = 'bold.italic', family = text.font),
            legend.text = element_text(size = 11 * text.size, face = 'italic', family = text.font),
            plot.title = element_text(size = 17 * text.size, face = 'bold', family = text.font),
            legend.key.height = unit(2.2 * legend.height, "cm"),
            legend.key.width  = unit(1 * legend.width, "cm"))

        ## Grouped spatial points
      } else {
        map <- data %>%
          ggplot() +
          geom_sf(aes(size = yVar, colour = grpVar), show.legend = "point") +
          labs(size = ifelse(is.null(legend.title[1]), str_wrap(quo_name(y_quo), width = 10 * lab.width), str_wrap(legend.title[1], width = 10 * lab.width)),
               colour = ifelse(is.null(legend.title[2]), str_wrap(quo_name(grp_quo), width = 10 * lab.width), str_wrap(legend.title[2], width = 10 * lab.width))) +
          scale_colour_viridis_d(alpha = alpha, option = color.option, direction = direction) +
          #scale_fill_viridis_c(alpha = alpha, option = color.option, direction = direction, trans = transform) +
          theme_minimal() +
          ggtitle(plot.title)+
          theme(#axis.text = element_blank(),
            legend.title = element_text(size = 14 * text.size, face = 'bold.italic', family = text.font),
            legend.text = element_text(size = 11 * text.size, face = 'italic', family = text.font),
            plot.title = element_text(size = 17 * text.size, face = 'bold', family = text.font),
            legend.key.height = unit(2.2 * legend.height, "cm"),
            legend.key.width  = unit(1 * legend.width, "cm"))
      }

      ## Plotting spatial polygons
    } else {
      # Make the spatial map
      map <- data %>%
        ggplot() +
        geom_sf(aes(fill = yVar), colour = line.color, lwd = line.width) +
        labs(fill = ifelse(is.null(legend.title), str_wrap(quo_name(y_quo), width = 10 * lab.width), str_wrap(legend.title, width = 10 * lab.width))) +
        scale_fill_viridis_c(alpha = alpha, option = color.option, direction = direction, trans = transform, limits = limits) +
        theme_minimal() +
        ggtitle(plot.title)+
        theme(#axis.text = element_blank(),
          legend.title = element_text(size = 14 * text.size, face = 'bold.italic', family = text.font),
          legend.text = element_text(size = 11 * text.size, face = 'italic', family = text.font),
          plot.title = element_text(size = 17 * text.size, face = 'bold', family = text.font),
          legend.key.height = unit(2.2 * legend.height, "cm"),
          legend.key.width  = unit(1 * legend.width, "cm"))
    }
    ## Animate if they want to
    if (animate){
      map <- map +
        transition_manual(YEAR) +
        labs(title = 'Year: {current_frame}')
    } else if(facet){
      map <- map + facet_wrap(~YEAR) +
        theme(strip.text = element_text(size = 12 * text.size, family = text.font),
              strip.background = element_rect(fill = 'gray'),
              panel.background = element_rect(color = 'black'))
    }

    ###### TIME SERIES PLOTS  (or UD x) ######
  } else {

    ## Identify the associated sampling error field
    if (se){
      #yv = data$yVar
      SE <- data
      for (i in 1:ncol(SE)){
        SE[,i] <- if_else(data[,i] == data$yVar, 1, 0)
      }

      SE <- SE %>%
        #mutate_all(.funs = ~if_elSE(yVar == ., 1, 0)) %>%
        #mutate_all(.funs = if_elSE, condition = yv == ., true =1, falSE = 0, missing = NULL) %>%
        select(-c(yVar)) %>%
        summarise_all(sum, na.rm = TRUE)

      yName <- names(data)[max.col(SE)]
      seName <- paste0(yName, '_SE')
      data$seVar <- (data[[seName]] / 100) * abs(data$yVar) * 1.96 # 95 % confidence
      data$ymin <- data$yVar - data$seVar
      data$ymax <- data$yVar + data$seVar
    }

    #print(data$seVar)

    ## Default to time series if x not specified
    if (quo_name(x_quo) == 'NULL'){
      data$xVar <- data$YEAR
      # Convert year to date format
      data$xVar <- as.Date(paste(data$YEAR, 1, 1, sep = "-"))
      # Make a new label for x-axis if not specified
      if (is.null(x.lab)) x.lab <- 'YEAR'
    } else if (animate == FALSE) {
      if (quo_name(grp_quo) != 'NULL'){
        data <- data %>%
          group_by(xVar, grpVar) %>%
          filter(YEAR == max(YEAR, na.rm = TRUE)) %>%
          ungroup()
      } else {
        data <- data %>%
          group_by(xVar) %>%
          filter(YEAR == max(YEAR, na.rm = TRUE)) %>%
          ungroup()
      }
    }

    if (se){
      ylim = if_else(any(data$ymin < 0), min(data$ymin,na.rm = TRUE) * 1.4, 0)
      #ylimMax = max(data$ymax,na.rm = TRUE) * 1.4
    } else {
      ylim = if_else(any(data$yVar < 0), min(data$yVar,na.rm = TRUE) * 1.4, 0)
    }

    # Simple time series
    if (quo_name(grp_quo) == 'NULL'){
      if (se){
        map <- data %>%
          select(yVar, xVar, ymin, ymax) %>%
          ungroup() %>%
          ggplot(aes(x = xVar, y = yVar, group = 1), group = 1) +
          geom_errorbar(aes(ymin=ymin, ymax=ymax), size = .67, alpha= .67, width = .5) +
          #geom_ribbon(aes(x = xVar, ymin = (yVar * ))) +
          geom_line(aes(group = 1), color = line.color, lwd = line.width) +
          #geom_line(color = line.color, lwd = line.width) +
          theme_bw() +
          ggtitle(plot.title) +
          xlab(ifelse(is.null(x.lab), quo_name(x_quo), x.lab)) +
          ylab(ifelse(is.null(y.lab), quo_name(y_quo), y.lab)) +
          #scale_y_continuous(limits = c(0, NA), expand = c(0,0))
          ylim(ylim, max(data$ymax) * 1.4) +
          theme(axis.text = element_text(size = 11 * text.size, family = text.font),
                axis.title = element_text(size = 15 * text.size, family = text.font),
                plot.title = element_text(size = 17 * text.size, face = 'bold', family = text.font))

      } else {
        map <- data %>%
          select(yVar, xVar) %>%
          ungroup() %>%
          ggplot(aes(x = xVar, y = yVar, group = 1), group = 1) +
          #geom_ribbon(aes(x = xVar, ymin = (yVar * ))) +
          geom_line(aes(group = 1), color = line.color, lwd = line.width) +
          #geom_line(color = line.color, lwd = line.width) +
          theme_bw() +
          ggtitle(plot.title) +
          xlab(ifelse(is.null(x.lab), quo_name(x_quo), x.lab)) +
          ylab(ifelse(is.null(y.lab), quo_name(y_quo), y.lab)) +
          #scale_y_continuous(limits = c(0, NA), expand = c(0,0))
          ylim(ylim, max(data$yVar) * 1.4) +
          theme(axis.text = element_text(size = 11 * text.size, family = text.font),
                axis.title = element_text(size = 15 * text.size, family = text.font),
                plot.title = element_text(size = 17 * text.size, face = 'bold', family = text.font))
      }


      # grouped time series
    } else {
      # Handle legend labels
      #if (is.null(legend.labs)) legend.labs = waiver()
      if (se){
        # Omit any NAs in grp
        data <- filter(data, !is.na(grpVar))
        map <- data %>%
          ggplot(aes(x = as.factor(xVar), y = yVar, colour = as.factor(grpVar), group = as.factor(grpVar))) +
          geom_line(lwd = line.width) +
          #labs(colour = ifelse(is.null(legend.title), str_wrap(y, width = 10 * lab.width), str_wrap(legend.title, width = 10 * lab.width))) +
          scale_colour_viridis_d(alpha = alpha, option = color.option, direction = direction, labels = legend.labs) +
          labs(colour = ifelse(is.null(legend.title), '', str_wrap(legend.title, width = 10 * lab.width))) +
          theme_bw() +
          ggtitle(plot.title) +
          xlab(ifelse(is.null(x.lab), quo_name(x_quo), x.lab)) +
          ylab(ifelse(is.null(y.lab), quo_name(y_quo), y.lab)) +
          ylim(ylim, max(data$ymax) * 1.4) +
          theme(axis.text = element_text(size = 11 * text.size, family = text.font),
                axis.title = element_text(size = 15 * text.size, family = text.font),
                plot.title = element_text(size = 17 * text.size, face = 'bold', family = text.font),
                legend.title = element_text(size = 15 * text.size, face = 'bold.italic', family = text.font),
                legend.text = element_text(size = 15 * text.size, face = 'italic', family = text.font)) +
          geom_errorbar(aes(ymin=ymin, ymax=ymax), size = .67, alpha= .67, width = .5)
      } else {
        # Omit any NAs in grp
        data <- filter(data, !is.na(grpVar))
        map <- data %>%
          ggplot(aes(x = as.factor(xVar), y = yVar, colour = as.factor(grpVar), group = as.factor(grpVar))) +
          geom_line(lwd = line.width) +
          #labs(colour = ifelse(is.null(legend.title), str_wrap(y, width = 10 * lab.width), str_wrap(legend.title, width = 10 * lab.width))) +
          scale_colour_viridis_d(alpha = alpha, option = color.option, direction = direction, labels = legend.labs) +
          labs(colour = ifelse(is.null(legend.title), '', str_wrap(legend.title, width = 10 * lab.width))) +
          theme_bw() +
          ggtitle(plot.title) +
          xlab(ifelse(is.null(x.lab), quo_name(x_quo), x.lab)) +
          ylab(ifelse(is.null(y.lab), quo_name(y_quo), y.lab)) +
          ylim(ylim, max(data$yVar) * 1.4) +
          theme(axis.text = element_text(size = 11 * text.size, family = text.font),
                axis.title = element_text(size = 15 * text.size, family = text.font),
                plot.title = element_text(size = 17 * text.size, face = 'bold', family = text.font),
                legend.title = element_text(size = 15 * text.size, face = 'bold.italic', family = text.font),
                legend.text = element_text(size = 15 * text.size, face = 'italic', family = text.font))
      }


    }


    ## IF you want to animate
    if (animate){
      # map <- map +
      #   transition_reveal(along = as.numeric(xVar))
      map <- map +
        transition_manual(YEAR) +
        labs(title = 'Year: {current_frame}')
    }
  }

  ## Save the plots if they want to
  if(!is.null(savePath) & !is.null(fileName)){
    if (animate){
      anim_save(filename = fileName, animation = map, path = savePath)
    } else {
      # Save the plot with the chosen device
      ggsave(filename = fileName, plot = map, device = device, path = savePath)
    }
  }


  return(map)
}




