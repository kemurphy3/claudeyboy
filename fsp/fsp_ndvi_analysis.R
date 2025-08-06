# FSP NDVI Analysis Script
# This script calculates NDVI ratios for all available spectral samples from 2019-2025
# Using average reflectance between 630-680 nm for red and 775-825 nm for NIR
# Also calculates spectral ratio between 995-1005 nm and 495-505 nm ranges

# Load required packages
library(neonUtilities)
library(dplyr)
library(tidyr)
library(ggplot2)
library(DT)

# Set date range for 2019-2025
startDate_checked <- "2019-01"
endDate_checked <- "2024-12"

# Check if FSP spectral data is already available in the environment with correct date range
if(exists("fsp_spectralData") && is.data.frame(fsp_spectralData) && nrow(fsp_spectralData) > 0) {
  # Check if data covers the desired date range
  if("collectDate" %in% names(fsp_spectralData)) {
    # Extract years from collectDate
    data_years <- as.numeric(substr(fsp_spectralData$collectDate, 1, 4))
    min_year <- min(data_years, na.rm = TRUE)
    max_year <- max(data_years, na.rm = TRUE)
    
    # Check if data covers the desired range (2019-2025)
    if(min_year <= substr(startDate_checked, 1, 4) && max_year >= substr(endDate_checked, 1, 4)) {
      cat("FSP spectral data already available in environment with correct date range (", min_year, "-", max_year, ").\n")
      cat("Total spectral samples found:", nrow(fsp_spectralData), "\n")
      spectra <- NULL  # Set spectra to NULL since data is already loaded
    } else {
      cat("FSP spectral data available but date range (", min_year, "-", max_year, ") doesn't match desired range (2019-2025).\n")
      cat("Loading fresh data for ",substr(startDate_checked, 1, 4), "-", substr(startDate_checked, 1, 4),"...\n")
      spectra <- try(neonUtilities::loadByProduct(
        dpID='DP1.30012.001',
        check.size=F, 
        startdate = startDate_checked,
        enddate = endDate_checked,
        site = "all",
        include.provisional = T,
        package = "expanded",  # Use expanded package to include dataQF field
        #release = "LATEST",
        token = Sys.getenv('NEON_PAT')), # Using NEON_PAT token
        silent = T)
    }
  } else {
    cat("FSP spectral data available but missing collectDate column. Loading fresh data...\n")
    spectra <- try(neonUtilities::loadByProduct(
      dpID='DP1.30012.001',
      check.size=F, 
      startdate = startDate_checked,
      enddate = endDate_checked,
      site = "all",
      include.provisional = T,
      package = "expanded",  # Use expanded package to include dataQF field
      #release = "LATEST",
      token = Sys.getenv('NEON_PAT')), # Using NEON_PAT token
      silent = T)
  }
} else {
  # Load FSP data for the entire date range
  cat("Loading FSP spectral data from 2019-2024...\n")
  spectra <- try(neonUtilities::loadByProduct(
    dpID='DP1.30012.001',
    check.size=F, 
    startdate = startDate_checked,
    enddate = endDate_checked,
    site = "all",
    include.provisional = T,
    package = "expanded",  # Use expanded package to include dataQF field
    #release = "LATEST",
    token = Sys.getenv('NEON_PAT')), # Using NEON_PAT token
    silent = T)
}

# Check if data loaded successfully or was already available
if(is.null(spectra)) {
  # Data was already available in environment
  cat("Using existing FSP spectral data.\n")
} else if(inherits(spectra, "try-error")){
  cat("Error loading FSP data. Please check your NEON_PAT token.\n")
  stop()
} else if(is.null(attr(spectra, "class"))){
  # Load data into environment
  invisible(list2env(spectra, envir=.GlobalEnv))
  cat("FSP data loaded successfully.\n")
  cat("Total spectral samples found:", nrow(fsp_spectralData), "\n")
} else {
  cat("No FSP data available in this date range.\n")
  stop()
}

# Function to calculate NDVI for a spectral sample
calculate_ndvi <- function(spectral_data) {
  # Extract red band (630-680 nm average)
  red_bands <- spectral_data$reflectance[spectral_data$wavelength >= 630 & spectral_data$wavelength <= 680]
  red_avg <- mean(red_bands, na.rm = TRUE)
  
  # Extract NIR band (775-825 nm average)  
  nir_bands <- spectral_data$reflectance[spectral_data$wavelength >= 775 & spectral_data$wavelength <= 825]
  nir_avg <- mean(nir_bands, na.rm = TRUE)
  
  # Calculate NDVI
  ndvi <- (nir_avg - red_avg) / (nir_avg + red_avg)
  
  return(list(
    red_avg = red_avg,
    nir_avg = nir_avg,
    ndvi = ndvi,
    red_bands_count = length(red_bands),
    nir_bands_count = length(nir_bands)
  ))
}

# Function to calculate spectral ratio (995-1005 nm vs 495-505 nm)
calculate_spectral_ratio <- function(spectral_data) {
  # Extract 495-505 nm range (lower wavelength)
  lower_bands <- spectral_data$reflectance[spectral_data$wavelength >= 495 & spectral_data$wavelength <= 505]
  lower_avg <- mean(lower_bands, na.rm = TRUE)
  
  # Extract 995-1005 nm range (higher wavelength)
  higher_bands <- spectral_data$reflectance[spectral_data$wavelength >= 995 & spectral_data$wavelength <= 1005]
  higher_avg <- mean(higher_bands, na.rm = TRUE)
  
  # Calculate ratio (higher should be greater than lower for healthy vegetation)
  spectral_ratio <- higher_avg / lower_avg
  
  # Check if ratio is valid (higher > lower)
  ratio_valid <- higher_avg > lower_avg
  
  return(list(
    lower_avg = lower_avg,
    higher_avg = higher_avg,
    spectral_ratio = spectral_ratio,
    ratio_valid = ratio_valid,
    lower_bands_count = length(lower_bands),
    higher_bands_count = length(higher_bands)
  ))
}

# Initialize results dataframe
ndvi_results <- data.frame(
  spectralSampleID = character(),
  siteID = character(),
  collectDate = character(),
  red_avg = numeric(),
  nir_avg = numeric(),
  ndvi = numeric(),
  red_bands_count = integer(),
  nir_bands_count = integer(),
  lower_avg = numeric(),
  higher_avg = numeric(),
  spectral_ratio = numeric(),
  ratio_valid = logical(),
  lower_bands_count = integer(),
  higher_bands_count = integer(),
  processing_status = character(),
  stringsAsFactors = FALSE
)

# Process each spectral sample
cat("Processing spectral samples for NDVI and spectral ratio calculations...\n")
total_samples <- nrow(fsp_spectralData)
processed_count <- 0
error_count <- 0

for(i in 1:total_samples) {
  sample_id <- fsp_spectralData$spectralSampleID[i]
  site_id <- fsp_spectralData$siteID[i]
  collect_date <- fsp_spectralData$collectDate[i]
  download_url <- fsp_spectralData$downloadFileUrl[i]
  
  # Progress indicator
  if(i %% 100 == 0) {
    cat("Processed", i, "of", total_samples, "samples...\n")
  }
  
  # Download and process spectral data
  temp_file <- tempfile(fileext = ".csv")
  tryCatch({
    # Download CSV file
    download_result <- download.file(download_url, temp_file, quiet = TRUE)
    
    if(download_result == 0 && file.exists(temp_file)) {
      # Read CSV data
      spectral_data <- read.csv(temp_file, stringsAsFactors = FALSE)
      
      # Check if required columns exist
      if(all(c("wavelength", "reflectance") %in% names(spectral_data))) {
        # Calculate NDVI
        ndvi_calc <- calculate_ndvi(spectral_data)
        
        # Calculate spectral ratio
        ratio_calc <- calculate_spectral_ratio(spectral_data)
        
        # Add to results
        ndvi_results <- rbind(ndvi_results, data.frame(
          spectralSampleID = sample_id,
          siteID = site_id,
          collectDate = collect_date,
          red_avg = ndvi_calc$red_avg,
          nir_avg = ndvi_calc$nir_avg,
          ndvi = ndvi_calc$ndvi,
          red_bands_count = ndvi_calc$red_bands_count,
          nir_bands_count = ndvi_calc$nir_bands_count,
          lower_avg = ratio_calc$lower_avg,
          higher_avg = ratio_calc$higher_avg,
          spectral_ratio = ratio_calc$spectral_ratio,
          ratio_valid = ratio_calc$ratio_valid,
          lower_bands_count = ratio_calc$lower_bands_count,
          higher_bands_count = ratio_calc$higher_bands_count,
          processing_status = "SUCCESS",
          stringsAsFactors = FALSE
        ))
        
        processed_count <- processed_count + 1
      } else {
        # Missing required columns
        ndvi_results <- rbind(ndvi_results, data.frame(
          spectralSampleID = sample_id,
          siteID = site_id,
          collectDate = collect_date,
          red_avg = NA,
          nir_avg = NA,
          ndvi = NA,
          red_bands_count = 0,
          nir_bands_count = 0,
          lower_avg = NA,
          higher_avg = NA,
          spectral_ratio = NA,
          ratio_valid = NA,
          lower_bands_count = 0,
          higher_bands_count = 0,
          processing_status = "ERROR: Missing wavelength or reflectance columns",
          stringsAsFactors = FALSE
        ))
        error_count <- error_count + 1
      }
    } else {
      # Download failed
      ndvi_results <- rbind(ndvi_results, data.frame(
        spectralSampleID = sample_id,
        siteID = site_id,
        collectDate = collect_date,
        red_avg = NA,
        nir_avg = NA,
        ndvi = NA,
        red_bands_count = 0,
        nir_bands_count = 0,
        lower_avg = NA,
        higher_avg = NA,
        spectral_ratio = NA,
        ratio_valid = NA,
        lower_bands_count = 0,
        higher_bands_count = 0,
        processing_status = "ERROR: Download failed",
        stringsAsFactors = FALSE
      ))
      error_count <- error_count + 1
    }
    
  }, error = function(e) {
    # Error processing file
    ndvi_results <- rbind(ndvi_results, data.frame(
      spectralSampleID = sample_id,
      siteID = site_id,
      collectDate = collect_date,
      red_avg = NA,
      nir_avg = NA,
      ndvi = NA,
      red_bands_count = 0,
      nir_bands_count = 0,
      lower_avg = NA,
      higher_avg = NA,
      spectral_ratio = NA,
      ratio_valid = NA,
      lower_bands_count = 0,
      higher_bands_count = 0,
      processing_status = paste("ERROR:", e$message),
      stringsAsFactors = FALSE
    ))
    error_count <- error_count + 1
  })
  
  # Clean up temporary file
  if(file.exists(temp_file)) {
    unlink(temp_file)
  }
}

# Summary statistics
cat("\n=== ANALYSIS SUMMARY ===\n")
cat("Total samples processed:", total_samples, "\n")
cat("Successfully processed:", processed_count, "\n")
cat("Errors encountered:", error_count, "\n")
cat("Success rate:", round(processed_count/total_samples * 100, 1), "%\n")

# Filter successful results
ndvi_successful <- ndvi_results %>% 
  filter(processing_status == "SUCCESS" & !is.na(ndvi))

cat("Valid NDVI calculations:", nrow(ndvi_successful), "\n")

if(nrow(ndvi_successful) > 0) {
  # NDVI statistics
  cat("\n=== NDVI STATISTICS ===\n")
  cat("Mean NDVI:", round(mean(ndvi_successful$ndvi, na.rm = TRUE), 4), "\n")
  cat("Median NDVI:", round(median(ndvi_successful$ndvi, na.rm = TRUE), 4), "\n")
  cat("Min NDVI:", round(min(ndvi_successful$ndvi, na.rm = TRUE), 4), "\n")
  cat("Max NDVI:", round(max(ndvi_successful$ndvi, na.rm = TRUE), 4), "\n")
  cat("SD NDVI:", round(sd(ndvi_successful$ndvi, na.rm = TRUE), 4), "\n")
  
  # Spectral ratio statistics
  cat("\n=== SPECTRAL RATIO STATISTICS ===\n")
  cat("Mean spectral ratio (995-1005/495-505):", round(mean(ndvi_successful$spectral_ratio, na.rm = TRUE), 4), "\n")
  cat("Median spectral ratio:", round(median(ndvi_successful$spectral_ratio, na.rm = TRUE), 4), "\n")
  cat("Min spectral ratio:", round(min(ndvi_successful$spectral_ratio, na.rm = TRUE), 4), "\n")
  cat("Max spectral ratio:", round(max(ndvi_successful$spectral_ratio, na.rm = TRUE), 4), "\n")
  cat("SD spectral ratio:", round(sd(ndvi_successful$spectral_ratio, na.rm = TRUE), 4), "\n")
  cat("Valid ratios (995-1005 > 495-505):", sum(ndvi_successful$ratio_valid, na.rm = TRUE), "of", nrow(ndvi_successful), "\n")
  cat("Valid ratio percentage:", round(sum(ndvi_successful$ratio_valid, na.rm = TRUE)/nrow(ndvi_successful) * 100, 1), "%\n")
  
  # NDVI by site
  cat("\n=== NDVI BY SITE ===\n")
  site_summary_ndvi <- ndvi_successful %>%
    group_by(siteID) %>%
    summarise(
      n_samples = n(),
      mean_ndvi = mean(ndvi, na.rm = TRUE),
      median_ndvi = median(ndvi, na.rm = TRUE),
      min_ndvi = min(ndvi, na.rm = TRUE),
      max_ndvi = max(ndvi, na.rm = TRUE),
      sd_ndvi = sd(ndvi, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(n_samples))
  
  print(site_summary_ndvi)
  
  # Spectral ratio by site
  cat("\n=== SPECTRAL RATIO BY SITE ===\n")
  site_summary_ratio <- ndvi_successful %>%
    group_by(siteID) %>%
    summarise(
      n_samples = n(),
      mean_ratio = mean(spectral_ratio, na.rm = TRUE),
      median_ratio = median(spectral_ratio, na.rm = TRUE),
      min_ratio = min(spectral_ratio, na.rm = TRUE),
      max_ratio = max(spectral_ratio, na.rm = TRUE),
      sd_ratio = sd(spectral_ratio, na.rm = TRUE),
      valid_ratios = sum(ratio_valid, na.rm = TRUE),
      valid_percentage = round(sum(ratio_valid, na.rm = TRUE)/n() * 100, 1),
      .groups = "drop"
    ) %>%
    arrange(desc(n_samples))
  
  print(site_summary_ratio)
  
  # NDVI by year
  cat("\n=== NDVI BY YEAR ===\n")
  ndvi_successful$year <- substr(ndvi_successful$collectDate, 1, 4)
  year_summary_ndvi <- ndvi_successful %>%
    group_by(year) %>%
    summarise(
      n_samples = n(),
      mean_ndvi = mean(ndvi, na.rm = TRUE),
      median_ndvi = median(ndvi, na.rm = TRUE),
      min_ndvi = min(ndvi, na.rm = TRUE),
      max_ndvi = max(ndvi, na.rm = TRUE),
      sd_ndvi = sd(ndvi, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(year)
  
  print(year_summary_ndvi)
  
  # Spectral ratio by year
  cat("\n=== SPECTRAL RATIO BY YEAR ===\n")
  year_summary_ratio <- ndvi_successful %>%
    group_by(year) %>%
    summarise(
      n_samples = n(),
      mean_ratio = mean(spectral_ratio, na.rm = TRUE),
      median_ratio = median(spectral_ratio, na.rm = TRUE),
      min_ratio = min(spectral_ratio, na.rm = TRUE),
      max_ratio = max(spectral_ratio, na.rm = TRUE),
      sd_ratio = sd(spectral_ratio, na.rm = TRUE),
      valid_ratios = sum(ratio_valid, na.rm = TRUE),
      valid_percentage = round(sum(ratio_valid, na.rm = TRUE)/n() * 100, 1),
      .groups = "drop"
    ) %>%
    arrange(year)
  
  print(year_summary_ratio)
  
  # Create visualizations
  cat("\n=== CREATING VISUALIZATIONS ===\n")
  
  # 1. NDVI distribution histogram
  p1 <- ggplot(ndvi_successful, aes(x = ndvi)) +
    geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
    labs(title = "Distribution of NDVI Values (2019-2025)",
         x = "NDVI", y = "Frequency") +
    theme_minimal()
  
  # 2. Spectral ratio distribution histogram
  p2 <- ggplot(ndvi_successful, aes(x = spectral_ratio)) +
    geom_histogram(bins = 50, fill = "orange", alpha = 0.7) +
    geom_vline(xintercept = 1, color = "red", linetype = "dashed", size = 1) +
    labs(title = "Distribution of Spectral Ratios (995-1005 nm / 495-505 nm)",
         x = "Spectral Ratio", y = "Frequency") +
    theme_minimal()
  
  # Add annotation after plot is created
  p2 <- p2 + annotate("text", x = 1.2, y = max(ggplot_build(p2)$data[[1]]$count) * 0.9, 
                      label = "Ratio = 1\n(995-1005 = 495-505)", color = "red")
  
  # 3. NDVI by site (boxplot)
  p3 <- ggplot(ndvi_successful, aes(x = reorder(siteID, ndvi, FUN = median), y = ndvi)) +
    geom_boxplot(fill = "lightgreen", alpha = 0.7) +
    labs(title = "NDVI Distribution by Site",
         x = "Site ID", y = "NDVI") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # 4. Spectral ratio by site (boxplot)
  p4 <- ggplot(ndvi_successful, aes(x = reorder(siteID, spectral_ratio, FUN = median), y = spectral_ratio)) +
    geom_boxplot(fill = "lightcoral", alpha = 0.7) +
    geom_hline(yintercept = 1, color = "red", linetype = "dashed", size = 1) +
    labs(title = "Spectral Ratio Distribution by Site",
         x = "Site ID", y = "Spectral Ratio (995-1005/495-505)") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # 5. NDVI by year (boxplot)
  p5 <- ggplot(ndvi_successful, aes(x = year, y = ndvi)) +
    geom_boxplot(fill = "orange", alpha = 0.7) +
    labs(title = "NDVI Distribution by Year",
         x = "Year", y = "NDVI") +
    theme_minimal()
  
  # 6. Spectral ratio by year (boxplot)
  p6 <- ggplot(ndvi_successful, aes(x = year, y = spectral_ratio)) +
    geom_boxplot(fill = "purple", alpha = 0.7) +
    geom_hline(yintercept = 1, color = "red", linetype = "dashed", size = 1) +
    labs(title = "Spectral Ratio Distribution by Year",
         x = "Year", y = "Spectral Ratio (995-1005/495-505)") +
    theme_minimal()
  
  # 7. Red vs NIR scatter plot
  p7 <- ggplot(ndvi_successful, aes(x = red_avg, y = nir_avg, color = ndvi)) +
    geom_point(alpha = 0.6) +
    scale_color_gradient2(low = "red", mid = "yellow", high = "green", 
                         midpoint = median(ndvi_successful$ndvi, na.rm = TRUE)) +
    labs(title = "Red vs NIR Reflectance with NDVI Color Coding",
         x = "Red Reflectance (630-680 nm avg)", 
         y = "NIR Reflectance (775-825 nm avg)",
         color = "NDVI") +
    theme_minimal()
  
  # 8. 495-505 vs 995-1005 scatter plot
  p8 <- ggplot(ndvi_successful, aes(x = lower_avg, y = higher_avg, color = spectral_ratio)) +
    geom_point(alpha = 0.6) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", size = 1) +
    scale_color_gradient2(low = "red", mid = "yellow", high = "green", 
                         midpoint = median(ndvi_successful$spectral_ratio, na.rm = TRUE)) +
    labs(title = "495-505 nm vs 995-1005 nm Reflectance with Ratio Color Coding",
         x = "495-505 nm Reflectance", 
         y = "995-1005 nm Reflectance",
         color = "Spectral Ratio") +
    theme_minimal()
  
  # Save plots
  ggsave("ndvi_distribution.png", p1, width = 10, height = 6)
  ggsave("spectral_ratio_distribution.png", p2, width = 10, height = 6)
  ggsave("ndvi_by_site.png", p3, width = 12, height = 6)
  ggsave("spectral_ratio_by_site.png", p4, width = 12, height = 6)
  ggsave("ndvi_by_year.png", p5, width = 10, height = 6)
  ggsave("spectral_ratio_by_year.png", p6, width = 10, height = 6)
  ggsave("red_vs_nir_scatter.png", p7, width = 10, height = 8)
  ggsave("spectral_ratio_scatter.png", p8, width = 10, height = 8)
  
  cat("Plots saved as PNG files.\n")
  
  # Save results to CSV
  write.csv(ndvi_results, "fsp_ndvi_results_all.csv", row.names = FALSE)
  write.csv(ndvi_successful, "fsp_ndvi_results_successful.csv", row.names = FALSE)
  write.csv(site_summary_ndvi, "fsp_ndvi_site_summary.csv", row.names = FALSE)
  write.csv(site_summary_ratio, "fsp_spectral_ratio_site_summary.csv", row.names = FALSE)
  write.csv(year_summary_ndvi, "fsp_ndvi_year_summary.csv", row.names = FALSE)
  write.csv(year_summary_ratio, "fsp_spectral_ratio_year_summary.csv", row.names = FALSE)
  
  cat("Results saved to CSV files.\n")
  
  # Display interactive table of successful results
  cat("\n=== INTERACTIVE TABLE OF RESULTS ===\n")
  cat("Use DT::datatable(ndvi_successful) to view the results interactively.\n")
  
} else {
  cat("No successful calculations to analyze.\n")
}

# Display error summary
if(error_count > 0) {
  cat("\n=== ERROR SUMMARY ===\n")
  error_summary <- ndvi_results %>%
    filter(processing_status != "SUCCESS") %>%
    group_by(processing_status) %>%
    summarise(count = n(), .groups = "drop") %>%
    arrange(desc(count))
  
  print(error_summary)
}

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Check the generated CSV files and PNG plots for detailed results.\n") 