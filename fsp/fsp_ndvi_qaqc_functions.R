# FSP NDVI and Spectral Ratio QAQC Functions
# These functions provide site-specific and NLCD-based quality checks

library(dplyr)
library(tidyr)

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

# Function to get NLCD-based expected ranges
get_nlcd_thresholds <- function(nlcd_class) {
  # Define expected NDVI and spectral ratio ranges by NLCD class
  # These are approximate ranges based on typical vegetation characteristics
  
  thresholds <- list()
  
  # Forest classes
  if (grepl("Deciduous Forest", nlcd_class, ignore.case = TRUE)) {
    thresholds$ndvi_min <- 0.3
    thresholds$ndvi_max <- 0.9
    thresholds$spectral_ratio_min <- 0.8
    thresholds$spectral_ratio_max <- 2.5
    thresholds$expected_ratio_valid <- TRUE
  } else if (grepl("Evergreen Forest", nlcd_class, ignore.case = TRUE)) {
    thresholds$ndvi_min <- 0.4
    thresholds$ndvi_max <- 0.9
    thresholds$spectral_ratio_min <- 0.9
    thresholds$spectral_ratio_max <- 2.8
    thresholds$expected_ratio_valid <- TRUE
  } else if (grepl("Mixed Forest", nlcd_class, ignore.case = TRUE)) {
    thresholds$ndvi_min <- 0.35
    thresholds$ndvi_max <- 0.9
    thresholds$spectral_ratio_min <- 0.85
    thresholds$spectral_ratio_max <- 2.6
    thresholds$expected_ratio_valid <- TRUE
  }
  # Shrub/Scrub classes
  else if (grepl("Shrub/Scrub", nlcd_class, ignore.case = TRUE)) {
    thresholds$ndvi_min <- 0.1
    thresholds$ndvi_max <- 0.7
    thresholds$spectral_ratio_min <- 0.6
    thresholds$spectral_ratio_max <- 2.0
    thresholds$expected_ratio_valid <- TRUE
  }
  # Herbaceous classes
  else if (grepl("Herbaceous", nlcd_class, ignore.case = TRUE) || 
           grepl("Grassland", nlcd_class, ignore.case = TRUE)) {
    thresholds$ndvi_min <- 0.1
    thresholds$ndvi_max <- 0.8
    thresholds$spectral_ratio_min <- 0.7
    thresholds$spectral_ratio_max <- 2.2
    thresholds$expected_ratio_valid <- TRUE
  }
  # Wetland classes
  else if (grepl("Wetland", nlcd_class, ignore.case = TRUE)) {
    thresholds$ndvi_min <- 0.05
    thresholds$ndvi_max <- 0.8
    thresholds$spectral_ratio_min <- 0.5
    thresholds$spectral_ratio_max <- 2.5
    thresholds$expected_ratio_valid <- TRUE
  }
  # Developed/Urban classes
  else if (grepl("Developed", nlcd_class, ignore.case = TRUE) || 
           grepl("Urban", nlcd_class, ignore.case = TRUE)) {
    thresholds$ndvi_min <- -0.1
    thresholds$ndvi_max <- 0.6
    thresholds$spectral_ratio_min <- 0.3
    thresholds$spectral_ratio_max <- 1.8
    thresholds$expected_ratio_valid <- FALSE  # Urban areas may not follow vegetation patterns
  }
  # Barren/Open Water
  else if (grepl("Barren", nlcd_class, ignore.case = TRUE) || 
           grepl("Open Water", nlcd_class, ignore.case = TRUE)) {
    thresholds$ndvi_min <- -0.2
    thresholds$ndvi_max <- 0.3
    thresholds$spectral_ratio_min <- 0.2
    thresholds$spectral_ratio_max <- 1.5
    thresholds$expected_ratio_valid <- FALSE
  }
  # Default for unknown classes
  else {
    thresholds$ndvi_min <- -0.2
    thresholds$ndvi_max <- 0.9
    thresholds$spectral_ratio_min <- 0.2
    thresholds$spectral_ratio_max <- 3.0
    thresholds$expected_ratio_valid <- TRUE
  }
  
  return(thresholds)
}

# Function to perform NDVI and spectral ratio QAQC checks
perform_spectral_qaqc <- function(fsp_spectralData, fsp_sampleMetadata, spectral_data_list) {
  # Join spectral and sample metadata
  metadata_joined <- fsp_spectralData %>%
    left_join(fsp_sampleMetadata, by = "spectralSampleID") %>%
    select(spectralSampleID, siteID, collectDate, nlcdClass, downloadFileUrl)
  
  # Initialize results
  qaqc_results <- data.frame(
    spectralSampleID = character(),
    siteID = character(),
    nlcdClass = character(),
    ndvi = numeric(),
    spectral_ratio = numeric(),
    ratio_valid = logical(),
    ndvi_check = character(),
    spectral_ratio_check = character(),
    ratio_validity_check = character(),
    overall_status = character(),
    stringsAsFactors = FALSE
  )
  
  # Process each spectral sample
  for(i in 1:nrow(metadata_joined)) {
    sample_id <- metadata_joined$spectralSampleID[i]
    site_id <- metadata_joined$siteID[i]
    nlcd_class <- metadata_joined$nlcdClass[i]
    
    # Get spectral data (assuming it's in the spectral_data_list)
    if(sample_id %in% names(spectral_data_list)) {
      spectral_data <- spectral_data_list[[sample_id]]
      
      # Calculate metrics
      ndvi_calc <- calculate_ndvi(spectral_data)
      ratio_calc <- calculate_spectral_ratio(spectral_data)
      
      # Get NLCD-based thresholds
      thresholds <- get_nlcd_thresholds(nlcd_class)
      
      # Perform checks
      ndvi_check <- ifelse(ndvi_calc$ndvi >= thresholds$ndvi_min & 
                          ndvi_calc$ndvi <= thresholds$ndvi_max, 
                          "PASS", "FAIL")
      
      spectral_ratio_check <- ifelse(ratio_calc$spectral_ratio >= thresholds$spectral_ratio_min & 
                                    ratio_calc$spectral_ratio <= thresholds$spectral_ratio_max, 
                                    "PASS", "FAIL")
      
      ratio_validity_check <- ifelse(thresholds$expected_ratio_valid == ratio_calc$ratio_valid, 
                                    "PASS", "FAIL")
      
      # Overall status
      overall_status <- ifelse(all(c(ndvi_check, spectral_ratio_check, ratio_validity_check) == "PASS"), 
                              "PASS", "FAIL")
      
      # Add to results
      qaqc_results <- rbind(qaqc_results, data.frame(
        spectralSampleID = sample_id,
        siteID = site_id,
        nlcdClass = nlcd_class,
        ndvi = ndvi_calc$ndvi,
        spectral_ratio = ratio_calc$spectral_ratio,
        ratio_valid = ratio_calc$ratio_valid,
        ndvi_check = ndvi_check,
        spectral_ratio_check = spectral_ratio_check,
        ratio_validity_check = ratio_validity_check,
        overall_status = overall_status,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  return(qaqc_results)
}

# Function to generate site-specific thresholds from historical data
generate_site_thresholds <- function(historical_ndvi_data) {
  # Calculate site-specific thresholds based on historical data
  site_thresholds <- historical_ndvi_data %>%
    group_by(siteID, nlcdClass) %>%
    summarise(
      ndvi_mean = mean(ndvi, na.rm = TRUE),
      ndvi_sd = sd(ndvi, na.rm = TRUE),
      spectral_ratio_mean = mean(spectral_ratio, na.rm = TRUE),
      spectral_ratio_sd = sd(spectral_ratio, na.rm = TRUE),
      ndvi_min = max(ndvi_mean - 3*ndvi_sd, -0.2),  # 3-sigma rule with physical bounds
      ndvi_max = min(ndvi_mean + 3*ndvi_sd, 0.9),
      spectral_ratio_min = max(spectral_ratio_mean - 3*spectral_ratio_sd, 0.1),
      spectral_ratio_max = min(spectral_ratio_mean + 3*spectral_ratio_sd, 3.0),
      n_samples = n(),
      .groups = "drop"
    ) %>%
    filter(n_samples >= 10)  # Only use thresholds if we have sufficient data
  
  return(site_thresholds)
}

# Function to create summary report
create_spectral_qaqc_summary <- function(qaqc_results) {
  summary_stats <- list()
  
  # Overall statistics
  summary_stats$total_samples <- nrow(qaqc_results)
  summary_stats$passing_samples <- sum(qaqc_results$overall_status == "PASS")
  summary_stats$failing_samples <- sum(qaqc_results$overall_status == "FAIL")
  summary_stats$pass_rate <- round(summary_stats$passing_samples / summary_stats$total_samples * 100, 1)
  
  # Breakdown by check type
  summary_stats$ndvi_failures <- sum(qaqc_results$ndvi_check == "FAIL")
  summary_stats$spectral_ratio_failures <- sum(qaqc_results$spectral_ratio_check == "FAIL")
  summary_stats$ratio_validity_failures <- sum(qaqc_results$ratio_validity_check == "FAIL")
  
  # Breakdown by NLCD class
  nlcd_summary <- qaqc_results %>%
    group_by(nlcdClass) %>%
    summarise(
      n_samples = n(),
      pass_rate = round(sum(overall_status == "PASS") / n() * 100, 1),
      .groups = "drop"
    )
  
  summary_stats$nlcd_breakdown <- nlcd_summary
  
  # Breakdown by site
  site_summary <- qaqc_results %>%
    group_by(siteID) %>%
    summarise(
      n_samples = n(),
      pass_rate = round(sum(overall_status == "PASS") / n() * 100, 1),
      .groups = "drop"
    )
  
  summary_stats$site_breakdown <- site_summary
  
  return(summary_stats)
}

# Example usage function
example_spectral_qaqc_usage <- function() {
  cat("Example usage of spectral QAQC functions:\n\n")
  
  cat("1. Load your data:\n")
  cat("   # fsp_spectralData and fsp_sampleMetadata should already be loaded\n")
  cat("   # spectral_data_list should contain the actual spectral curves\n\n")
  
  cat("2. Run QAQC checks:\n")
  cat("   qaqc_results <- perform_spectral_qaqc(fsp_spectralData, fsp_sampleMetadata, spectral_data_list)\n\n")
  
  cat("3. Generate summary:\n")
  cat("   summary_report <- create_spectral_qaqc_summary(qaqc_results)\n\n")
  
  cat("4. View results:\n")
  cat("   print(summary_report)\n")
  cat("   View(qaqc_results)\n\n")
  
  cat("5. Filter failing samples:\n")
  cat("   failing_samples <- qaqc_results %>% filter(overall_status == 'FAIL')\n")
} 