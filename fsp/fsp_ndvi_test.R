# FSP NDVI Analysis Test Script
# This script tests NDVI calculations with a smaller dataset
# Using average reflectance between 630-680 nm for red and 775-825 nm for NIR
# Also tests spectral ratio between 995-1005 nm and 495-505 nm ranges

# Load required packages
library(neonUtilities)
library(dplyr)
library(tidyr)
library(ggplot2)

# Test with a smaller date range first
startDate_checked <- "2022-01"
endDate_checked <- "2022-12"

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

# Create sample spectral data for testing (if no real data available)
create_sample_spectral_data <- function() {
  # Generate wavelength range from 400-1000 nm
  wavelength <- seq(400, 1000, by = 1)
  
  # Create realistic reflectance values
  # Healthy vegetation typically has low red reflectance and high NIR reflectance
  red_center <- 650  # Red wavelength center
  nir_center <- 800  # NIR wavelength center
  
  # Base reflectance with vegetation-like characteristics
  reflectance <- 0.1 + 0.3 * exp(-((wavelength - nir_center)^2) / (2 * 100^2)) - 
                 0.2 * exp(-((wavelength - red_center)^2) / (2 * 50^2))
  
  # Add some noise
  reflectance <- reflectance + rnorm(length(wavelength), 0, 0.02)
  
  # Ensure values are between 0 and 1
  reflectance <- pmax(0, pmin(1, reflectance))
  
  return(data.frame(
    wavelength = wavelength,
    reflectance = reflectance,
    stringsAsFactors = FALSE
  ))
}

# Test NDVI and spectral ratio calculations with sample data
cat("Testing NDVI and spectral ratio calculations with sample data...\n")
sample_data <- create_sample_spectral_data()

# Calculate NDVI for sample data
ndvi_test <- calculate_ndvi(sample_data)

# Calculate spectral ratio for sample data
ratio_test <- calculate_spectral_ratio(sample_data)

cat("Sample NDVI calculation results:\n")
cat("Red average (630-680 nm):", round(ndvi_test$red_avg, 4), "\n")
cat("NIR average (775-825 nm):", round(ndvi_test$nir_avg, 4), "\n")
cat("NDVI:", round(ndvi_test$ndvi, 4), "\n")
cat("Red bands count:", ndvi_test$red_bands_count, "\n")
cat("NIR bands count:", ndvi_test$nir_bands_count, "\n")

cat("\nSample spectral ratio calculation results:\n")
cat("495-505 nm average:", round(ratio_test$lower_avg, 4), "\n")
cat("995-1005 nm average:", round(ratio_test$higher_avg, 4), "\n")
cat("Spectral ratio (995-1005/495-505):", round(ratio_test$spectral_ratio, 4), "\n")
cat("Ratio valid (995-1005 > 495-505):", ratio_test$ratio_valid, "\n")
cat("495-505 bands count:", ratio_test$lower_bands_count, "\n")
cat("995-1005 bands count:", ratio_test$higher_bands_count, "\n")

# Plot sample spectral data with all bands highlighted
p_sample <- ggplot(sample_data, aes(x = wavelength, y = reflectance)) +
  geom_line(color = "darkgreen", size = 1) +
  # NDVI bands
  geom_vline(xintercept = c(630, 680), color = "red", linetype = "dashed", alpha = 0.7) +
  geom_vline(xintercept = c(775, 825), color = "blue", linetype = "dashed", alpha = 0.7) +
  # Spectral ratio bands
  geom_vline(xintercept = c(495, 505), color = "orange", linetype = "dotted", alpha = 0.7) +
  geom_vline(xintercept = c(995, 1005), color = "purple", linetype = "dotted", alpha = 0.7) +
  annotate("text", x = 655, y = 0.8, label = "Red Band\n(630-680 nm)", color = "red") +
  annotate("text", x = 800, y = 0.8, label = "NIR Band\n(775-825 nm)", color = "blue") +
  annotate("text", x = 500, y = 0.6, label = "495-505 nm", color = "orange") +
  annotate("text", x = 1000, y = 0.6, label = "995-1005 nm", color = "purple") +
  labs(title = "Sample Spectral Data with NDVI and Spectral Ratio Bands Highlighted",
       x = "Wavelength (nm)", y = "Reflectance") +
  theme_minimal()

ggsave("sample_spectral_data.png", p_sample, width = 10, height = 6)
cat("Sample spectral plot saved as 'sample_spectral_data.png'\n")

# Now try to load real FSP data
cat("\nAttempting to load real FSP data...\n")
spectra <- try(neonUtilities::loadByProduct(
  dpID='DP1.30012.001',
  check.size=F, 
  startdate = startDate_checked,
  enddate = endDate_checked,
  site = "all",
  include.provisional = T,
  package = "expanded",
  token = Sys.getenv('NEON_PAT'),
  silent = T))

# Check if data loaded successfully
if(inherits(spectra, "try-error")){
  cat("Error loading FSP data. Please check your NEON_PAT token.\n")
  cat("Running with sample data only.\n")
} else if(is.null(attr(spectra, "class"))){
  # Load data into environment
  invisible(list2env(spectra, envir=.GlobalEnv))
  cat("FSP data loaded successfully.\n")
  cat("Total spectral samples found:", nrow(fsp_spectralData), "\n")
  
  # Process first few samples as a test
  test_samples <- min(5, nrow(fsp_spectralData))
  cat("Processing first", test_samples, "samples as a test...\n")
  
  test_results <- data.frame(
    spectralSampleID = character(),
    siteID = character(),
    red_avg = numeric(),
    nir_avg = numeric(),
    ndvi = numeric(),
    lower_avg = numeric(),
    higher_avg = numeric(),
    spectral_ratio = numeric(),
    ratio_valid = logical(),
    processing_status = character(),
    stringsAsFactors = FALSE
  )
  
  for(i in 1:test_samples) {
    sample_id <- fsp_spectralData$spectralSampleID[i]
    site_id <- fsp_spectralData$siteID[i]
    download_url <- fsp_spectralData$downloadFileUrl[i]
    
    cat("Processing sample", i, "of", test_samples, ":", sample_id, "\n")
    
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
          test_results <- rbind(test_results, data.frame(
            spectralSampleID = sample_id,
            siteID = site_id,
            red_avg = ndvi_calc$red_avg,
            nir_avg = ndvi_calc$nir_avg,
            ndvi = ndvi_calc$ndvi,
            lower_avg = ratio_calc$lower_avg,
            higher_avg = ratio_calc$higher_avg,
            spectral_ratio = ratio_calc$spectral_ratio,
            ratio_valid = ratio_calc$ratio_valid,
            processing_status = "SUCCESS",
            stringsAsFactors = FALSE
          ))
          
          cat("  NDVI calculated:", round(ndvi_calc$ndvi, 4), "\n")
          cat("  Spectral ratio calculated:", round(ratio_calc$spectral_ratio, 4), "\n")
          cat("  Ratio valid:", ratio_calc$ratio_valid, "\n")
        } else {
          cat("  ERROR: Missing wavelength or reflectance columns\n")
          test_results <- rbind(test_results, data.frame(
            spectralSampleID = sample_id,
            siteID = site_id,
            red_avg = NA,
            nir_avg = NA,
            ndvi = NA,
            lower_avg = NA,
            higher_avg = NA,
            spectral_ratio = NA,
            ratio_valid = NA,
            processing_status = "ERROR: Missing columns",
            stringsAsFactors = FALSE
          ))
        }
      } else {
        cat("  ERROR: Download failed\n")
        test_results <- rbind(test_results, data.frame(
          spectralSampleID = sample_id,
          siteID = site_id,
          red_avg = NA,
          nir_avg = NA,
          ndvi = NA,
          lower_avg = NA,
          higher_avg = NA,
          spectral_ratio = NA,
          ratio_valid = NA,
          processing_status = "ERROR: Download failed",
          stringsAsFactors = FALSE
        ))
      }
      
    }, error = function(e) {
      cat("  ERROR:", e$message, "\n")
      test_results <- rbind(test_results, data.frame(
        spectralSampleID = sample_id,
        siteID = site_id,
        red_avg = NA,
        nir_avg = NA,
        ndvi = NA,
        lower_avg = NA,
        higher_avg = NA,
        spectral_ratio = NA,
        ratio_valid = NA,
        processing_status = paste("ERROR:", e$message),
        stringsAsFactors = FALSE
      ))
    })
    
    # Clean up temporary file
    if(file.exists(temp_file)) {
      unlink(temp_file)
    }
  }
  
  # Display test results
  cat("\n=== TEST RESULTS ===\n")
  print(test_results)
  
  # Save test results
  write.csv(test_results, "fsp_ndvi_test_results.csv", row.names = FALSE)
  cat("Test results saved to 'fsp_ndvi_test_results.csv'\n")
  
} else {
  cat("No FSP data available in this date range.\n")
  cat("Test completed with sample data only.\n")
}

cat("\n=== TEST COMPLETE ===\n")
cat("Check the generated files for results.\n") 