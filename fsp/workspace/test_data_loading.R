# Test FSP data loading to diagnose the issue

# Load required library
library(neonUtilities)

# Test 1: Check if NEON_PAT is set
cat("NEON_PAT token available:", ifelse(Sys.getenv('NEON_PAT') != "", "YES", "NO"), "\n")

# Test 2: Try loading FSP data with error handling
cat("\nAttempting to load FSP data for 2022...\n")

tryCatch({
  # Test FSP data loading
  spectra <- neonUtilities::loadByProduct(
    dpID='DP1.30012.001',
    check.size=F, 
    startdate = "2022-01",
    enddate = "2022-12",
    site = "all",
    include.provisional = T,
    release = "LATEST",
    token = Sys.getenv('NEON_PAT'),
    silent = F  # Set to FALSE to see what's happening
  )
  
  # Check what we got
  cat("Data loading result:\n")
  cat("Class of result:", class(spectra), "\n")
  cat("Is it a list?", is.list(spectra), "\n")
  if(is.list(spectra)) {
    cat("Number of tables:", length(spectra), "\n")
    cat("Table names:", names(spectra), "\n")
  }
  
}, error = function(e) {
  cat("ERROR occurred:", e$message, "\n")
}, warning = function(w) {
  cat("WARNING occurred:", w$message, "\n")
})

# Test 3: Try without token
cat("\n\nTrying without NEON_PAT token...\n")
tryCatch({
  spectra_no_token <- neonUtilities::loadByProduct(
    dpID='DP1.30012.001',
    check.size=F, 
    startdate = "2022-01",
    enddate = "2022-12",
    site = "all",
    include.provisional = T,
    release = "LATEST",
    silent = F
  )
  
  cat("Data loading result (no token):\n")
  cat("Class of result:", class(spectra_no_token), "\n")
  cat("Is it a list?", is.list(spectra_no_token), "\n")
  if(is.list(spectra_no_token)) {
    cat("Number of tables:", length(spectra_no_token), "\n")
    cat("Table names:", names(spectra_no_token), "\n")
  }
  
}, error = function(e) {
  cat("ERROR occurred (no token):", e$message, "\n")
}, warning = function(w) {
  cat("WARNING occurred (no token):", w$message, "\n")
}) 