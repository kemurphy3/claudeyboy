# Script to check primary keys in variables_30012 for FSP tables
# This will help us understand what removeDups() will actually use

# Load required packages
library(neonUtilities)

# Get the variables file for FSP
cat("Loading variables file for DP1.30012.001 (FSP)...\n")
variables_30012 <- neonUtilities::getVariables(dpID = "DP1.30012.001")

# Check primary keys for each FSP table
fsp_tables <- c("fsp_boutMetadata", "fsp_sampleMetadata", "fsp_spectralData")

cat("\n=== PRIMARY KEY ANALYSIS FOR FSP TABLES ===\n\n")

for (table_name in fsp_tables) {
  cat("Table:", table_name, "\n")
  
  # Get variables for this table
  table_vars <- variables_30012[variables_30012$table == table_name, ]
  
  # Check primary keys
  primary_keys <- table_vars[table_vars$primaryKey == "Y", ]
  
  if (nrow(primary_keys) > 0) {
    cat("  Primary keys found:\n")
    for (i in 1:nrow(primary_keys)) {
      cat("    -", primary_keys$fieldName[i], "\n")
    }
  } else {
    cat("  ❌ NO PRIMARY KEYS FOUND!\n")
  }
  
  # Show all fields for this table
  cat("  All fields in table:\n")
  for (i in 1:nrow(table_vars)) {
    pk_marker <- ifelse(table_vars$primaryKey[i] == "Y", " [PK]", "")
    cat("    -", table_vars$fieldName[i], pk_marker, "\n")
  }
  
  cat("\n")
}

# Summary
cat("=== SUMMARY ===\n")
cat("This analysis shows what primary keys removeDups() will actually use.\n")
cat("If the primary keys don't match what we expect, removeDups() may not work correctly.\n") 