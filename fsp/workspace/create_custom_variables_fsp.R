# Script to create a custom variables file for FSP with only the desired primary keys
# This will allow removeDups() to work with only the primary keys we want

library(neonUtilities)

# Get the original variables file
cat("Loading original variables file for DP1.30012.001 (FSP)...\n")
variables_30012_original <- neonUtilities::getVariables(dpID = "DP1.30012.001")

# Create a copy for modification
variables_30012_custom <- variables_30012_original

# Reset all primaryKey values to "N"
variables_30012_custom$primaryKey <- "N"

# Set only the specific primary keys we want to "Y"
# fsp_boutMetadata: eventID
variables_30012_custom$primaryKey[
  variables_30012_custom$table == "fsp_boutMetadata" & 
  variables_30012_custom$fieldName == "eventID"
] <- "Y"

# fsp_sampleMetadata: spectralSampleID
variables_30012_custom$primaryKey[
  variables_30012_custom$table == "fsp_sampleMetadata" & 
  variables_30012_custom$fieldName == "spectralSampleID"
] <- "Y"

# fsp_spectralData: spectralSampleID
variables_30012_custom$primaryKey[
  variables_30012_custom$table == "fsp_spectralData" & 
  variables_30012_custom$fieldName == "spectralSampleID"
] <- "Y"

# Verify the changes
cat("\n=== CUSTOM PRIMARY KEY VERIFICATION ===\n")
fsp_tables <- c("fsp_boutMetadata", "fsp_sampleMetadata", "fsp_spectralData")

for (table_name in fsp_tables) {
  cat("\nTable:", table_name, "\n")
  
  # Get variables for this table
  table_vars <- variables_30012_custom[variables_30012_custom$table == table_name, ]
  
  # Check primary keys
  primary_keys <- table_vars[table_vars$primaryKey == "Y", ]
  
  if (nrow(primary_keys) > 0) {
    cat("  Primary keys set to 'Y':\n")
    for (i in 1:nrow(primary_keys)) {
      cat("    -", primary_keys$fieldName[i], "\n")
    }
  } else {
    cat("  ❌ NO PRIMARY KEYS SET!\n")
  }
}

# Save the custom variables file
write.csv(variables_30012_custom, "variables_30012_custom.csv", row.names = FALSE)

cat("\n✅ Custom variables file created: variables_30012_custom.csv\n")
cat("This file has only the primary keys you specified marked as 'Y'.\n")
cat("You can now use this file with removeDups() instead of the original variables_30012.\n") 