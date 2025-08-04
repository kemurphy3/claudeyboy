# Custom duplicate checking functions for FSP tables
# These functions use only the specific primary keys we want, not all primary keys in variables file

library(dplyr)

# Function to check duplicates using a single specified primary key
check_duplicates_single_key <- function(data, primary_key_field, table_name) {
  
  # Convert to data frame if needed
  data <- as.data.frame(data, stringsAsFactors = FALSE)
  
  # Handle empty or single row data
  if (nrow(data) == 0) {
    data$duplicateRecordQF <- numeric()
    warning(paste("Data table", table_name, "is empty."))
    return(data)
  }
  
  if (nrow(data) == 1) {
    data$duplicateRecordQF <- 0
    warning(paste("Only one row of data present in", table_name, ". duplicateRecordQF set to 0."))
    return(data)
  }
  
  # Check if primary key field exists
  if (!primary_key_field %in% names(data)) {
    stop(paste("Primary key field", primary_key_field, "not found in", table_name))
  }
  
  # Initialize duplicate flag
  data$duplicateRecordQF <- 0
  
  # Convert primary key to character and lowercase for comparison
  data$keyvalue <- tolower(as.character(data[[primary_key_field]]))
  
  # Find duplicates based on the primary key
  duplicated_keys <- data$keyvalue[duplicated(data$keyvalue) | duplicated(data$keyvalue, fromLast = TRUE)]
  
  if (length(duplicated_keys) == 0) {
    message(paste("No duplicated key values found in", table_name, "!"))
    data$keyvalue <- NULL
    return(data)
  }
  
  # Get unique duplicated keys
  unique_duplicated_keys <- unique(duplicated_keys)
  
  message(paste(length(unique_duplicated_keys), "duplicated key values found in", table_name, 
                "representing", length(duplicated_keys), "non-unique records. Attempting to resolve."))
  
  # Process each set of duplicates
  for (key_val in unique_duplicated_keys) {
    # Get all rows with this key value
    dup_rows <- data[data$keyvalue == key_val, ]
    
    if (nrow(dup_rows) == 1) {
      # Shouldn't happen, but just in case
      next
    }
    
    # Check if all rows are identical (resolvable)
    dup_rows_no_key <- dup_rows[, !names(dup_rows) %in% c("duplicateRecordQF", "keyvalue")]
    
    if (nrow(unique(dup_rows_no_key)) == 1) {
      # All rows are identical - resolvable duplicate
      # Keep the first row, mark as resolved
      data$duplicateRecordQF[data$keyvalue == key_val] <- 1
      # Remove all but the first occurrence
      first_occurrence <- which(data$keyvalue == key_val)[1]
      data <- data[!(data$keyvalue == key_val & data$keyvalue != data$keyvalue[first_occurrence]), ]
    } else {
      # Rows are different - unresolvable duplicate
      data$duplicateRecordQF[data$keyvalue == key_val] <- 2
    }
  }
  
  # Clean up
  data$keyvalue <- NULL
  
  # Count results
  resolved_count <- sum(data$duplicateRecordQF == 1)
  unresolvable_count <- sum(data$duplicateRecordQF == 2)
  
  if (resolved_count > 0) {
    message(paste(resolved_count, "resolvable duplicates merged into matching records"))
  }
  if (unresolvable_count > 0) {
    message(paste(unresolvable_count, "unresolvable duplicates flagged with duplicateRecordQF=2"))
  }
  
  return(data)
}

# Function to create the three standard output dataframes (0, 1, 2)
create_duplicate_outputs <- function(data_with_flags, table_name) {
  
  # Create the three standard output dataframes
  data_0 <- dplyr::filter(data_with_flags, duplicateRecordQF == 0)
  data_1 <- dplyr::filter(data_with_flags, duplicateRecordQF == 1)
  data_2 <- dplyr::filter(data_with_flags, duplicateRecordQF == 2)
  
  # For unresolvable duplicates (QF=2), keep only the first instance
  if (nrow(data_2) > 0) {
    # Get the primary key field name based on table
    if (table_name == "fsp_boutMetadata") {
      primary_key <- "eventID"
    } else if (table_name == "fsp_sampleMetadata" || table_name == "fsp_spectralData") {
      primary_key <- "spectralSampleID"
    } else {
      stop("Unknown table name for primary key determination")
    }
    
    data_2_keep <- data_2 %>%
      dplyr::group_by(!!sym(primary_key)) %>%
      dplyr::filter(row_number() == 1) %>%
      dplyr::ungroup()
  } else {
    data_2_keep <- data_2
  }
  
  # Combine all data
  data_no_dups <- dplyr::bind_rows(data_0, data_1, data_2_keep)
  
  return(list(
    data_0 = data_0,
    data_1 = data_1, 
    data_2 = data_2,
    data_2_keep = data_2_keep,
    data_no_dups = data_no_dups
  ))
}

# Example usage functions for each FSP table
check_fsp_boutMetadata_duplicates <- function(data) {
  cat("Checking duplicates in fsp_boutMetadata using eventID as primary key...\n")
  result <- check_duplicates_single_key(data, "eventID", "fsp_boutMetadata")
  outputs <- create_duplicate_outputs(result, "fsp_boutMetadata")
  return(outputs)
}

check_fsp_sampleMetadata_duplicates <- function(data) {
  cat("Checking duplicates in fsp_sampleMetadata using spectralSampleID as primary key...\n")
  result <- check_duplicates_single_key(data, "spectralSampleID", "fsp_sampleMetadata")
  outputs <- create_duplicate_outputs(result, "fsp_sampleMetadata")
  return(outputs)
}

check_fsp_spectralData_duplicates <- function(data) {
  cat("Checking duplicates in fsp_spectralData using spectralSampleID as primary key...\n")
  result <- check_duplicates_single_key(data, "spectralSampleID", "fsp_spectralData")
  outputs <- create_duplicate_outputs(result, "fsp_spectralData")
  return(outputs)
}

cat("Custom duplicate checking functions loaded.\n")
cat("These functions use only the specific primary keys you specified:\n")
cat("- fsp_boutMetadata: eventID\n")
cat("- fsp_sampleMetadata: spectralSampleID\n") 
cat("- fsp_spectralData: spectralSampleID\n") 