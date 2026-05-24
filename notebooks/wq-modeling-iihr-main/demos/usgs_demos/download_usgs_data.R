# https://waterdata.usgs.gov/nwis/inventory/?site_no=05412500

library(dataRetrieval)
library(stringr)
library(sf)

siteNumber <- "USGS-05412500"
siteINFO <- read_waterdata_monitoring_location(siteNumber)

info <- read_waterdata_ts_meta(monitoring_location_id = siteNumber)
info_subset <- info[c("parameter_name", "unit_of_measure", "statistic_id", "parameter_code", "computation_identifier")]

pcode_to_name(info$parameter_code)

info_subset_instantaneous <- info_subset[info_subset$computation_identifier == "Instantaneous", ]

pcodes <- info_subset_instantaneous$parameter_code

data <- readNWISuv(siteNumbers = str_remove(siteNumber, "USGS-"), parameterCd = pcodes) #"00060"


data


plot(data$dateTime, data$X_99133_00000)


filename_base <- "/Users/jgomezvelez/Downloads/aorc_data/"

# Data
filename <- paste0(filename_base, "data_site_", siteNumber, ".csv")
write.csv(data, filename, row.names = FALSE)

# Metadata 
filename <- paste0(filename_base, "metadata_site_", siteNumber, ".csv")
info_subset_instantaneous_no_geom <- st_drop_geometry(info_subset_instantaneous)
write.csv(info_subset_instantaneous_no_geom, filename, row.names = FALSE)


