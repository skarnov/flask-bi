# Necessary libraries
library(ggplot2)
library(tidyr)
library(reshape2)
library(car)

# Import the file
data <- read.csv("In//brexit_data.csv")

# Checking for missing values #

# To indices of rows with missing values
rows_with_missing <- which(apply(data, 1, function(row) any(is.na(row))))

# To indices of columns with missing values
cols_with_missing <- which(colSums(is.na(data)) > 0)

# To indices subset of data containing only the rows with missing values.
data_with_missing <- data[apply(data, 1, function(row) any(is.na(row))), ]

# To indices subset of data containing only the columns with missing values
data[, colSums(is.na(data)) > 0]

# Remove duplicate entries
data <- unique(data)

# Filter out non-finite values in the 'Age_Mean' column. Since the data-set has non-finite values in the 'Age_Mean' column
data <- data[is.finite(data$age_mean), ]

# Plotting the histogram for respondents by Age
ggplot(data, aes(x=age_mean)) +
  geom_histogram(binwidth = 1, fill="skyblue", color="black") +
  labs(title="Distribution of Respondents by Age Mean", x="Age Mean", y="Number of Respondents") +
  theme_minimal()


# Reshape the data into a longer format
brexit_data_long <- pivot_longer(data, cols = c(Percent_Remain, Percent_Leave),
                                 names_to = "Vote_Type", values_to = "Percentage")

# Plotting the bar chart for comparison of remain and leave votes by the UK Regions
ggplot(brexit_data_long, aes(x = reorder(Region, Percentage), y = Percentage, fill = Vote_Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparison of Remain and Leave Votes by Region in the UK",
       x = "Region",
       y = "Percentage of Votes",
       fill = "Vote Type") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c("red", "blue"), labels = c("Leave", "Remain")) +
  guides(fill = guide_legend(title = "Vote Type")) +
  coord_flip()


# Plotting the scatter plot for finding the relationship between age and earnings behind the Brexit
scatterplotMatrix(~ age_mean + Earnings_Mean + Percent_Leave, data = data, 
                  main = "Scatter Plot Matrix of Age, Income, and Brexit")


# Plotting the heatmap for other socio-economic aspect
heatmap_data <- data[, c("Percent_Leave",
                                "Economically_active_percent", 
                                "Employed_of_economically_active_percent", 
                                "Unemployed_Age50_74_percent", 
                                "Occup_intermediate_percent", 
                                "Bachelors_deg_percent", 
                                "Birth_MidEast_Asia_percent")]

# Compute the correlation matrix
correlation_matrix <- cor(heatmap_data)

# Reshape the correlation matrix for plotting
melted_correlation <- melt(correlation_matrix)

# Heatmap
ggplot(melted_correlation, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 10, hjust = 1)) +
  coord_fixed() +
  labs(title = "Correlation Heatmap of Brexit Data",
       x = "Variables", y = "Variables")
