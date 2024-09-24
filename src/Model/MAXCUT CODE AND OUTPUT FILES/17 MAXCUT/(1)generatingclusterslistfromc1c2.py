import pandas as pd

# Read the first Excel file
excel1 = pd.read_excel("C:\\Users\\Rama Krishna\\Desktop\\17 MAXCUT\\17thCluster1.xlsx", usecols=['C1'])

# Read the second Excel file
excel2 = pd.read_excel("C:\\Users\\Rama Krishna\\Desktop\\17 MAXCUT\\17thCluster2.xlsx", usecols=['C2'])

# Merge the two dataframes on their indexes
merged_excel = pd.concat([excel1, excel2], axis=1)

# Write the merged dataframe to a new Excel file
merged_excel.to_excel("C:\\Users\\Rama Krishna\\Desktop\\17 MAXCUT\\17thCluster.xlsx", index=False)

print("Merged Excel file has been created.")
