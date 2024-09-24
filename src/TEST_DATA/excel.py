import pandas as pd

# Define the list of communities
communities = [
    [['3841', '5706', '5602', '1155', '5260', '5425', '394', '5705'], 
     ['5601', '5701', '3676', '5629', '5703', '5708', '450', '5704', '5707', '5699', '876', '5702', '4227', '2815', '5643', '1023', '5552']],
    [['450', '5597', '5601', '5689', '5643', '532'], 
     ['3676', '3170', '5114', '5688', '5552']],
    [['450', '2011', '5556', '5469', '308', '5550', '5552', '5554'], 
     ['143', '5558', '5557', '5545', '4678', '5549', '3713', '5551', '5555', '5553', '3747', '5548']],
    [['5152', '5153'], 
     ['5149', '1884', '1213']]
]

# Initialize dictionary to store clusters
clusters = {'Cluster 1': [], 'Cluster 2': []}

# Concatenate elements of the first and second list of each community
for community in communities:
    clusters['Cluster 1'].extend(community[0])
    clusters['Cluster 2'].extend(community[1])

# Create a DataFrame from the dictionary
df = pd.DataFrame.from_dict(clusters, orient='index').transpose()

# Save the DataFrame to a new Excel file
df.to_excel("C:\\Users\\Rama Krishna\\Desktop\\communities17.xlsx", index=False)
