import pandas as pd

def fun(input_file, extract_file):
    read_input = pd.read_excel(input_file)
    read_extract_input = pd.read_excel(extract_file)

    appended_data = []

    print("Clusters in input file:")
    print(read_input[['Cluster 1', 'Cluster 2']])

    print("Authors in extract file:")
    print(read_extract_input['Author ID'])

    for i in range(len(read_input['Cluster 1'])):
        if read_input['Cluster 1'][i] in read_extract_input['Author ID'].values:
            filtered_data = read_extract_input.loc[read_extract_input['Author ID'] == read_input['Cluster 1'][i], ['Text', 'Sentiment Score','Positive Count','Negative Count','Avg Positive','Avg Negative','Total Sentiment Score','Positive Sentiment Score','Negative Sentiment Score']]
            appended_data.append(filtered_data)

    for j in range(len(read_input['Cluster 2'])):
        if read_input['Cluster 2'][j] in read_extract_input['Author ID'].values:
            filtered_data1 = read_extract_input.loc[read_extract_input['Author ID'] == read_input['Cluster 2'][j],['Text','Sentiment Score','Positive Count','Negative Count','Avg Positive','Avg Negative','Total Sentiment Score','Positive Sentiment Score','Negative Sentiment Score']]
            appended_data.append(filtered_data1)

    if appended_data:
        appended_df = pd.concat(appended_data, ignore_index=True)
        with pd.ExcelWriter(input_file, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
            appended_df.to_excel(writer, index=False, header=False, startrow=len(pd.read_excel(input_file)) + 1)

fun('ClustersNodes.xlsx', 'TextDataWordnet_Sentiment_Score.xlsx')
