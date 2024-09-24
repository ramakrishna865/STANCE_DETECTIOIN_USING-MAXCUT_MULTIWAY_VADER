from Service.LoadInput import LoadInput
from Service.ConversationForest import ConversationForest
from Service.InteractionNetwork import InteractionNetwork
from Service.TwoCoreReduction import TwoCoreReduction
from Service.SDP import SDP
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random as random
import re
import math
import transformers
import statistics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import torch
import traceback
from torch.nn.functional import cosine_similarity
import numpy as np
from numpy.linalg import norm
import logging

logging.basicConfig(filename='program.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.warning("=@#$%^"*15)

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    'cardiffnlp/twitter-roberta-base-sentiment-latest')
# Load pre-trained model and tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')


def getGroundTruth(discussion_id, filePath):
    post = pd.read_excel(filePath, sheet_name="Sheet1")
    post_df = pd.DataFrame(post, columns=['Discussion ID', "cluster2", "Discussion Stance"])
    idfilter = post_df[post_df["Discussion ID"] == discussion_id]
    ret = {}
    for aut, stan in zip(idfilter["cluster2"], idfilter["Discussion Stance"]):
        if np.isnan(stan):
            stan = random.randint(0, 1)
        else:
            stan = int(stan)
        if aut not in ret.keys():
            ret[aut] = [stan]
        else:
            ret[aut].append(stan)
    for i in ret.keys():
        ret[i] = max(set(list(ret[i])), key=ret[i].count)
    return ret


def showGraph(graph, bestCut):
    pos = nx.spring_layout(graph, seed=3113794652)
    users = sorted(list(graph.nodes))
    postiveStance = []
    negativeStance = []
    for user in bestCut.cut:
        value = user.stance
        key = user.userId
        if value < float(0):
            negativeStance.append(key)
        else:
            postiveStance.append(key)

    usersLabels = {}
    for i in range(len(users)):
        usersLabels[users[i]] = users[i]
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    nx.draw_networkx_nodes(graph, pos, nodelist=negativeStance, node_color="tab:blue", **options)
    nx.draw_networkx_nodes(graph, pos, nodelist=postiveStance, node_color="tab:green", **options)
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, usersLabels, font_size=11, font_color="black")
    plt.tight_layout()
    plt.axis("off")
    plt.show()
    # plt.pause(2)
    # plt.close()


def calculateAcc(ids, Partition, ground_truth):
    ind = -1
    truth = []

    for i in ids:
        if i.isdigit():
            i = int(i)
            if i in ground_truth.keys():
                truth.append(ground_truth[i])
        else:
            ind = ids.index(i)
    Partition = np.delete(Partition, ind)

    match = 0
    for i, j in zip(Partition, truth):
        if i == j:
            match += 1

    cm = confusion_matrix(truth, Partition)
    precision = cm[0][0] / (cm[0][0] + cm[1][0])
    recall = cm[0][0] / (cm[0][0] + cm[0][1])
    F1 = 2 * (precision * recall) / (precision + recall)
    return [match, accuracy_score(truth, Partition), precision_score(truth, Partition, average='weighted'), F1]


def find_labels(path, discussion_id):
    df = pd.read_excel(path, sheet_name='discussion_stance')
    filtered_rows = df[df['discussion_id'] == discussion_id]
    stance_rows = filtered_rows['discussion_stance'].head(2).tolist()
    id_rows = filtered_rows['topic_id'].head(2).tolist()
    for i, j in enumerate(stance_rows):
        # print(id_rows[i], type(id_rows[i]), isinstance(id_rows[i], type('g')))
        if isinstance(id_rows[i], type('String')) and str(id_rows[i]).isnumeric() is False:
            stance_rows[i] = str(stance_rows[i] + id_rows[i])
        if str(stance_rows[i])[0] == "'":
            stance_rows[i] = stance_rows[i][1:]
        if str(stance_rows[i])[-1] == "'":
            stance_rows[i] = stance_rows[i][:-1]
    # print("Labels : ", stance_rows)
    return stance_rows


def Textsplit(path, d_list):
    cols1 = ['discussion_id', 'author_id', 'text_id']
    cols2 = ['text_id', "text"]

    post_df = pd.read_excel(path, sheet_name='post', usecols=cols1)
    post_df['text_id'] = pd.to_numeric(post_df['text_id'], errors='coerce')
    post_df.dropna(subset=['text_id'], inplace=True)
    post_df['text_id'] = post_df['text_id'].astype(int)

    text_df = pd.read_excel(path, sheet_name='text', usecols=cols2)
    text_df['text_id'] = pd.to_numeric(text_df['text_id'], errors='coerce')
    text_df.dropna(subset=['text_id'], inplace=True)
    text_df['text_id'] = text_df['text_id'].astype(int)

    # Merge the two DataFrames based on the 'text_id' column
    merged_df = pd.merge(post_df, text_df, on='text_id')
    result_df = merged_df[['discussion_id', 'author_id', 'text']]
    grouped_df = result_df.groupby('discussion_id')

    ret = {}
    np.set_printoptions(suppress=True)
    # creating individual dictionaries keys as discussion id and values as dictionaries of author ids and their texts
    for discuss_id, data in grouped_df:
        author_data = dict()
        if int(discuss_id) not in d_list:
            continue
        for index, j in data.iterrows():
            if j['author_id'] not in author_data.keys() and isinstance(j["author_id"], int):
                text_data = list()
                for ind, i in result_df.iterrows():
                    if i["discussion_id"] == discuss_id and i["author_id"] == j["author_id"]:
                        text_data.append(i['text'])
                if len(text_data) != 0:
                    author_data[j["author_id"]] = text_data
        ret[discuss_id] = author_data

    return ret


def getAuthorStance(data, label0, label1):
    embeddings = (generate_embeddings(data))
    # print(label0, label1, embeddings)
    simi1 = np.dot(embeddings, label0) / (norm(embeddings, axis=1) * norm(label0))
    simi2 = np.dot(embeddings, label1) / (norm(embeddings, axis=1) * norm(label1))
    label0vote = 0
    label1vote = 0
    for i, j in zip(simi1, simi2):
        if i > j:
            if i > 0:
                label0vote += i
            else:
                label0vote += 0.1  # Introducing noise
        else:
            if j > 0:
                label1vote += j
            else:
                label1vote += 0.1  # Introducing noise
    # print("similarities", simi1, simi2)
    # print("votes", label0vote, label1vote)
    # data = {
    #     "label0similarity": simi1,
    #     "label1similarity": simi2,
    #     "True": [-1] * len(simi1),
    #     "False": [-1] * len(simi1)
    # }
    # result = pd.DataFrame(data)
    #
    # result["label"] = result["label0similarity"] > result["label1similarity"]
    # print(result.head(15))
    # label = result["label"].value_counts().idxmax()
    # print(label)
    return {True: 0, False: 1}[label0vote > label1vote], {True: label0vote, False: label1vote}[label0vote > label1vote]


def generate_embeddings(data):
    # print("for procession", data)
    inputs = tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        model.eval()
        outputs = model(**inputs)
    return (outputs.logits).numpy()


datasets = [3]
Outputdata = pd.DataFrame(columns=["Dataset Name", "Discussion IF", "Stance label1", "Stance label2", "Key authors", "Swap [True/False]", "Number of nodes before 2-core reduction", "Number of nodes after 2-core reduction", "Text based swap", "Total number of correct predictions ", "Accuracy", "Precision", "F1", "Topic Accuracy", "Topic F1"])
all_communities_df = pd.DataFrame(columns=['Discussion_ID', 'Community_ID', 'User_ID'])

for topic in datasets:
    print("Topic id:", topic)
    filePath = "../input/" + str(topic) + ".xlsx"
    loadInput = LoadInput(filePath)
    loadInput.loadDataFromAllSheets()
    conversationForest = ConversationForest(loadInput)
    conversationForest.buildConversationTrees()
    trees = conversationForest.getAllConversationTrees()

    d_list = []
    for key in trees.keys():
        d_list += [int(key.split("_")[0])]

    text_input = Textsplit(filePath, d_list)

    over_all_acc = 0
    F1acc = 0
    n = 0
    for key in trees.keys():
        tree = trees[key]
        tree.showConversationTree()
        iNetwork = InteractionNetwork(tree, [])
        iNetwork.showInteractionNetwork()
        discussion_id = int(key.split("_")[0])
        before_nodes = len(iNetwork.graph.nodes)
        print("discussion id:", discussion_id)

        graphAfter2CoreReduction = TwoCoreReduction(iNetwork.graph).run2CoreReduction()
        after_nodes = len(graphAfter2CoreReduction.nodes)

        try:
            activeUsers = sorted(graphAfter2CoreReduction.nodes)
            if len(activeUsers) == 0:
                print("0 nodes i.e skipped")
                continue
            sdpObj = SDP(graphAfter2CoreReduction)
            sdpObj.runSDP()
            bestCut = sdpObj.getBestCut()
            showGraph(graphAfter2CoreReduction, bestCut)

            Partition = []
            Users = []
            for user in bestCut.cut:
                value = user.stance
                Users.append(user.userId)
                if value < float(0):
                    Partition.append(0)
                else:
                    Partition.append(1)

            communities = [[] for _ in range(2)]

            for i in range(len(graphAfter2CoreReduction.nodes)):
                communities[Partition[i]].append(list(graphAfter2CoreReduction.nodes)[i])

            print("Partitions:", Partition)
            print("Communities:", communities)

            # df = pd.DataFrame({'Cluster1': communities[0], 'Cluster2': communities[1]})
            # df.to_excel('ClusterData.xlsx', index=False)

            ground_truth = getGroundTruth(discussion_id, filePath)
            authors_text = text_input.get(discussion_id)
            labels = find_labels(filePath, discussion_id)
            label0 = labels[0]
            label1 = labels[1]
            label0_embeddings = generate_embeddings(label0)
            label1_embeddings = generate_embeddings(label1)

            textualPercentage = math.ceil(len(Users) * 0.25) + 1
            textual_author_ids = []
            for _ in range(textualPercentage):
                maxposts = 0
                auth = 0
                for author_id, posts in authors_text.items():
                    if maxposts < len(posts) and author_id not in textual_author_ids:
                        auth = author_id
                        maxposts = len(posts)
                textual_author_ids.append(auth)

            swap = [0, 0]  # [True, False]
            draw = textual_author_ids.pop()
            for author in textual_author_ids:
                posts = authors_text.get(author)
                authorStance, simi = getAuthorStance(posts, label0_embeddings[0], label1_embeddings[0])
                communityPartition = ({True: 0, False: 1}[str(author) in communities[0]])
                if authorStance != communityPartition:
                    swap[0] += simi
                else:
                    swap[1] += simi
                print(communityPartition, authorStance)
            print("Swap majority", swap)

            if swap[0] == swap[1]:
                posts = authors_text.get(draw)
                authorStance, simi = getAuthorStance(posts, label0_embeddings[0], label1_embeddings[0])
                communityPartition = ({True: 0, False: 1}[str(draw) in communities[0]])
                if authorStance != communityPartition:
                    swap[0] += simi
                else:
                    swap[1] += simi

            if swap[0] > swap[1]:
                Partition = list(np.logical_not(Partition).astype(int))
                print("swapped")

            accu = calculateAcc(Users, Partition.copy(), ground_truth)
            print("accuracy", accu)
            for i in range(len(accu)):
                if math.isnan(accu[i]):
                    accu[i] = 0
            n += 1
            over_all_acc = (over_all_acc * (n - 1) + accu[1]) / n
            F1acc = (F1acc * (n - 1) + accu[3]) / n

            Outputdata.loc[len(Outputdata.index)] = [({True: "Convince me", False: "Create debate"}[type(topic) == int]), str(topic), discussion_id, label0, label1, textual_author_ids, swap, before_nodes, after_nodes, swap[0] > swap[1], accu[0], accu[1], accu[2], accu[3], over_all_acc, F1acc]

            # Append community data to all_communities_df DataFrame
            community_data = pd.DataFrame(columns=['Discussion_ID', 'Community_ID', 'User_ID'])
            for community_id, users in enumerate(communities):
                for user_id in users:
                    community_data = community_data.append({'Discussion_ID': discussion_id, 'Community_ID': community_id, 'User_ID': user_id}, ignore_index=True)
            all_communities_df = pd.concat([all_communities_df, community_data], ignore_index=True)

        except Exception as e:
            print("error occurred: ", e)
            traceback.print_exc()
            logging.error("Exception occurred", exc_info=True)

print(Outputdata.head())
print(Outputdata.tail())

Outputdata.to_excel("Results.xlsx", index=False)
all_communities_df.to_excel("All_Communities.xlsx", index=False)

InteractionNetwork(tree, [])