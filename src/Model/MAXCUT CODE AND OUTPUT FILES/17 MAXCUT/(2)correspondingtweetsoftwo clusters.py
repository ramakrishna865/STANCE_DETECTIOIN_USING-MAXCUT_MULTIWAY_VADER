import pandas as pd

class Extract:
    def __init__(self, data, clusters):
        self.post = pd.read_excel(data, sheet_name="post", engine="openpyxl")
        self.text = pd.read_excel(data, sheet_name="text", engine="openpyxl")
        self.cluster = pd.read_excel(clusters, engine="openpyxl")

    def output(self, TextId):
        TextId = TextId.astype(str)
        Text = self.text["text"][self.text["text_id"].isin(TextId)].reset_index(drop=True)
        return Text

    def AuthText(self, ClusterVal):
        TextId = self.post["text_id"][self.post["author_id"].isin(self.cluster[ClusterVal].values)].reset_index(drop=True)
        AuthorId = self.post["author_id"][self.post["text_id"].isin(TextId.values)].reset_index(drop=True)
        DiscussionId = self.post["discussion_id"][self.post["text_id"].isin(TextId)].reset_index(drop=True)
        StanceId = self.post["discussion_stance_id"][self.post["text_id"].isin(TextId)].reset_index(drop=True)  # Assuming stance_id exists

        Texted = self.output(TextId)

        ClusterData = pd.DataFrame({
            "discussion_id": DiscussionId,
            "stance_id": StanceId,
            "author_id": AuthorId,
            "text": Texted
        })

        return ClusterData

    def FirstOne(self):
        Clustered1 = self.AuthText("C1")
        Clustered2 = self.AuthText("C2")

        Clustered1.to_excel("C:\\Users\\Rama Krishna\\Desktop\\17 MAXCUT\\17thc1tweets.xlsx", index=False)
        Clustered2.to_excel("C:\\Users\\Rama Krishna\\Desktop\\17 MAXCUT\\17thc2tweets.xlsx", index=False)

obj = Extract("C:\\Users\\Rama Krishna\\Desktop\\17 MAXCUT\\17th.xlsx", "C:\\Users\\Rama Krishna\\Desktop\\17 MAXCUT\\17thCluster.xlsx")
obj.FirstOne()
