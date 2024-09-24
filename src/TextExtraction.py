import csv
import pandas as pd
import numpy as np

#output:[discussion_id:author_id's:text_id's]

class DebateAnalyzer:
    def __init__(self, debate_file_path, text_file_path):
        self.df = pd.read_excel(debate_file_path, sheet_name="post")
        self.df2 = pd.read_excel(text_file_path, sheet_name="text")
        self.finaldict = {}

    def check_repetition(self, x, y):
        
        count = 0
        for i in range(len(self.df)):
            if self.df["discussion_id"][i] == x and self.df["author_id"][i] == y:
                count += 1
        return count == 1

    def process_data(self):
        for i in range(len(self.df)):
            discussion_id = self.df["discussion_id"][i]
            author_id = self.df["author_id"][i]
            text_id = self.df["text_id"][i]

            if self.check_repetition(discussion_id, author_id):
                if discussion_id in self.finaldict:
                    self.finaldict[discussion_id][author_id] = [text_id]
                else:
                    self.finaldict[discussion_id] = {author_id: [text_id]}
            else:
                if discussion_id in self.finaldict:
                    if author_id in self.finaldict[discussion_id]:
                        self.finaldict[discussion_id][author_id].append(text_id)
                    else:
                        self.finaldict[discussion_id][author_id] = [text_id]
                else:
                    self.finaldict[discussion_id] = {author_id: [text_id]}

    def merge_texts(self):
        for i in self.finaldict:
            for j in self.finaldict[i]:
                text = ""
                for k in self.finaldict[i][j]:
                    for a in range(len(self.df2)):
                        if self.df2["text_id"][a] == str(k):
                            text += self.df2["text"][a]
                self.finaldict[i][j] = [text]

    def get_final_dict(self):
        return self.finaldict

debate_analyzer = DebateAnalyzer("input/createdebate_released_no_parse.xlsx", "input/createdebate_released_no_parse.xlsx")
debate_analyzer.process_data()
debate_analyzer.merge_texts()
final_result = debate_analyzer.get_final_dict()
print(final_result)
