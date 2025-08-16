import re, string, jsonlines
import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

data_path = "./money_related_content.jsonl"

class Preprocess:
    def __init__(self):
        self.df = None

    def loadData(self):
        with jsonlines.open(data_path, "r") as reader:
            data = [obj for obj in reader]

        self.df = pd.DataFrame(data)
        print("data shape:", self.df.shape)

    def cleanData(self):
        self.df['raw_text'] = self.df['title'].fillna('') + ' ' + self.df['description'].fillna('') + ' ' + self.df['tags'].fillna('')
        def cleanText(raw_text):
            clean_text = raw_text.lower()
            clean_text = re.sub(r'https?://\S+', ' ', clean_text)
            clean_text = re.sub(r'[' + re.escape(string.punctuation) + ']', ' ', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text)
            clean_text = clean_text.strip()
            return clean_text
        self.df['clean_text'] = self.df['raw_text'].apply(cleanText)

        print(f"{self.df.shape[0]} entries are cleaned")



if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.loadData()
    preprocess.cleanData()
    # only save clean text column as the preprocess result
    preprocess.df[['clean_text']].to_parquet('./money_df_clean_text.parquet', compression='snappy')
