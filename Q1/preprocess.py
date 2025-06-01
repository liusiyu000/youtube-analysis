import re, string, jsonlines
import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

data_path = "F:\dissertationData\yt_metadata_en_money.jsonl"

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

        # money_keywords = [
        #     r'\bmake money\b', r'\bearn(ing)? money\b', r'\bpassive income\b',
        #     r'\bside hustle\b', r'\bextra income\b', r'\bfinancial freedom\b',
        #     r'\bwork from home\b', r'\bonline business\b', r'\bearn \$\d+',
        #     r'\baffiliate marketing\b', r'\bdropshipping\b', r'\bfreelance\b',
        #     r'\binvestment\b', r'\bstock trading\b', r'\bcryptocurrency\b',
        #     r'\bbitcoin\b', r'\bforex\b'
        # ]
        #
        # pattern = re.compile('|'.join(money_keywords))
        # self.df['is_money'] = self.df['clean_text'].str.contains(pattern, regex=True)
        # self.money_df = self.df[self.df['is_money']].copy()

        # print(f"{self.money_df.shape[0]} entries are about online money making")
        print(f"{self.df.shape[0]} entries are cleaned")



if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.loadData()
    preprocess.cleanData()
    # only save clean text column as the preprocess result
    preprocess.df[['clean_text']].to_parquet('./money_df_clean_text.parquet', compression='snappy')
