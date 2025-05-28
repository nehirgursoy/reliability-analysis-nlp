import pandas as pd
from tqdm import tqdm
from transformers import pipeline

#The datasets are merged into one with their identifying task tags
#So the datas are labeled for modelt to multi-task
#Now those datas will be analyzed by emotion

emotion_analyzer = pipeline(
    "text-classification", 
    model="nateraw/bert-base-uncased-emotion",
    tokenizer="nateraw/bert-base-uncased-emotion",
    truncation=True
    )

df = pd.read_csv("merged_datas_no_emotion.csv")

tqdm.pandas()
df["emotion"] = df["text"].progress_apply(lambda x: emotion_analyzer(x)[0]["label"])


df.to_csv("merged_datas_with_emotion.csv", index=False)
print("merged_datas_with_emotion.csv saved.")
