import pandas as pd
import os

datasets = {
    "data/amazon_review.csv": ("AMZ", "review_text", "sentiment"),
    "data/ai_human.csv": ("TEXT", "text", "label"),
    "data/fake_news.csv": ("NEWS", "content", "label")
}

def load_and_tag_dataset(path, task_name, text_col, label_col):
    try:
        df = pd.read_csv(path)
        
        if text_col not in df.columns or label_col not in df.columns:
            return None
        
        df = df.rename(columns={text_col: "text", label_col: "label"})
        df["task"] = task_name 
        return df[["task", "text", "label"]]
    except Exception as e:
        print("Error.")
        return None

def merge_datasets():
    all_dfs = []
    for path, (task_name, text_col, label_col) in datasets.items():
        if os.path.exists(path):
            df = load_and_tag_dataset(path, task_name, text_col, label_col)
            if df is not None:
                all_dfs.append(df)
        else:
            print(f"Error {path} not found.")

    if not all_dfs:
        print("Unable to load data.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total data: {len(merged_df)}")
    merged_df.to_csv("merged_datas_no_emotion.csv", index=False)
    print("Saced: merged_datas_no_emotion.csv")

if __name__ == "__main__":
    merge_datasets()
