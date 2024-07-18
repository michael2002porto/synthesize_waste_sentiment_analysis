import argparse
import sys
import json
import pandas as pd
from tqdm import tqdm

# access the parent folder
sys.path.append(".")

from model.llama3_70b import SynthesizeSentiment

def collect_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sentiment_rows_per_llm_batch", type=int, default=10)
    return parser.parse_args()

def load_existing_dataset():
    dataset = pd.read_csv("data/indonlu_smsa_without_duplicate.csv")
    dataset = dataset[["text", "label"]]
    return dataset

if __name__ == '__main__':
    args = collect_parser()
    num_sentiment = args.num_sentiment_rows_per_llm_batch

    i = 0
    general_sentiment = []
    waste_sentiment = []
    seen_waste_sentiment = set()  # To track seen titles

    dataset = load_existing_dataset()
    # total_sentiment = len(dataset.index)
    total_sentiment = 10
    label_yang_ada = dataset["label"].value_counts()
    print(label_yang_ada)

    pbar = tqdm(total=total_sentiment, desc="Synthesizing sentiment")

    while i < total_sentiment:
        try:
            label = dataset.iloc[i]["label"]
            text = dataset.iloc[i]["text"]
            synthesize_sentiment = SynthesizeSentiment(label, text)
            output_text = synthesize_sentiment.setup()
            output_text = output_text.replace('"', '').replace("'", "")  # Remove quotes

            print()
            print(i)
            print(label)
            print(output_text)

            if output_text in seen_waste_sentiment:
                continue

            seen_waste_sentiment.add(output_text)
            sentiment_dict = {
                "text": output_text,
                "label": label
            }
            waste_sentiment.append(sentiment_dict)
            i += 1
            pbar.update(1)

        except Exception as e:
            print(f"Error processing sentiment: {e}")
            # Save sentiments_data in case of error (optional)
            df = pd.DataFrame(waste_sentiment)
            df.to_excel("data/synthesized_waste_sentiment_partial.xlsx", index=False)

    pbar.close()
    df = pd.DataFrame(waste_sentiment)

    # Save DataFrame to Excel file
    output_file = "data/synthesized_waste_sentiment.xlsx"
    df.to_excel(output_file, index=False)