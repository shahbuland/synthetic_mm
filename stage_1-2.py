# Stitch multiple CSV files into a single CSV

if __name__ == "__main__":
    prefix = "./model_out_"
    out_path = "syn_prompts.csv"
    import os
    import pandas as pd

    csv_files = [f for f in os.listdir() if f.startswith(prefix) and f.endswith('.csv')]

    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    combined_df.to_csv(out_path, index=False)
