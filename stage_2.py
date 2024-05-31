from sd3_api_wrapper import generate_image
import os

fp = "model_out_0.csv"
outdir = "./samples/0"
os.makedirs(outdir, exist_ok = True)


import pandas as pd
import json
from multiprocessing import Pool, cpu_count
from functools import partial

from tqdm import tqdm

def process_row(row, index, outdir):
    json_filename = os.path.join(outdir, f"{index:08d}.json")
    png_filename = os.path.join(outdir, f"{index:08d}.png")
    
    # Check if files already exist
    if os.path.exists(json_filename) and os.path.exists(png_filename):
        return
    
    # Save JSON
    with open(json_filename, 'w') as json_file:
        json.dump(row.to_dict(), json_file)
    
    # Generate and save image
    success = False

    while not success:
        success = generate_image(row['prompt'], png_filename)

def main(fp, outdir):
    df = pd.read_csv(fp)
    os.makedirs(outdir, exist_ok=True)
    
    with Pool(processes=min(100,cpu_count())) as pool:
        list(tqdm(pool.starmap(partial(process_row, outdir=outdir), [(row, idx) for idx, row in df.iterrows()]), total=len(df)))

if __name__ == "__main__":
    main(fp, outdir)

