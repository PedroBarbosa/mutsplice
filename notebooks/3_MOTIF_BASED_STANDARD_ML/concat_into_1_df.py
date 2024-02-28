import pandas as pd
import glob

files = glob.glob('*.tsv.gz')

df = pd.concat(map(lambda file: pd.read_csv(file, sep="\t"), files))
df.to_csv('ALL_CONCAT.tsv.gz', sep='\t', compression='gzip', index=False)
