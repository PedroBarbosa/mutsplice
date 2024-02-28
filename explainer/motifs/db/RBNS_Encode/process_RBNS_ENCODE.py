import argparse
import os
import pandas as pd


def parse_file(motifs: str):
    """
    Reads the supp table 6 from ENCODE file
    to produce an output file similar to the
    rosina2017, listing the motifs per RBP
    in a format that can be consumed by
    a downstream pipeline

    :param str motifs: Input file
    """
    df = pd.read_csv(motifs, sep='\t')
    remame_map = {'Motif 5mer_logonum_stepwiseRminus1': 'TopKmer',
                  'Unnamed: 6': 'AllKmers'}

    df = df.rename(columns=remame_map)
    df['AllKmers'] = df.AllKmers.str.split(';')
    cols = [x for x in df.columns if x != 'AllKmers']
    df = df.set_index(cols).apply(pd.Series.explode).reset_index()
    df[['Motif', 'LogoNumber', 'StepwiseR_minus_1']] = df.AllKmers.str.split("_", expand=True)
    df = df.dropna()

    outfile = open('encode2020_RBNS_motifs.txt', 'w')
    for rbp, motifs in df.groupby('BP'):

        header = ">{}|{}|\n".format(rbp, motifs.Motif.nunique())
        motifs_flat = "*{}\n".format('|'.join(list(set(motifs.Motif))))
        outfile.write(header + motifs_flat + "\n")


def main():
    parser = argparse.ArgumentParser(description="Process RBNS-derived enriched k-mers from ENCODE RBP publication. "
                                                 "It will generate an output format similar to the rosina2017")

    parser.add_argument('--rbns_motifs_file', help='TSV file corresponding to the supplementary table 6 from '
                                                   'ENCODE publication (2020). If not set, it will look in the '
                                                   'current directory for the file: "raw_RBNS_motifs_suppData6.tsv".')
    args = parser.parse_args()

    motifs = "raw_RBNS_motifs_suppData6.tsv" if not args.rbns_motifs_file else args.rbns_motifs_file
    assert os.path.isfile(motifs), "Motifs file provided is not a valid file."

    parse_file(motifs)


if __name__ == '__main__':
    main()