import argparse
import os
from collections import defaultdict
import pandas as pd


def process_db(file: str):
    """
    Parse database file so that for
    each human RBP we get the list
    of associated IDs for downstream processing

    :param str file: ATrRACT db file

    :return dict: For each RBP (keys), the list
    of IDs (values) is returned
    """
    df = pd.read_csv(file, sep="\t")
    df = df[df.Organism == "Homo_sapiens"]
    return df.groupby('Gene_name')['Matrix_id'].apply(lambda x: x.unique().tolist()).to_dict()


def process_PWMs(pwms: str, rbp_map: dict):
    """
    Process PWMs to produce a single
    motifs file for the whole database

    :param str pwms: Input PWMs file
    :param dict rbp_map: Dictionary mapping
    RBP names to IDs.
    """
    # Read PWM file
    pwm_id_map, pwm_len_map = {}, {}
    matrix, _len = [], ""
    first = True
    f = open(pwms, 'r')
    for line in f:
        line = line.rstrip()

        if line.startswith(">"):

            # Id and length separated by "_" in a single string
            _id = line[1:].split()[0]
            _len = line[1:].split()[1]

            if first:
                prev_id = _id
                prev_len = _len
                continue

            else:

                pwm_id_map[prev_id] = matrix
                pwm_len_map[prev_id] = prev_len
                prev_id = _id
                prev_len = _len
                matrix = []

        else:
            first = False
            matrix.append(line.split())

    # Last PWM
    pwm_id_map[_id] = matrix
    pwm_len_map[_id] = _len

    # Convert to MEME format
    outfile = open("ATtRACT_PWMs_database.txt", "w")
    outfile.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
                  "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")

    for rbp, ids in rbp_map.items():

        for motif_id in ids:

            outfile.write("MOTIF {} {}\n\n".format(motif_id, rbp))
            outfile.write("letter-probability matrix: alength= 4 w= {}\n".format(pwm_len_map[motif_id]))
            pwm = pwm_id_map[motif_id]
            for row in pwm:
                outfile.write('\t'.join(row) + '\n')
            outfile.write('\n')


def main():
    parser = argparse.ArgumentParser(description="Process ATrRACT files so that a unique PWM file in MEME format "
                                                 "is created with all the RBPs in the database.")

    parser.add_argument('--pwm_file', help='Directory with all the PWM in the database. Filename '
                                           'in the database: "pwm.txt"')
    parser.add_argument('--db', help='File with metadata about the RBPs and motifs. '
                                     'File name in the database: "ATrRACT_db.txt"')

    args = parser.parse_args()

    pwms = "pwm.txt" if not args.pwm_file else args.pwm_file
    db = "ATtRACT_db.txt" if not args.db else args.db

    assert os.path.isfile(pwms), "PWMs argument provided is not a valid file."
    assert os.path.isfile(db), "DB argument provided is not a valid file"

    rbp_map = process_db(db)
    process_PWMs(pwms, rbp_map)


if __name__ == '__main__':
    main()
