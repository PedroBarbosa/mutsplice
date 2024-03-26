import argparse
import os
from collections import defaultdict
import pandas as pd
import glob

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
    df = df[df.RBP_Species == "Homo_sapiens"]
    ids_per_rbp = df.groupby('RBP_Name')['Motif_ID'].apply(lambda x: x.unique().tolist()).to_dict()
    out = {}
    for k, v in ids_per_rbp.items():
        v = [x for x in v if x != "."]
        if len(v) >= 1:
            out[k] = v
    return out

def process_PWMs(pwms: list, rbp_map: dict):
    """
    Process PWMs to produce a single
    motifs file for the whole database

    :param list pwms: List with pwms, each one representing a file
    :param dict rbp_map: Dictionary mapping
    RBP names to IDs.
    """

    # Convert to MEME format
    outfile = open("CISBP_RNA_PWMs_database.txt", "w")
    outfile.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
                  "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
    
    for rbp_name, motif_ids in rbp_map.items():

        for m_id in motif_ids:
            if f"pwms/{m_id}.txt" in pwms:
                has_lines = any(True for _ in open(f"pwms/{m_id}.txt", 'r'))
                if has_lines is False:
                    print(f'Removed motif {m_id} because it is an empty matrix')
                    continue
                else:
                    pwm = pd.read_csv(f"pwms/{m_id}.txt", sep='\t')
  
                    outfile.write("MOTIF {} {}\n\n".format(m_id, rbp_name))
                    outfile.write("letter-probability matrix: alength= 4 w= {}\n".format(pwm.shape[0]))

                    for _, _row in pwm.iterrows():
          
                        row = list(map(str, _row.to_list()[1:]))
                        outfile.write('\t'.join(row) + '\n')
                    outfile.write('\n')
            else:
                print(f"File of motif ID {m_id} of {rbp_map} RBP does not exist.")


def main():
    parser = argparse.ArgumentParser(description="Process CIS-BP_RNA files so that a unique PWM file in MEME format "
                                                 "is created with all the RBPs in the database.")

    parser.add_argument('--pwm_dir', default="pwms", help='Directory with all the PWM in the database.')
    parser.add_argument('--db', default="RBP_Information.txt", help='File with metadata about the RBPs and motifs. '
                                     'File name in the database: "RBP_Information.txt"')

    args = parser.parse_args()
    pwms = glob.glob(f"{args.pwm_dir}/M*txt")
    db = args.db

    assert all([os.path.isfile(f) for f in pwms])
    assert os.path.isfile(db), "DB argument provided is not a valid file"

    rbp_map = process_db(db)
    process_PWMs(pwms, rbp_map)


if __name__ == '__main__':
    main()
