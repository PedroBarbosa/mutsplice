import argparse
import os
from collections import defaultdict


def parse_ids(file: str):
    """
    Parse motif IDs file so that for
    each RBP we get the list of associated IDs
    for downstream processing

    :return dict: For each RBP (keys), the list
    of IDs (values) is returned
    """

    out = defaultdict(list)
    f = open(file, 'r')
    for line in f:
        line = "".join(line.split())
        _id = line.split(",")[0]
        rbp = line.split(",")[1].replace("\"", "")
        out[rbp.split("(")[0]].append(_id)

    return out


def process_PWMs(pwm_dir, rbp_map):
    """
    Process PWMs to produce a single
    motifs file for the whole database

    :param str pwm_dir: Input directory
    :param dict rbp_map: Dictionary mapping
    RBP names to IDs.
    """
    outfile = open("oRNAment_PWMs_database.txt", "w")
    outfile.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
                  "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")

    for rbp, ids in rbp_map.items():

        for _id in ids:

            filename = _id.zfill(3) + ".PWM"
            pwm_f = open(os.path.join(pwm_dir, filename), 'r')
            pwm = [line.rstrip().split()[1:] for line in pwm_f if line[0].isdigit()]
            outfile.write("MOTIF M{} {}\n\n".format(_id.zfill(3), rbp))
            outfile.write("letter-probability matrix: alength= 4 w= 7\n")
            for row in pwm:

                outfile.write('\t'.join(row) + '\n')
            outfile.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Process oRNAment files so that a unique PWM file in MEME format "
                                                 "is created with all the RBPs in the database.")
    parser.add_argument('--pwm_directory', help='Directory with all the PWM in the database. Filename '
                                                'in the database: "PWMs.tgz"')
    parser.add_argument('--motif_ids', help='File mapping motif IDs to RBP names. '
                                            'File name in the database: "RBP_id_encoding.csv.gz"')

    args = parser.parse_args()

    motifs_ids = "RBP_id_encoding.csv" if not args.motif_ids else args.motif_ids
    pwm_dir = "PWMs" if not args.pwm_directory else args.pwd_directory

    assert os.path.isfile(motifs_ids), "Motif IDs provided is not a valid file."
    assert os.path.isdir(pwm_dir), "PWMs provided is not a valid directory"

    rbp_map = parse_ids(motifs_ids)
    process_PWMs(pwm_dir, rbp_map)


if __name__ == '__main__':
    main()
