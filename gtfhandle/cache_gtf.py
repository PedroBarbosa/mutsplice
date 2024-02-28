import os
import argparse
import pandas as pd
from gtfhandle.parse_gtf import GTF
from gtfhandle.features import Exons, Introns, Transcripts, extract_surrounding_features


def main():
    parser = argparse.ArgumentParser(description="Parse GTF and cache info on several genomic "
                                                 "features for a quicker access later on.")

    parser.add_argument(dest='gtf', help='Path to the GTF file')
    parser.add_argument(
        dest='fasta', help='Path to the reference genome in Fasta format (uncompressed).')
    parser.add_argument(
        dest='out_dir', help='Path where output files will be written')

    args = parser.parse_args()

    #####################
    #### GTF parsing ####
    #####################

    GTF(args.gtf, False, args.out_dir, "GTF_attributes_expanded.tsv.gz")

    # #####################
    # #### TRANSCRIPTS ####
    # #####################
    INPUT = os.path.join(args.out_dir, "GTF_attributes_expanded.tsv.gz")
    tx_obj = Transcripts(INPUT,
                         gtf_is_processed=True,
                         out_dir=args.out_dir,
                         fasta=args.fasta)

    OUT = os.path.join(args.out_dir, "Transcripts.tsv.gz")
    tx_obj.transcripts.drop(columns=['Source', 'Frame', 'level',
                                     'ont', 'transcript_support_level',
                                     'exon_number', 'exon_id']).to_csv(OUT,
                                                                       sep="\t",
                                                                       compression='gzip',
                                                                       index=False)

    OUT = os.path.join(args.out_dir, "Transcripts_with_subfeatures.tsv.gz")
    tx_obj.transcripts_and_subfeatures.to_csv(OUT,
                                              sep="\t",
                                              compression='gzip',
                                              index=False)

    tx_obj.compute_genomic_attributes(per_subfeature=True)
    SUBFEATURES = tx_obj.explode_transcripts_subfeatures(extra_features=True)
    OUT = os.path.join(args.out_dir, "Transcripts_subfeatures_exploded.tsv.gz")
    SUBFEATURES.to_csv(OUT, sep="\t", compression='gzip', index=False)

    #################
    ##### Exons #####
    #################
    # SUBFEATURES = pd.read_csv(os.path.join(args.out_dir, "Transcripts_subfeatures_exploded.tsv.gz"), sep="\t")
    # TX_AND_SUBFT = pd.read_csv(os.path.join(args.out_dir, "Transcripts_with_subfeatures.tsv.gz"), sep="\t")

    exons_obj = Exons(tx_obj.transcripts_and_subfeatures,
                      gtf_is_processed=True,
                      out_dir=args.out_dir,
                      fasta=args.fasta)

    for level in [0, 1, 2]:
        EXONS_INFO = extract_surrounding_features(exons_obj.exons,
                                                  SUBFEATURES,
                                                  level=level)
        OUT = os.path.join(
            args.out_dir, "Exons_level_{}.tsv.gz".format(str(level)))
        EXONS_INFO.drop('index').to_csv(OUT, sep="\t", compression='gzip', index=False)

    #################
    #### Introns ####
    #################
    introns_obj = Introns(tx_obj.transcripts_and_subfeatures,
                          gtf_is_processed=True,
                          out_dir=args.out_dir,
                          fasta=args.fasta)

    for level in [0, 1, 2]:
        INTRONS_INFO = extract_surrounding_features(introns_obj.introns,
                                                    SUBFEATURES,
                                                    level=level,
                                                    what_is_under_study="introns")
        OUT = os.path.join(
            args.out_dir, "Introns_level_{}.tsv.gz".format(str(level)))
        INTRONS_INFO.to_csv(OUT, sep="\t", compression='gzip', index=False)


if __name__ == "__main__":
    main()
