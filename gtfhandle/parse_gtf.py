import pandas as pd
import numpy as np
from typing import Union, Optional, TextIO, List
import pathlib
import re
from gtfhandle.utils import remove_version_from_ensembl_ids, filter_redundancy_in_sexual_chromosomes
from collections import defaultdict
import sys
from loguru import logger
import argparse
import os


class GTF(object):
    """Representation of a gtf annotation file"""

    def __init__(self, gtf: Union[str, TextIO],
                 gtf_is_processed: bool,
                 out_dir: Union[str, pathlib.Path],
                 out_file: Optional[str] = None):
        """
        Creates a processed dataframe from a GTF file

        :param str gtf: path to the GTF file (or pd.DataFrame
            if `is_df_ready` is `True`)
        :param bool gtf_is_processed: whether `gtf_file`
            is a pyranges dataframe
        :param str out_dir: path to the output directory
            where all file will be written.
        :param str out_file: path to the output directory
            (optional). Default: Df is not written
        """
        self.gtf = gtf
        self.gtf_is_processed = gtf_is_processed
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self.out_dir = out_dir
        self.out_file = out_file

        if self.gtf_is_processed:
            df = pd.read_csv(self.gtf, low_memory=False)
            logger.info("Df loaded.")

            assert all(col in df.columns for col in ['Chromosome',
                                                     'Start', 'End',
                                                     'gene_id', 'transcript_id']), "Dataframe was not properly " \
                                                                                   "loaded. Are you sure that your " \
                                                                                   "GTF file was processed before ?"

        else:
            logger.info("Reading GTF.")
            # Can't use pyranges to read GTF because it discards
            # some attributes when there are multiple tags
            # for a given transcript (those tags are used to
            # rank transcripts)
            df = self.parse_gtf()

            remove_version_from_ensembl_ids(df)
            filter_redundancy_in_sexual_chromosomes(df)

            if out_file is not None:
                logger.info("Writing processed object to file.")
                df.to_csv(os.path.join(out_dir, out_file), index=False, compression='gzip')

        self.full_df = df
        self.available_gene_biotypes = self.full_df[self.full_df.Feature == 'gene']['gene_type'].unique()
        self.available_tx_biotypes = self.full_df[self.full_df.Feature == 'transcript']['transcript_type'].unique()

    ###################
    #### Parse GTF ####
    ###################
    def parse_gtf(self):
        """
        Parses a Gencode GTF file and creates
        a pd.Dataframe with a set of target attributes.
        Start coordinates are transformed so that a
        0-based (bed like) system is used for all
        the data  structures.

        :return pd.DataFrame df: Df with extended
            attributes represented in a tabular
            format.
        """
        gtf_cols = [
            "Chromosome",
            "Source",
            "Feature",
            "Start",
            "End",
            "Score",
            "Strand",
            "Frame",
            "attribute",
        ]
        try:
            df = pd.read_csv(
                self.gtf,
                sep="\t",
                comment="#",
                names=gtf_cols,
                skipinitialspace=True,
                engine="c",
                dtype={
                    "Start": np.int64,
                    "End": np.int64,
                    "Score": np.float32,
                    "Chromosome": str,
                },
                na_values=".")

        except ValueError:
            raise ValueError("Is the GTF file provided already processed? "
                             "If so, please set '--gtf_is_processed' flag.")
        logger.info("Done")
        df.Start -= 1
        return self.get_extend_attributes(df)

    def get_extend_attributes(self, df):
        """
        Adds additional columns to the dataframe
        present in the 'attribute' field of the gtf

        :param pd.DataFrame df: Df obtained by
            reading a GTF file
        :return pd.DataFrame: df with attribute
            field extended so sub attributes are
            stored in individual columns
        """

        logger.info("Getting additional gtf attributes.")
        additional_cols = [
            'gene_id',
            'gene_name',
            'gene_type',
            'level',
            'ont',
            'tag',
            'transcript_id',
            'transcript_name',
            'transcript_support_level',
            'transcript_type',
            'exon_id',
            'exon_number'
        ]

        logger.info("Applying function on each row to get additional gtf attributes..")
        df_new = df['attribute'].apply(self.extend_attributes, args=(additional_cols,)).to_frame()
        logger.info("Done")
        df_with_attributes = pd.DataFrame(df_new['attribute'].values.tolist(), index=df.index, columns=additional_cols)
        df_final = pd.concat([df.drop(['attribute'], axis=1), df_with_attributes], axis=1, sort=False)
        logger.info("Done. Whole gtf processed")
        return df_final

    def extend_attributes(self, x, gtf_attributes):
        """
        Adds additional attributes to the dataframe

        :param pd.Series x: Single row of whole GTF dataframe
        :param list gtf_attributes: List of additional attributes to add
        :return str: Values of `gtf_attributes` for the given row `x`, split by ";"
        """
        y = x.replace("\"", "").rstrip()
        R_SEMICOLON = re.compile(r'\s*;\s*')
        fields = [i for i in re.split(R_SEMICOLON, y) if i.strip()]
        attr = defaultdict(list)
        for f in fields:
            attr[re.split('\s+', f)[0]].append(re.split('\s+', f)[1])
        return [';'.join(attr[col]) if col in attr.keys() else np.nan for col in gtf_attributes]


def main():
    parser = argparse.ArgumentParser(description="Utilities to parse gtf files.")
    parser.add_argument(dest='gtf', help='Path to the GTF file')
    parser.add_argument(dest='out_dir', help='Path where output files will be written')
    parser.add_argument('--out_file', help='Filename to write the processed dataframe. If set, df will be compressed')
    parser.add_argument('--gtf_is_processed', action='store_true',
                        help='If set \'--gtf\' argument represents the processed'
                             'daraframe from an original gtf file.')

    args = parser.parse_args()
    if args.gtf_is_processed and args.out_file:
        raise ValueError("'--out_file' can't be set when GTF is already processed ('--gtf_is_processed' was set True)")
    GTF(args.gtf, args.gtf_is_processed, args.out_dir, args.out_file)


if __name__ == "__main__":
    main()
