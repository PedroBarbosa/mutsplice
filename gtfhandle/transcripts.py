import pandas as pd
from gtfhandle.utils import write_bed_file, remove_overlaps
from gtfhandle.features import Transcripts, Exons
from gtfhandle.utils import *
import sys
from loguru import logger
from typing import Union, Optional, TextIO
import pathlib
import numpy as np
import os


################################################
####### Get transcript sequences to train ######
################################################
class TrainableTranscripts(Transcripts):
    """Representation of transcripts sequences so they can be used to train sequence-based models"""

    def __init__(self, gtf: Union[TextIO, str],
                 gtf_is_processed: bool,
                 out_dir: Union[pathlib.Path, str],
                 fasta: Optional[Union[TextIO, str]] = None,
                 biotype: Optional[list] = None,
                 tx_ids: Optional[TextIO] = None,
                 extract_splice_sites: bool = True,
                 remove_overlapping: bool = False):

        """
        Create an instance of a TrainableTranscript object
        :param TextIO gtf: Path to the GTF file
        :param bool gtf_is_processed: Whether GTF was processed
            before. If `True` `gtf` is a pandas Dataframe.
        :param str out_dir: Path to write output files
        :param str fasta: path to the reference genome in fasta
        :param list biotype: Retrieve transcripts of
            particular biotype(s) (default: Features belonging
            to Protein Coding genes are retrieved)
        :param TextIO tx_ids: File with a list of transcript
            IDs to be retrieved (e.g. most expressed transcript
            per gene in a given tissue). Only transcripts
            belonging to the given `biotype` will be selected
        :param bool extract_splice_sites. Retrieve splice site
            information from multi exon transcripts
        :param bool remove_overlapping: Remove transcripts that overlap
            with another. By default, longest transcript
            that overlaps is kept
        """
        super().__init__(gtf, gtf_is_processed, out_dir, fasta, biotype, tx_ids)

        if remove_overlapping:
            self.transcripts = remove_overlaps(self.transcripts, keep="longest")

        if extract_splice_sites:
            tx_ids = self.get_transcript_ids()
            exons_object = Exons(self.full_biotype_df,
                                 self.gtf_is_processed,
                                 self.out_dir,
                                 gene_biotypes=self.gene_biotypes,
                                 transcript_biotypes=self.tx_biotypes,
                                 transcript_ids=tx_ids)

            ss = exons_object.get_splice_sites()
            n_ss = ss.shape[0]
            ss_duplicates = remove_overlaps(ss, invert=True)
            ss = remove_overlaps(ss, keep=None, ignore_strand=False)

            logger.info("Removing overlapping splice sites.. ({} found)".format(n_ss - ss.shape[0]))

            for i, df in enumerate([ss, ss_duplicates]):
                output = "splice_sites.bed.gz" if i == 0 else "splice_sites_overlapping.bed.gz"
                if df.shape[0] > 0:
                    write_bed_file(df, name=os.path.join(self.out_dir, output),
                                   compression='gzip',
                                   bed6=True,
                                   is_1_based=False,
                                   use_as_name_or_score={'transcript_id': 'Name',
                                                         'exon_number': 'Score'},
                                   additional_fields=['gene_name'])

            self.splice_site_df_to_train, self.per_transcript_ss_idx = self.get_training_data(ss)
            write_fasta_sequences(self.splice_site_df_to_train,
                                  outname=os.path.join(self.out_dir, "transcripts.fasta"),
                                  header_col='transcript_id')
            write_bed_file(self.splice_site_df_to_train,
                           name=os.path.join(self.out_dir, "transcripts.bed"),
                           bed6=True, is_1_based=False)

            self.splice_site_df_to_train.to_pickle(os.path.join(self.out_dir, "trainable_splice_sites_df.pickle"))
            self.per_transcript_ss_idx.to_pickle(os.path.join(self.out_dir, "per_transcript_ss_indexes.pickle"))

    def get_training_data(self, ss):
        """
        Process transcripts and splice site dataframes
        to generate training dataframes with splice site
        classes

        :param pd.Dataframe ss: Splice site coordinates of
            transcripts from `transcripts` dataframe.
            Only transcripts with multiple exons are
            represented. It's inherited from `Transcript`
             object
        :param bool remove_overlaps: Whether transcripts
        that overlap should be removed. If `true` longest
        transcript is kept.

        :return pd.Dataframe: Dataframe with transcripts
            information, including class values for each
            position within a transcript sequence
            (`0` -> not a splice site, `1` -> donor splice site,
             `2` -> acceptor splice site)
        :return pd.Series: List of splice site indexes
            (first position of the dinucleotide) for each
            transcript ID
        """
        pd.options.mode.chained_assignment = None
        logger.info("Retrieving transcripts sequences")
        assert self.fasta is not None, "Fasta file must be provided in order to extract sequence-based data."

        self.transcripts['Fasta'] = self.transcripts.apply(get_fasta_sequences, fasta=self.fasta, axis=1)

        # transcripts start coord
        self.transcripts['start_transcript_wise'] = self.transcripts.apply(
            lambda x: np.where(x['Strand'] == '+', x['Start'], x['End']), axis=1)

        logger.info("Retrieving splice site indexes within each transcript")
        tx_start = self.transcripts.set_index('transcript_id')['start_transcript_wise']

        exon_groups = ss.groupby('transcript_id')
        ss_idx = []

        for name, group in exon_groups:
            ss_idx.append(self._get_ss_idx_by_strand(group, tx_start[name]))
        exons_with_ss_idx = pd.concat(ss_idx)

        per_transcript_start_idx = exons_with_ss_idx.groupby('transcript_id')['ss_idx'].apply(np.hstack)
        logger.info("Merging idx to fasta dataframe and creating class array")

        self.transcripts = self.transcripts.merge(per_transcript_start_idx.rename('ss_idx').to_frame(),
                                                  left_on="transcript_id", right_index=True)

        # create class array filled with zeros
        self.transcripts['Labels'] = self.transcripts['Fasta'].apply(lambda x: np.zeros(len(x), dtype=int))

        logger.info("Assigning class for splicing donors and splicing acceptors")
        # setting class splice donors (1) and splice acceptor (2)
        self.transcripts.apply(self._set_class, axis=1)
        splicing_df = self.transcripts[['Chromosome', 'Start', 'End', 'Strand', 'transcript_id', 'Fasta', 'Labels']]

        return splicing_df.sort_values(['Chromosome', 'Start', 'End']), per_transcript_start_idx

    def _set_class(self, x):
        """
        Sets splice sites indexes within each sequence indexes

        :param pd.Series x: Series with all information about
            each transcript, including fasta sequence and the
            correct order of the splice site indexes
        :return pd.Series: Returns a mutated row where Y column
            contains class values for splice donors (1) and
            splice acceptors (2) at the index where they occur
            within the transcript
        """

        y = np.array([1, 2])
        l_y = np.tile(y, len(x['ss_idx']) // 2)
        np.put(x['Labels'], x['ss_idx'], l_y)
        np.put(x['Labels'], x['ss_idx'] + 1, l_y)

    def _get_ss_idx_by_strand(self, x, tx_start_coord):
        """
        Collects splice site indexes taking into
        account the strand of the transcript

        :param pd.Dataframe x: Dataframe with splice site
            coordinates of all exons of a given transcript
        :param int tx_start_coord: Start coordinate of the transcript
        :return pd.Dataframe: Additional column where indexes
            of splice sites are retrieved in a strand-wise faction
        """
        if x['Strand'].iloc[0] == "+":
            x['ss_idx'] = x['Start'] - tx_start_coord
            return x
        elif x['Strand'].iloc[0] == "-":
            x['ss_idx'] = tx_start_coord - x['End']
            return x[::-1]
