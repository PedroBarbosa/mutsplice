from gtfhandle.utils import bed_is_ok, write_bed_file
import pathlib
from typing import Union, Optional, TextIO
import pandas as pd
from functools import reduce
import pyranges as pr
from operator import attrgetter
from loguru import logger

class Regions(object):
    """
    Create an instance of a Regions object
    """

    def __init__(self, transcripts_df: pd.DataFrame):
        self.transcripts = transcripts_df
        self.tx_groups = transcripts_df.groupby('transcript_id')

    @classmethod
    def map_regions(cls, transcripts_df: pd.DataFrame,
                    no_map_exons: bool = False,
                    no_map_cds: bool = False,
                    no_map_introns: bool = False,
                    no_map_utr5: bool = False,
                    no_map_utr3: bool = False
                    ):
        """
        Map indexes of each region (5'UTR, exons, CDS, intron, 3'UTR)
            within a `Transcripts` dataframe. Default: all regions
            are mapped. Individual regions are defined half-open
            intervals (like bed format, 0-based).
            Coordinates of multiple subregions (multiple exons,
            introns, CDS) come sorted by their position within
            a transcript ([1st exon, 2nd exon, etc]), regardless
            of the transcript strand

        :param pd.Dataframe transcripts_df: Dataframe with full transcripts
        information (subfeatures included)
        :param bool no_map_exons: Do not map coordinates of exonic regions
        :param bool no_map_cds: Do not map coordinantes of CDS regions
        :param bool no_map_introns: Do not map coordinates of intronic regions
        :param bool no_map_utr5: Do not map coordinates of 5'UTR regions
        :param bool no_map_utr3: Do not map coordinates of 3'UTR regions
        :return pd.Series: Additional columns with information providing the
            index positions of each region within each transcript.
        """
        regions = cls(transcripts_df)
        utr5, utr3, exons, cds, introns = (None for i in range(5))

        if no_map_utr5 is False:
            logger.info("Mapping 5'UTRs")
            utr5 = regions.tx_groups.apply(regions.get_5utr_coordinates).dropna()
            logger.info("Number of 5'UTRs: {}".format(utr5.shape[0]))

        if no_map_utr3 is False:
            logger.info("Mapping 3'UTRs")
            utr3 = regions.tx_groups.apply(regions.get_3utr_coordinates).dropna()
            logger.info("Number of 3'UTRs: {}".format(utr3.shape[0]))

        if no_map_exons is False:
            logger.info("Mapping exons")
            exons = regions.tx_groups.apply(regions.get_multiple_subfeature_coordinates).dropna()
            logger.info("Number of exons: {}".format(exons.explode('Intervals_exon').shape[0]))

        if no_map_cds is False:
            logger.info("Mapping CDS")
            cds = regions.tx_groups.apply(regions.get_multiple_subfeature_coordinates, "CDS").dropna()
            logger.info("Number of CDS: {}".format(cds.explode('Intervals_CDS').shape[0]))

        if no_map_introns is False:
            logger.info("Mapping introns")
            introns = regions.get_intron_coordinates(utr5, utr3).dropna()

            # Order intron coordinates by their order within a transcript
            introns['Intervals_intron'] = introns.apply(lambda x: x.Intervals_intron[::-1]
            if x.Strand == "-" else x.Intervals_intron, axis=1)

        to_merge = [x.loc[:, x.columns.str.startswith('Intervals')] for x in [utr5, utr3, exons, introns, cds]
                    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)]
        dfs = [regions.transcripts[regions.transcripts.Feature == "transcript"].set_index('transcript_id')] + to_merge

        if len(dfs) == 2:  # if only 1 region was mapped
            regions.transcripts = pd.merge(regions.transcripts, dfs[1],
                                           left_index=True,
                                           right_index=True,
                                           how='left')
        else:
            regions.transcripts = reduce(lambda left, right: pd.merge(left, right,
                                                                      left_index=True,
                                                                      right_index=True,
                                                                      how='left'), dfs)

        return regions.transcripts.drop(columns=['exon_id', 'exon_number'])

    def get_5utr_coordinates(self, group: pd.DataFrame) -> Union[pd.Series, None]:
        """
        Extracts 5'UTR coordinates and returns
        them as a right-closed pd.Interval

        :param pd.GroupBy group: GroupBy object with
            all features of a single transcript
        :return pd.Series: Series (keyed by transcript ID)
            with Chromosome, pd.Intervals and Strand
            representing 5'UTR coordinates
        """
        strand = group.iloc[0]['Strand']
        chromosome = group.iloc[0]['Chromosome']
        start_codon = group[group.Feature == "start_codon"]

        if start_codon.empty:
            return None

        # multiple frames
        if start_codon.shape[0] > 1:
            start_codon = start_codon.iloc[0]

        if strand == "+":
            utr_start = group[group.Feature == "transcript"]['Start']
            utr_end = start_codon['Start']
            try:
                utr = pd.Interval(left=int(utr_start), right=int(utr_end))
            except ValueError:  # If no UTR
                return None

        else:
            utr_start = group[group.Feature == "transcript"]['End']
            utr_end = start_codon['End']
            try:
                utr = pd.Interval(left=int(utr_end), right=int(utr_start))
            except ValueError:
                return None

        return pd.Series([chromosome, utr, strand], index=["Chromosome", "Intervals_utr5", "Strand"])

    def get_3utr_coordinates(self, group: pd.DataFrame) -> Union[pd.Series, None]:
        """
        Extracts 3'UTR coordinates and returns them
        as a right-closed pd.Interval

        :param pd.GroupBy group: GroupBy object with
            all features of a single transcript
        :return pd.Series: Series (keyed by transcript ID)
            with Chromosome, pd.Intervals and Strand
            representing 3'UTR coordinates
        """

        strand = group.iloc[0]['Strand']
        chromosome = group.iloc[0]['Chromosome']
        stop_codon = group[group.Feature == "stop_codon"]
        # multiple frames
        if stop_codon.shape[0] > 1:
            stop_codon = stop_codon.iloc[0]

        if stop_codon.empty:
            return None

        if strand == "+":
            utr_start = stop_codon['Start']
            utr_end = group[group.Feature == "transcript"]['End']
            utr = pd.Interval(left=int(utr_start), right=int(utr_end))

        else:
            utr_start = stop_codon['End']
            utr_end = group[group.Feature == "transcript"]['Start']
            utr = pd.Interval(left=int(utr_end), right=int(utr_start))
        return pd.Series([chromosome, utr, strand], index=["Chromosome", "Intervals_utr3", "Strand"])

    def get_multiple_subfeature_coordinates(self, group: pd.DataFrame, feature: str = "exon") -> \
            pd.Series:
        """
        Extracts coordinates of subfeatures that may occur
            more than once in a transcript (exon, CDS, intron)
            and returns them as a pandas IntervalArray with
            right-closed pd.Intervals. If feature is `CDS`,
            it includes coordinates of stop_codons as part of
            the feature

        :param pd.DataFrame group: GroupBy object with all
            features of a single transcript
        :param str feature: Subfeature type to extract
            coordinates (default: exon)
        :return pd.Series: Series (keyed by transcript ID) with
            Chromosome, pd.arrays.IntervalArray and Strand where
            each interval represents a `feature` within a transcript
        """
        assert feature in {'exon', 'CDS', 'intron'}, "Ony exon, CDS and intron " \
                                                     "features are to be extracted by " \
                                                     "this method. "

        tmp = group[group.Feature == feature]
        chromosome = group.iloc[0]['Chromosome']
        strand = group.iloc[0]['Strand']
        if strand == "+":
            feat_start = tmp['Start']
            feat_end = tmp['End']
            # if feature == "CDS":
            #    feat_end[feat_end.idxmax()] = feat_end[feat_end.idxmax()]
            result = pd.concat([feat_start, feat_end], axis=1).apply(lambda x: pd.Interval(left=int(x[0]),
                                                                                           right=int(x[1])),
                                                                     axis=1)
        else:
            feat_start = tmp['End']
            feat_end = tmp['Start']
            # if feature == "CDS":
            #    feat_end[feat_end.idxmin()] = feat_end[feat_end.idxmin()]
            result = pd.concat([feat_end, feat_start], axis=1).apply(lambda x: pd.Interval(left=int(x[0]),
                                                                                           right=int(x[1])),
                                                                     axis=1)

        return pd.Series([chromosome, pd.arrays.IntervalArray(result), strand],
                         index=["Chromosome", "Intervals_{}".format(feature), "Strand"])

    def _introns_in_utr(self, single_intron, utr5, utr3):
        """"""
        if utr5 is not None:
            try:
                tx_utr = utr5.loc[single_intron.transcript_id]
                in_utr5 = "True" if (tx_utr.Start < single_intron.End) & (
                        single_intron.Start < tx_utr.End) else "False"
            except KeyError:
                in_utr5 = "no_UTR5"
        else:
            in_utr5 = "False"

        if utr3 is not None:
            try:
                tx_utr = utr3.loc[single_intron.transcript_id]
                in_utr3 = "True" if (tx_utr.Start < single_intron.End) & (
                        single_intron.Start < tx_utr.End) else "False"
            except KeyError:
                in_utr3 = "no_UTR3"
        else:
            in_utr3 = "False"

        return pd.Series([in_utr5, in_utr3])

    def get_intron_coordinates(self, utr5: Optional[pd.DataFrame] = None,
                               utr3: Optional[pd.DataFrame] = None,
                               discard_in_utr: bool = True) -> pd.Series:
        """
        Extracts introns coordinates and returns them
        as a pandas Interval array of right-closed pd.Intervals

        :param pd.Dataframe utr5: Dataframe of 5' UTR
            genomic intervals
        :param pd.Dataframe utr3: Dataframe of 3' UTR
            genomic intervals
        :param bool discard_in_utr: Discard introns
            found within UTR regions of a transcript.
            At least one of the UTR (either `utr5`,
            `utr3` or both) must be set so this flag
            can be properly used. Default: True
        :return pd.Series: Series (keyed by transcript ID)
            with Chromosome, pd.arrays.IntervalArray and
            Strand where each interval represents an intron
            within a transcript
        """
        pyrng = pr.PyRanges(self.transcripts)
        introns = pyrng.features.introns(by="transcript")

        if isinstance(introns, pr.PyRanges) and not introns.empty:
            introns = introns.as_df()
            logger.info("Number of introns: {}".format(introns.shape[0]))

            # introns in UTR's
            if discard_in_utr and any(x is not None for x in [utr5, utr3]):

                if utr5 is not None:
                    utr5['Start'] = utr5.iloc[:, 1].map(attrgetter('left'))
                    utr5['End'] = utr5.iloc[:, 1].map(attrgetter('right'))
                if utr3 is not None:
                    utr3['Start'] = utr3.iloc[:, 1].map(attrgetter('left'))
                    utr3['End'] = utr3.iloc[:, 1].map(attrgetter('right'))
                introns[['is_in_UTR5', 'is_in_UTR3']] = introns.apply(self._introns_in_utr,
                                                                      utr5=utr5,
                                                                      utr3=utr3,
                                                                      axis=1)
                logger.info("introns in UTR5: {}".format((introns.is_in_UTR5 == "True").sum()))
                logger.info("introns in UTR3: {}".format((introns.is_in_UTR3 == "True").sum()))
                introns = introns[(introns.is_in_UTR5 != "True") & (introns.is_in_UTR3 != "True")]
                logger.info(introns.shape)

            return introns.groupby('transcript_id').apply(self.get_multiple_subfeature_coordinates, "intron")


class TrainableRegions(object):
    """Creates region intervals to be fed into deep learning models"""

    def __init__(self, regions: Union[pd.DataFrame, str, TextIO],
                 out_dir: Union[str, pathlib.Path]):
        """
        Create an instance of a TrainableRegions object

        :param Union[pd.DataFrame, str, TextIO] regions: Dataframe
            with `Regions` mapped to each transcript. If a file,
            it's a set of bed intervals ready to be fed into
            deep learning libraries
        :param Union[str, pathlib.Path] out_dir: Output directory

        """
        self.regions = regions
        self.out_dir = out_dir

    def create_regions_intervals(self):
        """
        Creates closed intervals from transcript regions
        according to their Intervals of region coordinates

        :return pd.Dataframe: Bed-like df (with 0-based
            coordinates) with regions within
            transcripts defined
        """
        if not isinstance(self.regions, pd.DataFrame):
            if bed_is_ok(self.regions):
                final_regions = pd.read_csv(self.regions, sep="\t", names=["Chromosome",
                                                                           "Start",
                                                                           "End",
                                                                           "Name",
                                                                           "Score",
                                                                           "Strand"])

                assert len(list(final_regions.dropna(axis=1))) >= 6, "Bed6 is required to create " \
                                                                     "regions intervals."
                return final_regions
        else:
            logger.info("Exploding region intervals into bed-like format")
            pd.options.mode.chained_assignment = None
            _regions_to_encode = ["none"] + [col for col in list(self.regions) if "Intervals" in col]
            # _regions_to_encode = [col for col in list(self.transcripts) if "Intervals" in col]
            labels = {region: i for i, region in enumerate(_regions_to_encode)}
            _df = self.regions.drop(['Feature'], axis=1)
            _df = _df.reset_index()
            final_bed, _tmp = [], []
            for region in _regions_to_encode[1:]:
                _tmp = _df[["Chromosome", "transcript_id", "gene_name", region, "Strand"]].dropna(subset=[region])

                if any(s in region for s in ['intron', 'CDS', 'exon']):
                    _tmp = _tmp.explode(region)

                _tmp['Start'] = _tmp[region].map(attrgetter('left')).astype(int)
                _tmp['End'] = _tmp[region].map(attrgetter('right')).astype(int)
                _tmp['Name'] = region.replace("Intervals_", "") + "_" + _tmp['transcript_id'] + "_" + _tmp['gene_name']
                _tmp['Score'] = labels[region]
                final_bed.append(_tmp[['Chromosome', 'Start', 'End', 'Name', 'Score', 'Strand']])

            final_bed = pd.concat(final_bed).sort_values(by=['Chromosome', 'Start', 'End'])
            # write_bed_file(final_bed, name=os.path.join(self.out_dir, "regions.bed.gz"),
            #                compression='gzip',
            #                is_1_based=False,
            #                bed6=True)
            logger.info("Done")
            return final_bed
