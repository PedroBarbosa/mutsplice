import os
from typing import Union, Optional, TextIO, List
import pathlib
from pyfaidx import Fasta
import pandas as pd
import itertools
from functools import reduce
from tqdm import tqdm
from loguru import logger
import random
from multiprocessing import cpu_count, Pool
pd.options.mode.chained_assignment = None
import numpy as np
from operator import attrgetter
import pyranges as pr
from pyranges import PyRanges
from gtfhandle.parse_gtf import GTF
from gtfhandle.regions import Regions, TrainableRegions
from gtfhandle.utils import read_features_file, open_fasta, get_fasta_sequences, compute_sequence_based_features


class Features(GTF):
    """
    Representation of a genomic feature
    Inherits from `GTF` class
    """
    def __init__(self,
                 gtf: Union[str, TextIO, pd.DataFrame],
                 gtf_is_processed: bool,
                 out_dir: Union[str, pathlib.Path],
                 fasta: Optional[Union[str, TextIO, Fasta]] = None,
                 gene_biotypes: Optional[list] = None,
                 transcript_biotypes: Optional[list] = None,
                 gene_ids: Optional[Union[TextIO, str, List]] = None,
                 gene_names: Optional[Union[TextIO, str, List]] = None):
        """
        Creates an instance of a Feature object
        :param Union[str, TextIO, pd.DataFrame gtf]: Path to the GTF file or to the
        directory where the cache is located.
        :param bool is_gtf_ready: Whether GTF was processed before. If `True` `gtf` is a pandas Dataframe.
        :param pathlib.Path out_dir: Path to write output files
        :param TextIO fasta: path to the reference genome in fasta
        :param list gene_biotypes: Retrieve features whose their parent
        gene belongs to particular gene biotype(s)
        :param list transcript_biotypes: Retrieve features whose their
        parent transcript belongs to particular transcript biotypes(s)
        :param Optional[Union[TextIO, str, List]] gene_ids:
            Only features belonging to these gene_ids will be used
        :param Optional[Union[TextIO, str, List]] gene_names:
            Only features belonging to these gene_names will be used
        """
        self.gene_biotypes = gene_biotypes
        self.tx_biotypes = transcript_biotypes
        assert not all(x for x in [gene_ids, gene_names]), "Please set just one of the filters " \
                                                           "(--gene_ids or --gene_names)"
        if isinstance(gtf, pd.DataFrame):
            self.full_biotype_df = gtf

            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            self.out_dir = out_dir

        # If Cache exists
        elif os.path.isdir(gtf):
            super().__init__(gtf, gtf_is_processed, out_dir)
            self.full_biotype_df = self._filter_biotypes()

        # If GTF
        else:
            if gtf_is_processed:
                super().__init__(gtf, gtf_is_processed, out_dir)
            else:
                super().__init__(gtf,
                                 gtf_is_processed,
                                 out_dir,
                                 out_file="processed_gtf.csv.gz")

            logger.info('Number of genes in GTF: {}'.format(
                self.full_df.gene_id.nunique()))
            self.full_biotype_df = self._filter_biotypes()

        if gene_names:
            if not isinstance(gene_names, list):
                gene_names = read_features_file(gene_names)
            self.full_biotype_df = self.full_biotype_df[
                self.full_biotype_df.gene_name.isin(gene_names)]

        if gene_ids:
            if not isinstance(gene_ids, list):
                gene_ids = read_features_file(gene_ids)
            self.full_biotype_df = self.full_biotype_df[
                self.full_biotype_df.gene_id.isin(gene_ids)]

        self.fasta = open_fasta(fasta)
        self.possible_hexamers = [
            ''.join(x) for x in list(itertools.product('ATCG', repeat=6))
        ]

    def _return_neighbour_features_old(self,
                                       neighbours: pd.DataFrame,
                                       from_where: str,
                                       unique_id: str,
                                       is_variant: bool = False):
        """
        Get genomic attributes of a surrounding
        (upstream or downstream) feature.
        :param pd.Series neighbours: Single
        row displaying the nearest genomic interval
        of a given target feature
        :param str from_where: From which genomic location
        the `neighbour_features` refer. Default: `upstream`.
        :param str unique_id: Unique target feature ID to
        serve as key to the output
        :param bool is_variant: Whether target features
        correspond to genomic intervals where a variant
        lies. Default: `False`. Expects that the target
        feature is an exon or intron.
        :return pd.Series: Neighbour genomic attributes
        to be merged by `unique_id`
        """
        assert from_where in [
            'upstream', 'downstream'
        ], "Wrong value provided in the 'from_where' argument"

        cols = ['gc_', 'len_', 'cg_dinuc_', 'hexamers_']
        if is_variant:
            cols.append('distance_ss_')

        cols = [s + from_where for s in cols]
        cols.extend([unique_id, 'number_' + from_where])

        # 1) If not adjacent, it is a first or last exon
        # Nearest interval is from different gene
        # 2) If it is adjacent and it's not from the same
        # gene
        if (neighbours.Distance != 1 and is_variant is False) or \
                (neighbours.Distance == 1 and
                 neighbours.gene_id != neighbours.gene_id_b and
                 is_variant is False):

            return pd.Series([
                np.nan, np.nan, np.nan, np.nan, neighbours[unique_id], np.nan
            ],
                             index=cols)

        else:
            assert neighbours.transcript_id == \
                   neighbours.transcript_id_b, "Neighbour feature relates " \
                                               "to a different transcript.\n" \
                                               "Target feature: {}\n" \
                                               "Neighbour feature: {}".format(neighbours.transcript_id,
                                                                              neighbours.transcript_id_b)
            # if the target feature is an intron
            if neighbours.Feature_b == "exon":
                if from_where == "upstream":
                    number = neighbours.number
                else:
                    number = int(neighbours.number) - 1
            else:
                number = neighbours.number

            if is_variant:

                return pd.Series([
                    neighbours.gc_b, neighbours.length_b,
                    neighbours.cg_dinuc_b, neighbours.hexamers_b,
                    neighbours.Distance, neighbours[unique_id], number
                ],
                                 index=cols)
            else:
                return pd.Series([
                    neighbours.gc_b, neighbours.length_b,
                    neighbours.cg_dinuc_b, neighbours.hexamers_b,
                    neighbours[unique_id], number
                ],
                                 index=cols)

    def _filter_biotypes(self):
        """
        Filters dataframe to include
        only features from GTF that
        belong to the given gene and
        transcript biotypes

        :return pd.DataFrame: Filtered df
        """
        full_biotype_df = self.full_df.copy()
        if self.gene_biotypes:
            assert all(_type in self.available_gene_biotypes
                       for _type in self.gene_biotypes), "Some biotype(s) provided do " \
                                                         "not match the list of possible " \
                                                         "gene biotypes: \n{}".format(self.available_gene_biotypes)

            full_biotype_df = full_biotype_df[full_biotype_df.gene_type.isin(
                self.gene_biotypes)]

        if self.tx_biotypes:
            assert all(_type in self.available_tx_biotypes
                       for _type in self.tx_biotypes), 'Some biotype(s) provided do ' \
                                                       'not match the list of possible ' \
                                                       'transcript biotypes: \n{}'.format(self.available_tx_biotypes)

            full_biotype_df = full_biotype_df[
                full_biotype_df.transcript_type.isin(self.tx_biotypes)]

        logger.info(
            "Number of genes (and transcripts) with '{}' gene biotype(s) "
            "and '{}' transcript biotype (s): {} ({})".format(
                self.gene_biotypes, self.tx_biotypes,
                full_biotype_df.gene_id.nunique(),
                full_biotype_df.transcript_id.nunique()))
        return full_biotype_df


class Transcripts(Features):
    """
    Representation of transcripts annotations.
    Inherits from `Feature` class and coordinates
    are 0-based.
    """
    def __init__(self,
                 gtf: Union[str, TextIO],
                 gtf_is_processed: bool,
                 out_dir: Union[str, pathlib.Path],
                 fasta: Optional[Union[str, TextIO, Fasta]] = None,
                 gene_biotypes: Optional[list] = None,
                 transcript_biotypes: Optional[list] = None,
                 select_top: Optional[bool] = False,
                 gene_ids: Optional[Union[TextIO, str, List]] = None,
                 gene_names: Optional[Union[TextIO, str, List]] = None,
                 transcript_ids: Optional[Union[TextIO, str, List]] = None):
        """
        Create an instance of a Transcripts object
        :param TextIO gtf: Path to the GTF file or to the directory where
        the cache is located.
        :param bool gtf_is_processed: Whether GTF
         was processed before. If `True` `gtf` is a file
         that can be loaded as a pandas Dataframe.
        :param str out_dir: Path to write output files
        :param str fasta: path to the reference genome in fasta
        :param list gene_biotypes: Retrieve transcripts whose
            parent gene belongs to the given biotype(s)
        :param list transcript_biotypes: Retrieve transcripts
            of particular transcript biotype(s)
        :param bool select_top: Selects a top transcript per gene,
            when multiple transcripts exist for the same gene
            (be it of the same biotype or not) and when `transcript_ids`
            is not provided. Default: `True`. If `False`, all the
            transcripts are kept
        :param Optional[Union[TextIO, str, List]] gene_ids:
            File with a list of gene IDs to be retrieved.
            Only transcripts belonging to the given `biotype`
            and `gene_ids` will be selected. If provided, it
            overrides the  `select_top` argument
        :param Optional[Union[TextIO, str, List]] gene_names:
            File with a list of gene names to be retrieved.
            Only transcripts belonging to the given `biotype`
            and `gene_names` will be selected. If provided,
            it overrides the `select_top` argument
        :param Optional[Union[TextIO, str, List]] transcript_ids:
            File with a list of transcript IDs to be retrieved
            (e.g. most expressed transcript per gene in a given tissue).
            Only transcripts belonging to the given `biotype`
            will be selected. If provided, it overrides the
            `select_top` argument
        """
        super().__init__(gtf,
                         gtf_is_processed,
                         out_dir,
                         fasta=fasta,
                         gene_biotypes=gene_biotypes,
                         transcript_biotypes=transcript_biotypes,
                         gene_ids=gene_ids,
                         gene_names=gene_names)

        cols = [
            "Chromosome", "Feature", "Start", "End", "Strand", "transcript_id",
            "gene_id", "gene_name", "transcript_type"
        ]

        if transcript_ids is None or isinstance(transcript_ids, list):
            self.transcript_ids = transcript_ids
        else:
            self.transcript_ids = read_features_file(transcript_ids)

        full_tx_df = self.full_biotype_df[self.full_biotype_df.Feature ==
                                          "transcript"]
        self.select_top = select_top
        self.transcripts = self.select_transcripts(full_tx_df)

        if self.transcripts.shape[0] == 0:
            raise ValueError("No transcripts were kept after filtering. "
                             "Please check the GTF or the transcripts IDs "
                             "provided.")

        tx_subft = self.full_biotype_df[
            self.full_biotype_df.transcript_id.isin(
                self.transcripts.transcript_id)][cols +
                                                 ['exon_id', 'exon_number']]

        self.transcripts_and_subfeatures = pd.merge(
            tx_subft,
            self.transcripts[['transcript_id', 'rank_score']],
            on='transcript_id')

        logger.info("Number of selected transcripts: {}".format(
            self.transcripts.shape[0]))

    def select_transcripts(self, full_tx_df: pd.DataFrame):
        """
        Filters transcripts based on a list of IDs or
        returns 1 transcript per gene based on a simple
        ranking procedure (from a set of attributes that
        the transcript may have).

        :param pd.Dataframe full_tx_df: Dataframe with all the
            transcripts from genes of provided biotype (s)
        :return: pd.Dataframe: Dataframe with filtered transcripts
        """

        if self.transcript_ids is not None:
            logger.info(
                "Selecting the transcripts based on the IDs provided.")
            full_tx_df = full_tx_df[full_tx_df['transcript_id'].isin(
                self.transcript_ids)]

        if self.select_top:
            logger.info(
                "Ranking transcripts per gene and selecting the representative one."
            )
            ranked = full_tx_df.groupby('gene_id').apply(self.rank_transcripts,
                                                         return_top=True)

            logger.info(
                "Number of selected transcripts (in theory, 1 per gene) may be lower than "
                "the number of genes because all transcripts of some genes may lack the "
                "tx biotype(s) {} provided:\t{}".format(
                    self.tx_biotypes, ranked.shape[0]))
            return ranked.reset_index(drop=True)

        logger.info("Ranking transcripts per gene.")
        return full_tx_df.groupby('gene_id').apply(
            self.rank_transcripts).reset_index(drop=True)

    def rank_transcripts(self,
                         group: pd.DataFrame.groupby,
                         return_top: bool = False):
        """
        Ranks transcripts per gene based on a set of
        attributes that the transcript may have

        :param pd.DataFrame.groupby group: groupBy object
            with information about each transcript of a
            single gene
        :param bool return_top: Returns top-ranked
            transcript of each gene. Default: `False`
        """
        ranks = {
            'MANE_Select': 12,
            'CCDS': 11,
            'appris_principal_1': 10,
            'appris_principal_2': 9,
            'appris_principal_3': 8,
            'appris_principal_4': 7,
            'appris_principal_5': 6,
            'basic': 5,
            'appris_alternative_1': 4,
            'appris_candidate': 3,
            'TAGENE': 2,
            'RNA_Seq_supported_only': 1,
            'CAGE_supported_TSS': 1
        }

        group_ = group[group['tag'].notna()]

        # If cached, tx were already ranked before
        if return_top and "rank_score" in group.columns:
            max_rank = group_[group_['rank_score'] ==
                              group_['rank_score'].max()]
            return group_.loc[(max_rank['End'] - max_rank['Start']).idxmax()]

        if group_.shape[0] == 0:
            logger.info("Transcripts of gene {} have no tag.".format(
                group.name))

            if return_top:
                group_ = group.sample(1)
                group_['rank_score'] = 1
                return group_.iloc[0]

            else:
                group_['rank_score'] = -1
                i = random.randint(0, group_.shape[0])
                group_.iloc[i, 'rank_score'] = 1
                return group_

        else:

            group_['rank_score'] = group_['tag'].apply(
                lambda x: sum([ranks.get(field, 0) for field in x.split(";")]))
            # lambda x: sum(list(filter(None.__ne__, list(map(ranks.get, x))))))

            if return_top:
                max_rank = group_[group_['rank_score'] ==
                                  group_['rank_score'].max()]
                return group_.loc[(max_rank['End'] -
                                   max_rank['Start']).idxmax()]

            else:
                _no_tag = group[group['tag'].isna()]
                if _no_tag.shape[0] > 0:
                    _no_tag['rank_score'] = -1
                    return pd.concat([group_, _no_tag])
                else:
                    return group_

    @property
    def ids(self):
        """
        Returns a list of transcript IDs
        present in the `transcripts` dataframe

        :return list: List with the unique
            transcript IDs found in df
        return self.tx_ids
        """
        return list(set(self.transcripts.transcript_id))

    @property
    def count_tx_biotypes(self):
        """
        Returns the number of transcripts
        on each biotype class

        :return dict: Dict with the number
        of transcript per biotype class
        """
        return self.transcripts['transcript_type'].value_counts().to_dict()

    @property
    def count_tx_and_gene_biotypes(self):
        """
        Maps the number of each biotype class on
        gene and transcript features by writing
        the count matrix to a file
        """
        pd.crosstab(self.transcripts.transcript_type,
                    self.transcripts.gene_type).to_csv(os.path.join(
                        self.out_dir, "gene_tx_biotype_count.csv"),
                                                       sep="\t")

    def compute_genomic_attributes(self,
                                   per_subfeature: bool = False,
                                   just_get_intervals: bool = False):
        """
        Computes the gc content of each transcript
        and stores the results in a column called
        `gc_content`

        :param bool per_subfeature: Calculate attributes
            for each exon/intron of a transcript. If set
            to True, an instance of the `Regions` class
            will be created to map these subfeatures
        :param bool just_get_intervals: When per_subfeature is
            `True` just return the intervals rather than
            computing feature values
        """

        assert self.fasta is not None, "Fasta file is required to calculate gc content"

        if per_subfeature:
            logger.info("Mapping subregions within transcripts.")
            subfeature_intervals = Regions.map_regions(
                self.transcripts_and_subfeatures,
                no_map_utr5=True,
                no_map_utr3=True,
                no_map_cds=True)

            # subfeature_intervals.to_csv(os.path.join(self.out_dir, 'Transcripts_with_regions.tsv.gz'),
            #                             sep="\t", index=False)
            # subfeature_intervals = pd.read_csv(os.path.join(self.out_dir, 'Transcripts_with_regions.tsv.gz'),
            #                                   sep="\t", )
            if just_get_intervals:

                self.transcripts = pd.merge(self.transcripts,
                                            subfeature_intervals[[
                                                "Intervals_exon",
                                                "Intervals_intron"
                                            ]],
                                            left_on='transcript_id',
                                            right_index=True)

            else:
                regions = TrainableRegions(
                    subfeature_intervals,
                    self.out_dir).create_regions_intervals()
                # regions = pd.read_csv(os.path.join(self.out_dir, "regions.bed.gz"),
                #                        names=["Chromosome", "Start", "End",
                #                               "Name", "Score", "Strand"], sep="\t")

                logger.info(
                    "Computing genomic attributes of all subfeatures in all transcripts"
                )

                regions[['subfeature', 'transcript_id',
                         'gene_name']] = regions['Name'].str.split('_',
                                                                   n=2,
                                                                   expand=True)

                grouped = regions.groupby('transcript_id')
                # gc_exons_dict, gc_introns_dict, length_exons_dict, length_introns_dict, \
                # cg_dinuc_exons_dict, hexamers_exons_dict, cg_dinuc_introns_dict, hexamers_introns_dict = [{} for i in
                #                                                                                           range(8)]
                gc_exons_dict, gc_introns_dict, length_exons_dict, length_introns_dict = [
                    {} for i in range(4)
                ]
                for tx, subfeat in grouped:
                    strand = subfeat.iloc[0].Strand

                    exons = subfeat[subfeat.subfeature == "exon"]
                    length_exons = compute_length(exons)['length'].values.tolist()
                    out = compute_gc_and_hexamers(exons, self.fasta)[['gc', 'cg_dinuc', 'hexamers']]
                    gc_exons = out['gc'].values.tolist()
                    # cg_dinuc_exons = out['cg_dinuc'].values.tolist()
                    # hexamers_exons = out['hexamers'].values.tolist()

                    if strand == "-":
                        length_exons = length_exons[::-1]
                        gc_exons = gc_exons[::-1]
                        # cg_dinuc_exons = cg_dinuc_exons[::-1]
                        # hexamers_exons = hexamers_exons[::-1]
                    length_exons_dict[tx] = length_exons
                    gc_exons_dict[tx] = gc_exons
                    # cg_dinuc_exons_dict[tx] = cg_dinuc_exons
                    # hexamers_exons_dict[tx] = hexamers_exons

                    introns = subfeat[subfeat.subfeature == "intron"]
                    if not introns.empty:
                        length_introns = compute_length(
                            introns)['length'].values.tolist()
                        out = compute_gc_and_hexamers(introns, self.fasta)[[
                            'gc', 'cg_dinuc', 'hexamers'
                        ]]
                        gc_introns = out['gc'].values.tolist()
                        # cg_dinuc_introns = out['cg_dinuc'].values.tolist()
                        # hexamers_introns = out['hexamers'].values.tolist()
                        if strand == "-":
                            length_introns = length_introns[::-1]
                            gc_introns = gc_introns[::-1]
                            # cg_dinuc_introns = cg_dinuc_introns[::-1]
                            # hexamers_introns = hexamers_introns[::-1]
                        length_introns_dict[tx] = length_introns
                        gc_introns_dict[tx] = gc_introns
                        # cg_dinuc_introns_dict[tx] = cg_dinuc_introns
                        # hexamers_introns_dict[tx] = hexamers_introns

                # _l = [gc_exons_dict, gc_introns_dict, length_exons_dict, length_introns_dict,
                #      cg_dinuc_exons_dict, cg_dinuc_introns_dict, hexamers_exons_dict, hexamers_introns_dict]
                _l = [
                    gc_exons_dict, gc_introns_dict, length_exons_dict,
                    length_introns_dict
                ]
                _attrib = pd.DataFrame.from_records(_l).T.rename(
                    {
                        0: 'gc_exon',
                        1: 'gc_intron',
                        2: 'length_exon',
                        3: 'length_intron',
                        4: 'cg_dinuc_exon',
                        5: 'cg_dinuc_intron',
                        6: 'hexamers_exon',
                        7: 'hexamers_intron'
                    },
                    axis=1)

                self.transcripts = pd.merge(
                    self.transcripts,
                    subfeature_intervals[[
                        "Intervals_exon", "Intervals_intron"
                    ]],
                    left_on='transcript_id',
                    right_index=True).merge(_attrib,
                                            left_on='transcript_id',
                                            right_index=True)

        else:
            compute_gc_and_hexamers(self.transcripts, self.fasta)
            compute_length(self.transcripts)

        logger.info("Done.")

    def explode_transcripts_subfeatures(self, extra_features: bool = False):
        """
        Explode transcripts df so that subfeatures properties
        are represented in a row-wise faction. Requires that
        `compute_genomic_attributes` method has been previously
        run
        
        :param bool extra_features: Return additional columns
        representing genome architecture features (e.g. GC, length)
 
        :return pd.DataFrame: Much larger df with exon/intron
        information with closed intervals (1-based coordinates)
        """
        logger.info("Exploding transcripts subfeatures")
        assert 'Intervals_exon' and 'Intervals_intron' in self.transcripts.columns, "Subfeature information doesn't" \
                                                                                    "exist in the transcript df"

        cols_to_keep = [
            'Chromosome', 'Strand', 'transcript_id', 'transcript_type',
            'gene_id', 'gene_name', 'rank_score'
        ]
        final_cols_ordered = [
            'Chromosome', 'Start', 'End', 'transcript_id', 'transcript_type',
            'gene_name', 'Strand', 'gene_id', 'number', 'rank_score'
        ]

        if extra_features:
            assert all(
                x in self.transcripts.columns for x in
                ['gc_exon', 'gc_intron', 'length_exon', 'length_intron']
            ), "If extra_features is set, previous 'compute_genomic_attributes' function must be run with just_get_interval=False"

            final_cols_ordered.extend(['gc', 'length'])

        _df = self.transcripts.copy()

        exploded_dfs = []

        for f in ["exon", "intron"]:

            to_explode_cols = ["Intervals_" + f]
            if extra_features:

                gc_col = "gc_" + f
                length_col = "length_" + f
                to_explode_cols.extend([gc_col, length_col])
                to_rename = {gc_col: 'gc', length_col: 'length'}

            else:
                to_rename = {}

            #cols = cols_to_keep + to_explode_cols

            # Removes intron rows that are NA (single exon transcripts)
            _df = _df.dropna(subset=['Intervals_' + f])

            _df_exploded = _df.explode(to_explode_cols)
            _df_exploded['number'] = _df_exploded.groupby(
                _df_exploded.index).cumcount() + 1

            idx = pd.IntervalIndex(_df_exploded["Intervals_" + f])
            _df_exploded['Start'] = idx.left
            _df_exploded['End'] = idx.right

            _df_exploded = _df_exploded.rename(
                columns=to_rename)[final_cols_ordered]

            _df_exploded['Feature'] = f

            exploded_dfs.append(_df_exploded)

        final = pd.concat(exploded_dfs)
        logger.info("Done.")
        return final.sort_values(['Chromosome', 'Start', 'End'])


class Exons(Features):
    """
    Representation of exon annotations.
    Inherits from `Features` class and
    coordinates are 0-based.
    """
    def __init__(
        self,
        gtf: Union[str, TextIO],
        gtf_is_processed: bool,
        out_dir: Union[str, pathlib.Path],
        fasta: Optional[Union[str, TextIO]] = None,
        gene_biotypes: Optional[list] = None,
        transcript_biotypes: Optional[list] = None,
        gene_ids: Optional[Union[TextIO, str, List]] = None,
        gene_names: Optional[Union[TextIO, str, List]] = None,
        transcript_ids: Optional[Union[list, TextIO]] = None,
        exon_ids: Optional[TextIO] = None,
        exon_coordinates: Optional[Union[pd.DataFrame, PyRanges]] = None,
    ):
        """
        Create an instance of a Exon object.
        Coordinates refer to 0-based half-open intervals
        :param TextIO gtf: Path to the GTF file or
        :param bool gtf_is_processed: Whether GTF
         was processed before. If `True` `gtf` is a file
         that can be loaded as a pandas Dataframe.
        :param str out_dir: Path to write output files
        :param str fasta: path to the reference genome in fasta
        :param list gene_biotypes: Retrieve exons whose
            parent gene belongs to the given biotype(s)
        :param list transcript_biotypes: Retrieve exons whose
            parent transcript belongs to the given biotype(s)
        :param Optional[Union[TextIO, str, List]] gene_ids:
            File with a list of gene IDs to be retrieved.
            Only exons belonging to the given `biotype`
            and `gene_ids` will be selected. If provided, it
            overrides the `select_top` argument
        :param Optional[Union[TextIO, str, List]] gene_names:
            File with a list of gene names to be retrieved.
            Only exons belonging to the given `biotype`
            and `gene_names` will be selected. If provided,
            it overrides the `select_top` argument
        :param Optional[list, TextIO] transcript_ids: Transcript IDs to
        select exons from. Can be a list provided from a `Transcripts` object,
        or a file with transcript IDs, just like the `exon_ids` argument
        :param TextIO exon_ids: File with a list of exon IDs
        to be retrieved. Only transcripts belonging to the
        given `biotype` will be selected
        :param Optional[pd.DataFrame, PyRanges] exon_coordinates: Df with
        exon coordinates to be retrieved.
        """
        cols = [
            "Chromosome", "Feature", "Start", "End", "Strand", "gene_id",
            "gene_name", "transcript_id", "transcript_type", "exon_id",
            "exon_number", "rank_score"
        ]

        super().__init__(gtf,
                         gtf_is_processed,
                         out_dir,
                         fasta=fasta,
                         gene_biotypes=gene_biotypes,
                         transcript_biotypes=transcript_biotypes,
                         gene_ids=gene_ids,
                         gene_names=gene_names)

        self.exon_ids = read_features_file(
            exon_ids) if exon_ids is not None else exon_ids
        self.exon_coordinates = exon_coordinates if exon_coordinates is not None else exon_coordinates

        if isinstance(transcript_ids, list) or transcript_ids is None:
            self.transcript_ids = transcript_ids
        else:
            self.transcript_ids = read_features_file(transcript_ids)

        full_exon_df = self.full_biotype_df[self.full_biotype_df.Feature ==
                                            "exon"][cols]

        if self.exon_ids or self.transcript_ids:
            self.exons = self.select_exons_by_ID(full_exon_df)
        else:
            self.exons = full_exon_df

        if self.exon_coordinates is not None:
            self.exons, self.absent_in_gtf = self.select_exons_by_coordinates()

            for c in list(self.exon_coordinates):
                if c not in cols:
                    cols.append(c)

        self.exons = self.exons[cols].sort_values(
            ['Chromosome', 'Start', 'End'])

    def select_exons_by_coordinates(self, select_top: bool = True):
        """
        Retrieves exons matching the coordinates
          represented in a list of exons

        :param bool select_top: Retrieve 1 match per exon.
        Exon will be selected based on the rank score of
        the transcript that each exon is associated

        :return: pd.Dataframe: Dataframe with filtered exons
        """
        cols = ['Chromosome', 'Start', 'End']

        for c in [
                'Strand', 'gene_name', 'gene_id', 'transcript_id',
                'transcript_type'
        ]:
            if c in self.exon_coordinates.columns:
                cols.append(c)

        if isinstance(self.exon_coordinates, PyRanges):
            self.exon_coordinates = self.exon_coordinates.as_df()

        df = pd.merge(self.exons,
                      self.exon_coordinates,
                      left_on=cols,
                      right_on=cols)

        if select_top:
            idx = df.groupby(cols)['rank_score'].transform(
                max) == df['rank_score']
            df = df[idx].drop_duplicates(subset=list(self.exon_coordinates))

        if df.shape[0] > 0:
            _absent = self.exon_coordinates.merge(
                df,
                how='left',
                left_on=list(self.exon_coordinates),
                right_on=list(self.exon_coordinates),
                indicator=True)

            absent_in_gtf = _absent[_absent['_merge'] == 'left_only'].drop(
                columns='_merge').dropna(axis=1)

        # If no exon is known
        else:
            absent_in_gtf = self.exon_coordinates

        return df, absent_in_gtf

    def select_exons_by_ID(self, full_exon_df: pd.DataFrame):
        """
        Retrieves exons from a list of IDs
         (exon IDs, transcript IDs, or both)
        :param pd.DataFrame full_exon_df: Dataframe
         with all exons from genes of provided biotype (s)
        :return: pd.Dataframe: Dataframe with filtered exons
        """
        if self.transcript_ids and self.exon_ids is None:
            return full_exon_df[full_exon_df['transcript_id'].isin(
                self.transcript_ids)]

        elif self.exon_ids and self.transcript_ids is None:
            return full_exon_df[full_exon_df['exon_id'].isin(self.exon_ids)]

        else:
            return full_exon_df[
                (full_exon_df['transcript_id'].isin(self.transcript_ids))
                & (full_exon_df['exon_id'].isin(self.exon_ids))]

    def get_splice_sites(self):
        """
        Creates a dataframe with all the splice sites from
         the `exons` and generates bed files of those.

        Start coord of 1st exon and last coord of last exon
        are masked so that junctions on these sites won't
        be retrieved (which makes sense because they do not exist).
        This operation also removes single exon transcripts,
        which do not contain any splice sites.

        :return pd.Dataframe: Df with splice site coordinates
        from the exons df of Exons object
        """

        logger.info("Extracting splice site coordinates from exons df ..")
        n_transcripts = self.exons['transcript_id'].nunique()
        logger.info(
            "Number of transcripts to get splice sites from: {}".format(
                n_transcripts))

        exons_to_bed = self.exons.sort_values(['Chromosome', 'Start',
                                               'End']).copy()
        exons_to_bed.loc[:, 'before_start'], exons_to_bed.loc[:, 'after_end'], exons_to_bed.loc[:, 'Start'] = \
            exons_to_bed.Start - 2, exons_to_bed.End + 2, exons_to_bed.Start

        start = pd.melt(exons_to_bed,
                        id_vars=[
                            "Chromosome", "before_start", "Start", "gene_name",
                            "exon_number", "Strand"
                        ],
                        value_vars="transcript_id")

        mask_start = start.groupby(['value'])['value'].transform(
            self.mask_first_row).astype(bool)
        start = start.loc[mask_start]
        start.columns = [
            "Chromosome", "Start", "End", "gene_name", "exon_number", "Strand",
            "group", "transcript_id"
        ]
        n_transcripts_remaining = start.groupby(["gene_name",
                                                 "transcript_id"]).ngroups

        end = pd.melt(exons_to_bed,
                      id_vars=[
                          "Chromosome", "End", "after_end", "gene_name",
                          "exon_number", "Strand"
                      ],
                      value_vars="transcript_id")
        mask_end = end.groupby(['value'])['value'].transform(
            self.mask_last_row).astype(bool)
        end = end.loc[mask_end]
        end.columns = [
            "Chromosome", "Start", "End", "gene_name", "exon_number", "Strand",
            "group", "transcript_id"
        ]

        logger.info(
            "Transcripts with just one exon (no splice sites): {}".format(
                n_transcripts - n_transcripts_remaining))
        logger.info("Transcripts with multiple exons: {}".format(
            n_transcripts_remaining))
        final_bed = pd.concat([start, end], axis=0)

        final_bed.sort_values(['Chromosome', 'Start', 'End'], inplace=True)
        del final_bed['group']
        return final_bed

    def get_all_exon_coordinates(self, df_exons: pd.DataFrame):
        """
        As opposed to `get_non_redundant_exonic_coordinates` method,
        this function will return all non-merged exonic
        coordinates in the exons dataframe.

        :param pd.DataFrame df_exons: Dataframe with exons
        :return pd.DataFrame: Df with all unique exonic
        coordinates
        """

        df_exons['transcript_ids'] = df_exons.groupby(
            'exon_id')['transcript_id'].transform(','.join)

        exons = df_exons.copy()
        df_exons.drop('transcript_ids', axis=1, inplace=True)
        exons = exons.drop(
            ['Source', 'Feature', 'exon_id', 'transcript_id', 'exon_number'],
            axis=1).drop_duplicates()
        return exons

    def get_non_redundant_exonic_coordinates(self,
                                             df_exons: pd.DataFrame,
                                             by_gene: bool = True):
        """
        Generate non-redundant set of stranded exon coordinates
        per gene considering just the transcripts for which `df_exons`
        belong to.

        :param pd.DataFrame df_exons: Dataframe with exons
        :param bool by_gene: Return intervals in a gene-by-gene
        basis. Default: `True`, exonic intervals of different
        genes that overlap will be repeated in the output.
        :return pd.Dataframe: Sorted df
        """

        if by_gene:
            _output = PyRanges(df_exons).merge(strand=True,
                                               count=True,
                                               by=['gene_id', 'gene_name'])
        else:
            _output = PyRanges(df_exons).merge(strand=True, count=True)

        n_intervals = _output.as_df().shape[0]
        logger.info(
            "Number of exonic intervals returned: {}".format(n_intervals))

        return _output.sort().as_df()

    def get_non_redundant_inverse_coordinates(
            self,
            df_exons: pd.DataFrame,
            by_gene: bool = True,
            shrink_5prime: int = 0,
            shrink_3prime: int = 0) -> pd.DataFrame:
        """
        Generate non-redundant set of stranded intronic
        coordinates per gene considering just the transcripts for
        which the `df_exons` belong to. It allows the generation
        of different sets of intronic intervals (e.g. If df_exons
        contains just protein_coding transcripts, more intervals
        should be returned compared to an input df that has
        also unprocessed RNAs, for instance)

        Intronic sequences can be shrinked to include genomic
        intervals that represent deeper intronic regions.

        :param pd.DataFrame df_exons: Dataframe with exons.
        Can be the class variable `self.exons`, or, if a more
        stringent set of exons is desired, exons belonging
        to a set of filtered transcripts (after running
        `exons_obj.extract_surrounding_features(self.exons,..)`
        can be provided.

        :param bool by_gene: Return intervals subtracted from
        merged exons in a gene-by-gene basis. Default: `true`,
        intronic intervals of different genes that overlap will
        be repeated in the output. If set to `False`, non-overlapping
        set of coordinates will be returned, always taking strand
        into consideration (meaning that even with `by_gene` set to
        `False`, overlapping coordinates in different strands will
        be kept in different intervals

        :param int shrink_5prime: How many base pairs to shrink
        intronic intervals from 5'ss coordinate. Default: `0`

        :param int shrink_3prime: How many base pairs to shrink
        intronic intervals from the 3'ss coordinate. Default: `0`

        :return pd.DataFrame: Df with non-exonic coordinates in
        bed-like format
        """
        logger.info("Generating non-redundant intronic coordinates.")
        tx_ids = df_exons['transcript_id'].unique()
        tx_boundaries = PyRanges(self.full_biotype_df[
            (self.full_biotype_df.Feature == "transcript")
            & self.full_biotype_df.transcript_id.isin(tx_ids)])

        intron_coord = tx_boundaries.subtract(PyRanges(df_exons))

        if by_gene:
            _output = intron_coord.cluster(strand=True,
                                           count=True,
                                           by="gene_id",
                                           nb_cpu=4)
        else:
            _output = intron_coord.cluster(strand=True, count=True, nb_cpu=4)

        def _shrink_by_strand(row: pd.Series, n_to_shrink: int, is_5ss=True):
            """
            Does coordinate shrinking based
            on the strand

            :param pd.Series row: Single interval to update
            :param int n_to_shrink: Number of basepairs to
            shrink
            :param bool is_5ss: Wether one is shrinking
            from 5'ss coordinates. Default: `True`. If
            `False`, shrinking from 3'ss is performed instead
            """

            if is_5ss:
                if row.Strand == "+":
                    row.Start += n_to_shrink
                else:
                    row.End -= n_to_shrink
            else:
                if row.Strand == "+":
                    row.End -= n_to_shrink
                else:
                    row.Start += n_to_shrink
            return row

        if shrink_5prime != 0:
            assert shrink_5prime > 0, "Only positive values are allowed for" \
                                      " 'shrink_5prime' arg"

            _output = PyRanges(_output.as_df().apply(_shrink_by_strand,
                                                     n_to_shrink=shrink_5prime,
                                                     axis=1))

        if shrink_3prime != 0:
            assert shrink_3prime > 0, "Only positive values are allowed for" \
                                      " 'shrink_bp_from_3prime' arg"

            _output = PyRanges(_output.as_df().apply(_shrink_by_strand,
                                                     n_to_shrink=shrink_3prime,
                                                     is_5ss=False,
                                                     axis=1))

        n_intervals = _output.as_df().shape[0]
        _output = _output[_output.Start < _output.End]

        if shrink_5prime != 0 or shrink_3prime != 0:
            logger.info(
                "Number of intronic intervals removed after shrinking "
                "(start coord > end coord): {}".format(
                    n_intervals - _output.as_df().shape[0]))
            n_intervals = _output.as_df().shape[0]

        logger.info(
            "Number of intronic intervals returned: {}".format(n_intervals))
        cols = [
            'Chromosome', 'Start', 'End', 'Score', 'Strand', 'transcript_id',
            'gene_id', 'gene_name'
        ]
        return _output.sort().as_df()[cols]

    def mask_first_row(self, x):
        """
        Masks start coordinate of first exon
        (it's not a splice site)
        :param x:
        :return:
        """
        result = np.ones_like(x)
        result[0] = 0
        return result

    def mask_last_row(self, x):
        """
        Masks end coordinate of last exon
        (it's not a splice site)
        :param x:
        :return:
        """
        result = np.ones_like(x)
        result[-1] = 0
        return result


class Introns(Features):
    """
    Representation of intron annotations.
    Inherits from `Features` class and
    coordinates are 0-based
    """
    def __init__(
        self,
        gtf: TextIO,
        gtf_is_processed: bool,
        out_dir: Union[str, pathlib.Path],
        fasta: Optional[Union[str, TextIO]] = None,
        gene_biotypes: Optional[list] = None,
        transcript_biotypes: Optional[list] = None,
        gene_ids: Optional[Union[TextIO, str, List]] = None,
        gene_names: Optional[Union[TextIO, str, List]] = None,
        transcript_ids: Optional[Union[list, TextIO]] = None,
    ):
        """
        Create an instance of an Intron object. Coordinates refer
        to 0-based half-open intervals

        :param TextIO gtf: Path to the GTF file
        :param bool gtf_is_processed: Whether GTF
         was processed before. If `True` `gtf` is a file
         that can be loaded as a pandas Dataframe.
        :param str out_dir: Path to write output files
        :param str fasta: path to the reference genome in fasta
        :param list gene_biotypes: Retrieve introns whose
            parent gene belongs to the given biotype(s)
        :param list transcript_biotypes: Retrieve introns whose
            parent transcript belongs to the given biotype(s)
                :param Optional[Union[TextIO, str, List]] gene_ids:
            File with a list of gene IDs to be retrieved.
            Only introns belonging to the given `biotype`
            and `gene_ids` will be selected. If provided, it
            overrides the  `select_top` argument
        :param Optional[Union[TextIO, str, List]] gene_names:
            File with a list of gene names to be retrieved.
            Only introns belonging to the given `biotype`
            and `gene_names` will be selected. If provided,
            it overrides the `select_top` argument
        :param Optional[list, TextIO] transcript_ids: Transcript IDs to
        select introns from. Can be a list provided from a `Transcripts` object,
        or a file with transcript IDs.
        """
        cols = [
            "Chromosome", "Feature", "Start", "End", "Strand", "gene_id",
            "gene_name", "transcript_id", "transcript_type", "intron_id"
        ]

        super().__init__(gtf,
                         gtf_is_processed,
                         out_dir,
                         fasta=fasta,
                         gene_biotypes=gene_biotypes,
                         transcript_biotypes=transcript_biotypes,
                         gene_ids=gene_ids,
                         gene_names=gene_names)

        if isinstance(transcript_ids, list) or transcript_ids is None:
            self.transcript_ids = transcript_ids
        else:
            self.transcript_ids = read_features_file(transcript_ids)

        pyrng = PyRanges(self.full_biotype_df)

        introns = pyrng.features.introns(by="transcript")

        if introns.empty:
            raise ValueError("No introns in the dataframe.")

        introns = introns.assign(
            "intron_id", lambda df: df.Chromosome.astype(str) + "_" + df.Start.
            astype(str) + "_" + df.End.astype(str))
        self.introns = self.select_introns(
            introns.as_df()) if self.transcript_ids else introns
        self.introns = self.introns[cols].sort().as_df()

    def select_introns(self, full_df: pd.DataFrame):
        """
        Retrieves exons from a list of transcript IDs
        :param pd.DataFrame full_df: Dataframe
         with all features from genes of provided biotype (s)
        :return: pd.Dataframe: Dataframe with filtered introns
        """
        return PyRanges(full_df[full_df['transcript_id'].isin(
            self.transcript_ids)])

    def get_all_intron_coordinates(self):
        """
        As opposed to `get_inverse_coordinates` method
        in the exons class, this function will return
        all intronic coordinates in the introns dataframe

        :return:
        """
        self.introns['transcript_ids'] = self.introns.groupby(
            'intron_id')['transcript_id'].transform(','.join)

        introns = self.introns.copy()
        self.introns.drop('transcript_ids', axis=1, inplace=True)
        introns = introns.drop(
            ['Source', 'Feature', 'intron_id', 'transcript_id'],
            axis=1).drop_duplicates()

        return introns

    def extend_coordinates_from_splice_sites(self,
                                             extend_5prime: int = 2,
                                             extend_3prime: int = 2):
        """
        From the dataframe with the introns,
        returns the genomic coordinates that
        compose the beginning and end of introns.

        By default, it extracts splice site coordinates
        by extending the beginning of intron with 2bp and
        subtracting 2bp from the end of the intron.

        :param int extend_5prime: Number of basepairs to extend
        in the beginning of the intron
        :param int extend_3prime: Number of basepairs to extend
        backwards in the end of the intron

        :return pd.DataFrame:
        """
        logger.info("Extracting splice site regions.")
        df = self.introns.copy()

        def _extend_by_strand(row: pd.Series, n_in_5ss: int, n_in_3ss: int):
            """
            Returns intronic intervals up to a
            given number of basepairs

            :param pd.Series row: Single interval to update
            :param int n_in_5ss: Number of basepairs to
            extend in 5'ss
            :param int n_in_3ss: Number of basepairs to
            subtract in 3'ss
            """

            if row.Strand == "-":
                row['5ss_start'] = row.End - n_in_5ss
                row['5ss_end'] = row.End
                row['3ss_start'] = row.Start
                row['3ss_end'] = row.Start + n_in_3ss

            else:
                row['5ss_start'] = row.Start
                row['5ss_end'] = row.Start + n_in_5ss
                row['3ss_start'] = row.End - n_in_3ss
                row['3ss_end'] = row.End

            return row

        tqdm.pandas()
        df = df.progress_apply(_extend_by_strand,
                               n_in_5ss=extend_5prime,
                               n_in_3ss=extend_3prime,
                               axis=1)

        d_cols = [
            'Chromosome', '5ss_start', '5ss_end', 'gene_id', 'gene_name',
            'Strand', 'transcript_id'
        ]
        a_cols = [
            'Chromosome', '3ss_start', '3ss_end', 'gene_id', 'gene_name',
            'Strand', 'transcript_id'
        ]

        donor_regions = df[d_cols].rename(columns={
            '5ss_start': 'Start',
            '5ss_end': 'End'
        })
        donor_regions['region'] = 'donor'

        acceptor_regions = df[a_cols].rename(columns={
            '3ss_start': 'Start',
            '3ss_end': 'End'
        })
        acceptor_regions['region'] = 'acceptor'

        return pd.concat([donor_regions, acceptor_regions
                          ]).sort_values(['Chromosome', 'Start', 'End'])


def compute_gc_and_hexamers(df: pd.DataFrame,
                            fasta: Union[str, Fasta],
                            one_based: bool = False):
    """
        Computes the gc content of each subfeature
        and stores the results in a column called
        `gc_content`. Additionally, computes hexamer
        counts

        :param df:
        :param Union[str, Fasta]: Fasta file/object to compute features
        :param bool one_based: Whether start coordinates
        of `df` are 1-based (e.g. gtf,bam). Default: `False`,
        coordinates are 0-based
        :return pd.DataFrame: df with GC content added
        """
    possible_hexamers = [''.join(x) for x in list(itertools.product('ATCG', repeat=6))]
    
    if isinstance(fasta, str):
        fasta = open_fasta(fasta)
 
    df[['gc', 'cg_dinuc', 'hexamers'
        ]] = (df.apply(get_fasta_sequences,
                       fasta=fasta,
                       one_based=one_based,
                       axis=1)).apply(compute_sequence_based_features,
                                      all_hexamers=possible_hexamers)
    return df


def compute_length(df: pd.DataFrame = None, one_based: bool = False):
    """
    Computes the length of each subfeature
    and stores the results in a column called
    `length`

    :param pd.DataFrame df: Df with features to
        compute the length
    :param bool one_based: Whether start coordinates
        of `df` are 1-based. Default: `False`.
    :return pd.DataFrame: df with feature length added
    """
    df['length'] = df.apply(lambda x: x['End'] - x['Start'] + 1
                            if one_based else x['End'] - x['Start'],
                            axis=1)
    return df

def extract_surrounding_features(target_df: Union[pd.DataFrame, PyRanges],
                                 subfeatures_df: Union[pd.DataFrame, PyRanges],
                                 what_is_under_study: str = "exons",
                                 level: int = 1):
    """
    Retrieve information about exons or introns surrounding each
        feature in the target_df. Final dataframe may be smaller than
        target_df, since only features belonging to transcripts present
        in the dataframe of subfeatures (they are dependent on the
        `select_top` arguments) will be kept.

    :param Union[pd.DataFrame, PyRanges] target_df: Df (or pyranges)
        with genomic features to add attributes from upstream
        and downstream features
    :param Union[pd.DataFrame, PyRanges] subfeatures_df: Df (or pyranges)
        with exon/intro info generated by running `explode_transcripts_subfeatures`
        on a `Transcript` object. Coordinates are 1-based if pd.Dataframe
    :param str what_is_under_study: Whether `target_df` refers to a set of exons,
        introns, or variants. Default: "exons".
    :param bool just_intervals: Return just the genomic intervals of the
    features that surround the target feature. Default: `False`, return
    genome architecture features of each neighbour subfeature
    :param int level: Number of levels to return neighbour subfeatures
    both upstream and downstream of the target features. Default: `1`.
    E.g. If the target feature is an exon, returns the info about the
    upstream and downstream intron. If level is set to `2`, and the target
    feature is an exon, it return both the upstream and downstream introns
    as well as the upstream and downstream exons.

    :return: Exonic or Intronic df with information regarding
        the upstream and downstream features
    """
    logger.info("Extracting upstream and downstream genomic features.")
    assert isinstance(
        target_df,
        (pd.DataFrame, PyRanges)), '"target_df" is of an invalid type.'
    assert isinstance(
        subfeatures_df,
        (pd.DataFrame, PyRanges)), '"subfeatures_df" is of an invalid type.'
    assert what_is_under_study in [
        'exons', 'introns', 'variants'
    ], "Wrong value in 'what_is_under_study' arg."

    # Only features/variants from transcripts
    # represented in subfeatures_df will be kept
    pr_target_df = target_df.copy()
    pr_target_df = pr_target_df[pr_target_df.transcript_id.isin(set(subfeatures_df.transcript_id))]
    
    if pr_target_df.empty:
        raise ValueError(
            "No target features matching transcript IDs present in the "
            "subfeatures df. Either you set --gene_biotype and --tx_biotype "
            "in the Transcripts object (they limit the available tx IDs in the "
            "database) or you set --select_top argument (just 1 tx ID per gene "
            "is available, which probably is not present in the target df.")
    is_variant = False
    if what_is_under_study == "exons":
        _id = "exon_id"
    elif what_is_under_study == "introns":
        _id = "intron_id"
    else:
        is_variant = True
        _id = "HGVSc" if "HGVSc" in pr_target_df.columns else "ID"

    assert _id in pr_target_df.columns, "{} column doesn't exist. Please check " \
                                        "'what_is_under_study' arg".format(_id)

    if isinstance(subfeatures_df, pd.DataFrame):
        subfeatures_df = PyRanges(subfeatures_df)

    # To avoid extracting nearest intervals corresponding to
    # different transcript IDs, a groupby is  first performed 
    # so that per-transcript nearest intervals are retrieved.

    # Before, different gene_id were kept in the subfeature df
    # so that first and last exons or single exon transcripts
    # are kept (a neighbour is found)
    pr_target_df["id"] = pr_target_df[[_id, 'transcript_id']].apply('_'.join, axis=1)
    unique_id = _id if is_variant else "id"

    grouped_tx = pr_target_df.groupby("transcript_id")

    if grouped_tx.ngroups > 10:
        
        with Pool(cpu_count()) as p:
            
            groups = [group for _, group in grouped_tx]
            inputs = zip(groups, itertools.repeat(subfeatures_df),
                         itertools.repeat(level))

            res = p.starmap(_return_neighbour_attributes,
                            tqdm(inputs, total=len(groups)))

            neighbours = pd.concat(res)

    else:
        tqdm.pandas()
        neighbours = grouped_tx.progress_apply(
            lambda g: _return_neighbour_attributes(
                g, subfeatures_df, level=level))

    int_cols = list(
        neighbours.filter(regex=r'(Start_|End_|Number_|Distance_|Length_)'))
    neighbours[int_cols] = neighbours[int_cols].astype('Int64')
    logger.info("Done.")
    return neighbours.sort_values(['Chromosome', 'Start', 'End']).reset_index(drop=True)


def _return_neighbour_attributes(target: pd.DataFrame,
                                 subfeatures: PyRanges, level: int):
    """
    Get genomics intervals of surrounding
    (upstream or downstream) features

    :param pd.DataFrame target_feature: Single target
    feature that represents a single group
    (per transcript ID) from a groupby object

    :param PyRanges subfeatures: PyRanges object
    with ordered subfeatures (exons and introns) for
    each transcript in a Transcripts object

    :param int level: How many levels of upstream
    and downstream intervals to inspect. If set to
    `0`, only returns info on the target intervals

    :return pd.DataFrame: Additional info
    for each row in the `target_feature`
    """

    assert "transcript_id" in target.columns, "transcript_id does not exist in the target " \
                                                      "feature df "

    def _rename_cols(loc: str):
        """
        Creates the dict to rename
        the results df based on the
        searched location

        :param str loc: Searched location
        :return dict:
        """
        return {
            'Start_b': 'Start_{}'.format(loc),
            'End_b': 'End_{}'.format(loc),
            'Feature_b': 'Feature_{}'.format(loc),
            'number': 'Number_{}'.format(loc),
            'Distance': 'Distance_{}'.format(loc),
            'gc': 'GC_{}'.format(loc),
            'length': 'Length_{}'.format(loc)
        }
        # 'cg_dinuc': 'CG_dinuc_{}'.format(loc),
        # 'hexamers': 'Hexamers_{}'.format(loc)}

    # Subset subfeatures so that only rows matching
    # the tx id of the target feature are kept
    tx_id = target.transcript_id.iloc[0]
    mask = (subfeatures.transcript_id == tx_id)
    subset = subfeatures[mask]

    # Extract GC, etc on target intervals
    if 'gc' in subset.columns:
        cols = [
            'Chromosome', 'Start', 'End', 'Strand', 'gene_id', 'gene_name',
            'transcript_id', 'transcript_type', 'Feature'
        ]

        target = target.merge(subset.as_df(),
                              left_on=cols,
                              right_on=cols, 
                              suffixes=('', '_repeat'))
    
    target = target[[c for c in target.columns if not c.endswith('_repeat')]]
    target = target.rename(columns=_rename_cols('target'))

    # If the goal is to retrieve
    # additional features just on
    # the target intervals
    if level == 0:
        return target

    pr_target_feature = PyRanges(target)
    if level == 1:
        res = []
        for loc in ['upstream', 'downstream']:

            _r = pr_target_feature.nearest(subset,
                                           overlap=False,
                                           strandedness='same',
                                           how=loc).as_df()

            if not _r.empty:
                _r = _r.rename(columns=_rename_cols(loc))
                _r.drop(columns=list(_r.filter(regex='_b')), inplace=True)
                res.append(_r)

    elif level > 1:
        res = []

        for loc in ['upstream', 'downstream']:

            _r = pr_target_feature.k_nearest(subset,
                                             k=level,
                                             overlap=False,
                                             strandedness='same',
                                             how=loc).as_df()

            if not _r.empty:
                _r = _r.rename(columns=_rename_cols(loc))
                _r.drop(columns=list(_r.filter(regex='_b')), inplace=True)

                # Long to wide
                _r['idx'] = _r.groupby(list(target)).cumcount() + 1

                _r = pd.pivot_table(_r,
                                    index=list(target),
                                    columns='idx',
                                    values=list(
                                        _r.filter(regex='{}'.format(loc))),
                                    observed=True,
                                    aggfunc='first')

                _r = _r.sort_index(axis=1, level=1)
                _r.columns = [f'{x}_{y}' for x, y in _r.columns]
                _r = _r.reset_index()
                res.append(_r)

    elif level < 0:
        raise ValueError('Level can not be a negative number')

    if len(res) == 2:

        res_df = reduce(lambda df1, df2: pd.merge(df1, df2, how='outer'), res)
        res_df = res_df.sort_values('Start').reset_index()

        return res_df

    elif len(res) == 1:
        return res[0].sort_values('Start').reset_index()

    else:
        return

def insert_exons(new_exons: pd.DataFrame,
                 subfeatures: pd.DataFrame,
                 look_for: str = 'gene_name',
                 select_top: bool = True):
    """
    Insert new exons coordinates within a subfeatures dataframe.

    It will insert the exon in all the transcripts which have
    non-overlapping exons both upstream and downstream of the
    new exon location

    :param pd.DataFrame new_exons: New exons to be added
    :param pd.DataFrame subfeatures: Exploded subfeatures
    :param str look_for: Gene identifiers to look for
    :param bool select_top: Select top transcript associated
    with the new exon

    :return pd.DataFrame: Updated new exons df with transcript IDs
    assigned
    :return pd.DataFrame: Exploded subfeatures with transcripts
    for which it was possible to insert a new exon.
    :return pd.DataFrame: Df of new exons for which it was not
    possible to assign any transcript ID
    """
    logger.info("Inserting new exon coordinates and fixing transcripts structure")

    _new_exons = new_exons.to_dict('records')
    all_gene_names = set(subfeatures[look_for])

    # get gene_name_col
    _row = _new_exons[0]

    gene_name_idx = [x in all_gene_names
                     for x in list(_row.values())].index(True)
    gene_name_col = list(_row.keys())[gene_name_idx]

    updated_subfeatures = []
    updated_exons = []
    discarded_exons = []

    if len(_new_exons) > 10:
 
        with Pool(cpu_count()) as p:

            inputs = zip(_new_exons, itertools.repeat(subfeatures),
                        itertools.repeat(gene_name_col), itertools.repeat(look_for))

            res = p.starmap(_insert_exon, tqdm(inputs, total=len(_new_exons)))

            for r in res:
                if r[2] is None:
                    updated_exons.append(r[0])
                    updated_subfeatures.append(r[1])
                else:
                    discarded_exons.append(r[2])

    else:
    
        for _exon in tqdm(_new_exons):
            _updt_e, _updt_subf, discard_exon = _insert_exon(_exon, subfeatures, gene_name_col, look_for)

            if discard_exon is None:
                updated_exons.append(_updt_e)
                updated_subfeatures.append(_updt_subf)
            else:
                discarded_exons.append(discard_exon)

    updated_exons = pd.concat(updated_exons)

    if select_top:
        idx = updated_exons.groupby(['exon_id'])['rank_score'].transform(max) == updated_exons['rank_score']
        updated_exons = updated_exons[idx].drop_duplicates(subset=['exon_id', 'gene_id'])

    logger.info("Done")
    
    if not all(x in updated_exons.columns for x in ['gene_name', 'gene_id']):
        merged_on = "gene_id" if "gene_id" in updated_exons.columns else "gene_name"
        updated_exons = pd.merge(updated_exons, subfeatures[['gene_name', 'gene_id']].drop_duplicates(), 
                                 how='left',
                                 on=merged_on)

    return updated_exons, pd.concat(updated_subfeatures), pd.DataFrame(discarded_exons)

def _insert_exon(_exon, subfeatures: pd.DataFrame, gene_identifier_col: str, look_for: str = 'gene_name'):
    discard = True
    _updt_exons, _updt_subft = [], []
    
    # get gene name and subset subfeatures df
    gene = _exon[gene_identifier_col]
    _subfeatures = subfeatures.copy()
    _subfeatures = _subfeatures[(_subfeatures[look_for] == gene)]

    # Remove feature values (if they were calculated)
    # because they will need to be computed again
    if 'gc' in _subfeatures.columns:
        _subfeatures.drop(columns=['gc', 'length'], inplace=True)

    # creates pyranges for the new exon
    d = {k: [v] for k, v in _exon.items()}
    exon = pr.from_dict(d)

    # select transcript based on the highest rank
    # tx_id = _subfeatures.loc[_subfeatures.rank_score.idxmax()].iloc[0].transcript_id
    # _subfeatures = PyRanges(_subfeatures[subfeatures.transcript_id == tx_id])

    # Insert new exon in all the transcripts of the gene
    for name, group in _subfeatures.groupby('transcript_id'):
        group = group.reset_index(drop=True)
        tx = PyRanges(group)

        # If new exon is out of the boundaries of the transcript, continue
        if tx.intersect(exon).empty:
            continue

        # if can't extract non-overlapping upstream and downstream exons, continue
        _upstream = exon.nearest(tx[tx.Feature == "exon"],
                                    strandedness="same",
                                    overlap=False,
                                    how="upstream")

        _downstream = exon.nearest(tx[tx.Feature == "exon"],
                                    strandedness="same",
                                    overlap=False,
                                    how="downstream")
        
        # If no upstream or downstream, continue
        if any(x.empty for x in [_upstream, _downstream]):
            continue
        
        else:
            discard = False
            # If no overlap, but just adjacent, continue
            for gr in [_upstream, _downstream]:
                gr = gr[(gr.End != gr.Start_b) & (gr.End_b != gr.Start)]
            
                if gr.empty:
                    discard = True
                    break

            if discard:
                continue
            isoform = random.randint(0, 10000)
            _upstream = _upstream.as_df().iloc[0]
            _downstream = _downstream.as_df().iloc[0]
            chrom = exon.Chromosome.iloc[0]
            tx_id = tx.transcript_id.iloc[0] + "_" + str(isoform)
            tx_type = tx.transcript_type.iloc[0]
            if look_for == 'gene_name':
                gene_name = gene
                gene_id = tx.gene_id.iloc[0]
            elif look_for == 'gene_id':
                gene_id = gene
                gene_name = tx.gene_name.iloc[0]
                
            strand = tx.Strand.iloc[0]
            rank_score = tx.rank_score.iloc[0]

            # Remove features between upstream and downstream exons by index
            # This removes both introns and exons that overlap with the new exon
            ups_row_index = group[(group.Start == _upstream.Start_b) &
                                    (group.End == _upstream.End_b)].index[0]

            down_row_index = group[(group.Start == _downstream.Start_b) & (
                group.End == _downstream.End_b)].index[0]

            group = group.drop(
                range(
                    min(ups_row_index, down_row_index) + 1,
                    max(ups_row_index, down_row_index)))
            
            group['transcript_id'] = tx_id
            group['transcript_type'] = tx_type

            # Insert new exon and generate new surrounding coordinates
            start_i_ups = _upstream.End_b if strand == "+" else _upstream.End
            end_i_ups = _upstream.Start if strand == "+" else _upstream.Start_b
            start_i_down = _downstream.End if strand == "+" else _downstream.End_b
            end_i_down = _downstream.Start_b if strand == "+" else _downstream.Start

            upstream_intron = [
                chrom, start_i_ups, end_i_ups, tx_id, tx_type, gene_name,
                strand, gene_id, _upstream.number, rank_score, "intron"
            ]
            downstream_intron = [
                chrom, start_i_down, end_i_down, tx_id, tx_type, gene_name,
                strand, gene_id, _upstream.number + 1, rank_score, "intron"
            ]
            new_exon = [
                chrom, exon.Start.iloc[0], exon.End.iloc[0], tx_id,
                tx_type, gene_name, strand, gene_id, _upstream.number + 1,
                rank_score, "exon"
            ]

            df = pd.DataFrame.from_records(
                [upstream_intron, new_exon, downstream_intron],
                columns=group.columns)
            
            group = PyRanges(pd.concat([group, df])).sort().as_df()

            # update feature numbers
            _repeat = group.groupby('number').filter(lambda x: len(x) > 2)
            if not _repeat.empty:
                if strand == "+":
                    idx_start = _repeat.iloc[2].name
                    idx_end = group.shape[0]

                else:
                    idx_start = 0
                    idx_end = _repeat.iloc[2].name

                to_sum = list(range(idx_start, idx_end))
                group['number'] = group.apply(
                    lambda x: x.number + 1
                    if x.name in to_sum else x.number,
                    axis=1)

            # Assign tx_id to new exon
            _e = exon.as_df().copy()
            _e['gene_id'] = gene_id
            _e['transcript_id'] = tx_id
            _e['transcript_type'] = tx_type
            _e['rank_score'] = rank_score
            _e['exon_id'] = '{}:{}-{}'.format(chrom,
                                                str(exon.Start.iloc[0]),
                                                str(exon.End.iloc[0]))
            _e['Feature'] = 'exon'
            _e['number'] = group[(group.Start == exon.Start.iloc[0]) & (
                group.End == exon.End.iloc[0])].number.values[0]

            _updt_exons.append(_e)
            _updt_subft.append(group)
            
    if discard:    
        return None, None, _exon
    else:
        return pd.concat(_updt_exons), pd.concat(_updt_subft), None

