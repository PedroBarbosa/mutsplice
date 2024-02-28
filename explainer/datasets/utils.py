from cmath import log
from curses.ascii import LF
from functools import total_ordering
import itertools
from math import inf
from typing import List, Union
from itertools import chain
from Bio import SeqIO
from natsort.natsort import index_realsorted
from numpy import dtype, full, invert
from pyfaidx import Fasta
import pandas as pd
import numpy as np
import re
import os
import sys
import pyranges as pr
from pyranges import PyRanges
from gtfhandle.utils import write_fasta_sequences, get_fasta_sequences, write_bed_file
from loguru import logger

def setup_logger(level: int = 0, multiprocessing: bool = False):
    """
    Create loguru handler based on verbosity level provided
    
    """
    level_dict = {0: 'MAIN', 
                  1: 'INFO',
                  2: 'DEBUG'}
    
    log_level = level_dict[level]
    
    try:
        logger.remove(0)
    except ValueError:
        pass
    
    try:
        logger.level("MAIN", no=38, color="<magenta>")
    except TypeError:
        pass 
    
    enqueue = True if multiprocessing else False        
    logger.add(sys.stderr, 
               format = "<green>{time:YYYY-MM-DD|HH:mm:ss}</green> | <lvl>{level}</lvl>: <bold>{message}</bold>", 
               backtrace=False, 
               colorize=True,
               enqueue=enqueue,
               level=log_level)
    

def _location_specific_normalization(df: pd.DataFrame, len_df: pd.DataFrame):
    """
    Normalize counts on each genomic location by its individual length

    :param pd.DataFrame df: Df to be normalized
    :param pd.DataFrame len_df: Df with exon/intron lengths

    :return: Normalized df
    """
    loc_map = {'Intron_upstream_2': 'len_intron_upstream2',
               'Intron_downstream_2': 'len_intron_downstream2',
               'Intron_upstream': 'len_intron_upstream',
               'Intron_downstream': 'len_intron_downstream',
               'Exon_upstream_fully_contained': 'len_exon_upstream',
               'Exon_upstream_acceptor_region': 'len_exon_upstream',
               'Exon_upstream_donor_region': 'len_exon_upstream',
               'Exon_cassette_fully_contained': 'len_exon_cassette',
               'Exon_cassette_acceptor_region': 'len_exon_cassette',
               'Exon_cassette_donor_region': 'len_exon_cassette',
               'Exon_downstream_fully_contained': 'len_exon_downstream',
               'Exon_downstream_acceptor_region': 'len_exon_downstream',
               'Exon_downstream_donor_region': 'len_exon_downstream',
               }

    out = []

    for location, feature_length in loc_map.items():
        features_at_loc = [c for c in df.columns if c.endswith(location)]
        out.append(df[features_at_loc].div(len_df[feature_length], axis=0))

    _df = pd.concat(out, axis=1)

    # Fill NA when len of upstream_2/downstream_2 is 0 (no extended borders)
    cols = [c for c in _df.columns if c.endswith(
        'Intron_upstream_2') or c.endswith('Intron_downstream_2')]

    _df[cols] = _df[cols].fillna(0)

    # Retrieve seq_ids back
    _df = pd.concat([_df, df.seq_id], axis=1)

    # Merge additional columns from original df
    absent_cols = list(set(df.columns).difference(set(_df.columns)))
    if absent_cols:
        _df = pd.merge(_df, df[absent_cols], on='seq_id')

    return _df


def _get_len(df: pd.DataFrame, seqs: str):
    """
    Get overall len of each sequence using ss indexes and number 
    of nucleotides extended from original input sequences.

    From there, generate length of each exon/intron, taking
    into account that splice site indexes (0-based) represent the first
    and last exonic positions
    """
    f = open(seqs, 'rU')
    seq_len = {}
    for rec in SeqIO.parse(f, 'fasta'):
        seq_len[rec.id] = len(rec)
    f.close()
    len_df = pd.DataFrame(seq_len.values(), index=seq_len.keys()).reset_index()
    len_df.columns = ['seq_id', 'len_seq']

    _df = df.copy()
    _df = pd.merge(_df, len_df, on='seq_id')
    
    cols = ['start_exon_upstream', 'end_exon_upstream', 
            'start_exon_cassette', 'end_exon_cassette',
            'start_exon_downstream', 'end_exon_downstream']
    _df[cols] = _df[cols].replace('<NA>', np.nan)
    
   
    _df['len_intron_upstream2'] = _df.start_exon_upstream - 0
    _df['len_intron_upstream'] = _df.start_exon_cassette - \
        _df.end_exon_upstream - 1
    _df['len_intron_downstream'] = _df.start_exon_downstream - \
        _df.end_exon_cassette - 1
    _df['len_intron_downstream2'] = _df.len_seq - _df.end_exon_downstream - 1
    _df['len_exon_upstream'] = _df.end_exon_upstream - \
        _df.start_exon_upstream + 1
    _df['len_exon_cassette'] = _df.end_exon_cassette - \
        _df.start_exon_cassette + 1
    _df['len_exon_downstream'] = _df.end_exon_downstream - \
        _df.start_exon_downstream + 1

    return _df


def _adjust_ss_idx_on_mutated_seqs(info: Union[pd.Series, dict],
                                   ref_ss_idx: Union[List, dict],
                                   other_idx: int = None):
    """
    Adjusts splice site indexes in sequences 
    whose mutation was not a SNV.
    """

    if other_idx is not None:
        if info.type == "DEL":
            _shift = info.end - info.start

            if info.start <= other_idx <= info.end:
                return pd.NA
            elif other_idx > info.end:
                adj = other_idx - _shift
                return adj if adj >= 0 else pd.NA
            else:
                return other_idx

    # If dict, refers to the ref seq
    if isinstance(info, dict) or info.type == "SNV":
        info['start_exon_upstream'] = ref_ss_idx[0][0]
        info['end_exon_upstream'] = ref_ss_idx[0][1]
        info['start_exon_cassette'] = ref_ss_idx[1][0]
        info['end_exon_cassette'] = ref_ss_idx[1][1]
        info['start_exon_downstream'] = ref_ss_idx[2][0]
        info['end_exon_downstream'] = ref_ss_idx[2][1]

    else:
        new_ss_idx = []
        if info.type == "DEL":
            _shift = info.end - info.start

            for ss in list(itertools.chain(*ref_ss_idx)):

                # if <NA>
                if isinstance(ss, str):
                    new_ss_idx.append(ss)

                # If DEL disrupts splice site
                elif info.start <= ss <= info.end:
                    # new_ss_idx.append(pd.NA)
                    new_ss_idx.append(info.start)

                elif ss > info.end:
                    adj = ss - _shift
                    new_ss_idx.append(
                        adj) if adj >= 0 else new_ss_idx.append(pd.NA)

                else:
                    new_ss_idx.append(ss)

        info['start_exon_upstream'] = new_ss_idx[0]
        info['end_exon_upstream'] = new_ss_idx[1]
        info['start_exon_cassette'] = new_ss_idx[2]
        info['end_exon_cassette'] = new_ss_idx[3]
        info['start_exon_downstream'] = new_ss_idx[4]
        info['end_exon_downstream'] = new_ss_idx[5]

    return info


def _get_loc_of_motif(info: pd.DataFrame, ref_ss_idx: dict):
    """
    Generate spatial information of where 
    motif features occur in the sequences.

    :param pd.DataFrame info: Dataframe with motif occurrences
    :param dict ref_ss_idx: Dictionary with the indexes (0-based)
    of splice sites in the reference sequences

    :return pd.DataFrame: Motif ocurrences with additional
    spatial resolution: location, and distances to known
    splice sites
    """

    def _generate_intervals(ref_ss_idx: dict):
        """
        Generates pyranges of the exon/intron 
        intervals for each input sequence with 
        splice site information 

        :return pr.PyRanges: Exon intervals
        :return pr.PyRanges: Intron intervals
        """
        # Pyranges of the reference seqs
        chrom_e, start_e, end_e, label_e = [], [], [], []
        chrom_i, start_i, end_i, label_i = [], [], [], []
        tags_e = ['Exon_upstream', 'Exon_cassette', 'Exon_downstream']
  
        for seq_id, ss_idx in ref_ss_idx.items():

            ref_coords = re.split(r'[(?:\()-]', seq_id)
            interval_len = int(ref_coords[2]) - int(ref_coords[1])

            # exons
            _ups = ss_idx[0]
            _cass = ss_idx[1]
            _down = ss_idx[2]

            _ups = [x if isinstance(x, int) else 0 for x in _ups]
            _down = [x if isinstance(x, int) else interval_len for x in _down]

            zipped = list(zip(*[_ups, _cass, _down]))
            chrom_e.extend([seq_id]*3)
            start_e.extend(zipped[0])
            end_e.extend(zipped[1])
            label_e.extend(tags_e)

            # introns
            if _ups[0] != 0:
                chrom_i.append(seq_id)
                start_i.append(0)
                end_i.append(_ups[0])
                label_i.append("Intron_upstream_2")

            i_ups = [_ups[1], _cass[0]]
            i_down = [_cass[1], _down[0]]
            zipped = list(zip(*[i_ups, i_down]))
            chrom_i.extend([seq_id]*2)
            start_i.extend(zipped[0])
            end_i.extend(zipped[1])
            label_i.extend(['Intron_upstream', 'Intron_downstream'])

            if _down[1] != ref_coords[1]:
                chrom_i.append(seq_id)
                start_i.append(_down[1])
                end_i.append(interval_len)
                label_i.append("Intron_downstream_2")

        d_e = {'Chromosome': chrom_e,
               'Start': start_e,
               'End': end_e,
               'Strand': ['+'] * len(chrom_e),
               'Name': label_e}

        exons = pr.from_dict(d_e)
        exons_cassette = exons[exons.Name == "Exon_cassette"]

        d_i = {'Chromosome': chrom_i,
               'Start': start_i,
               'End': end_i,
               'Strand': ['+'] * len(chrom_i),
               'Name': label_i}

        introns = pr.from_dict(d_i)

        return exons, exons_cassette, introns

    def _distance_to_cassette(motifs: pr.PyRanges,
                              exons: pr.PyRanges):
        """
        Generates motif distances to cassette exons splice site indexes

        :param pr.PyRanges motifs: All motifs ocurrences in the dataset
        :param pr.PyRanges exons: Intervals where cassette exons are located

        :return pr.PyRanges: motifs with 2 additional columns representing the 
        distances to cassette splice sites 
        """
        to_drop_cols = ['Start_b', 'End_b', 'Strand_b', 'Name', 'Distance']
        cass = motifs.nearest(exons, strandedness='same')

        # Motifs that overlap cassette
        cass_overlap = cass[cass.Distance == 0]
        if not cass_overlap.empty:
            cass_overlap.distance_to_cassette_acceptor = cass_overlap.Start - cass_overlap.Start_b
            cass_overlap.distance_to_cassette_donor = cass_overlap.End_b - cass_overlap.End

        # Motifs that locate upstream
        cass_upstream = cass[cass.End <= cass.Start_b]
        if not cass_upstream.empty:
            cass_upstream.distance_to_cassette_acceptor = cass_upstream.Distance
            cass_upstream.distance_to_cassette_donor = cass_upstream.End_b - cass_upstream.End

        # Motifs that locate downstream
        cass_downstream = cass[cass.Start >= cass.End_b]
        if not cass_downstream.empty:
            cass_downstream.distance_to_cassette_acceptor = cass_downstream.Start - \
            cass_downstream.Start_b
            cass_downstream.distance_to_cassette_donor = cass_downstream.Distance

        motifs = pr.concat([cass_upstream, cass_overlap, cass_downstream])
        motifs.distance_to_cassette_acceptor = motifs.distance_to_cassette_acceptor.clip(0)
        motifs.distance_to_cassette_donor = motifs.distance_to_cassette_donor.clip(0)
        return motifs.drop(to_drop_cols).sort()

    def _map_exonic_motifs(motifs: pr.PyRanges, exons: pr.PyRanges):
        """
        Maps location of motifs that overlap with exonic intervals

        :param pr.PyRanges motifs: All motifs ocurrences in the dataset
        :param pr.PyRanges exons: Intervals where exons are located

        :return pr.PyRanges: motifs with 3 additional columns representing the 
        discrete location as well as the distance to the splice sites of the
        exon where the motif was found
        """
        final_to_concat = []
        to_drop_cols = ['Start_b', 'End_b', 'Strand_b', 'Name', 'Overlap']

        _exon_match = motifs.join(exons, report_overlap=True, nb_cpu=1)

        if _exon_match.__len__() > 0:
            _exon_match.is_in_exon = True

            # Subtract first/last nucleotide of exon
            # So that they can latter be assigned to ss region
            fully = _exon_match[(_exon_match.Overlap == _exon_match.End - _exon_match.Start) &
                                (_exon_match.Start - _exon_match.Start_b > 1) &
                                (_exon_match.End_b - _exon_match.End > 1)]
    
            # There are fully contained
            if fully.__len__() > 0:
                fully.location = fully.Name + "_fully_contained"
                final_to_concat.append(fully)

                # There are some partial
                if _exon_match.__len__() != fully.__len__():
                    _p = pd.merge(_exon_match.df, fully.df, 
                                  on=list(_exon_match.df),
                                  how='left',
                                  indicator=True)
                    
                    partial = pr.PyRanges(_p[_p._merge == "left_only"].drop('_merge', axis=1))
                else:
                    partial = pr.PyRanges()
                    
            # All are partial
            else:
                partial = _exon_match.copy()
            

            # PARTIAL
            if partial.__len__() > 0:

                # SHORT EXONS FULLY SPANNED BY MOTIF
                full_span = partial[(partial.Start <= partial.Start_b) &
                                    (partial.End >= partial.End_b)]

                if full_span.__len__()  > 0:

                    full_span.location = full_span.Name + "_fully_contained"
                    #full_span.location = full_span.Name + "_full_span"
                    final_to_concat.append(full_span)
                    
                    # There are some partial that are not full span
                    if partial.__len__() != full_span.__len__():
                        _p = pd.merge(partial.df, full_span.df, 
                                      on=list(partial.df),
                                      how='left',
                                      indicator=True)
                    
                        partial = pr.PyRanges(_p[_p._merge == "left_only"].drop('_merge', axis=1))

                # MOTIFS NEAR/SPANNING ACCEPTORS
                acceptor_region = partial[(partial.End < partial.End_b) |
                                        (partial.Start - partial.Start_b < 2)]

                if acceptor_region.__len__() > 0:
                    acceptor_region.location = acceptor_region.Name + "_acceptor_region"
                    final_to_concat.append(acceptor_region)

                # MOTIFS NEAR/SPANNING DONORS
                donor_region = partial[(partial.End > partial.End_b) |
                                    (partial.End_b - partial.End < 2)]
                if donor_region.__len__() > 0:
                    donor_region.location = donor_region.Name + "_donor_region"
                    final_to_concat.append(donor_region)

            exonic_motifs = pr.concat(final_to_concat)
            exonic_motifs.distance_to_acceptor = exonic_motifs.Start - exonic_motifs.Start_b
            exonic_motifs.distance_to_donor = exonic_motifs.End_b - exonic_motifs.End
            exonic_motifs = exonic_motifs.drop(to_drop_cols)

        else:
            return pr.PyRanges()

        return exonic_motifs

    def _map_intronic_motifs(motifs: pr.PyRanges, introns: pr.PyRanges):
        """
        Maps location of motifs that exclusively overlap with intronic intervals

        :param pr.PyRanges motifs: All motifs ocurrences in the dataset
        :param pr.PyRanges introns: Intervals where introns are located

        :return pr.PyRanges: motifs with 3 additional columns representing the 
        discrete location as well as the distance to the splice sites of the
        intron where the motif was found
        """
        to_drop_cols = ['Start_b', 'End_b', 'Strand_b', 'Name', 'Overlap']
        _intron_match = motifs.join(introns, report_overlap=True, nb_cpu=1, apply_strand_suffix=False)

        if _intron_match.__len__() > 0:
            i_motifs = _intron_match[_intron_match.Overlap ==
                                            _intron_match.End - _intron_match.Start]

            if i_motifs.__len__() > 0:
                i_motifs.is_in_exon = False
                i_motifs.location = i_motifs.Name
                i_motifs.distance_to_acceptor = i_motifs.End_b - i_motifs.End
                i_motifs.distance_to_donor = i_motifs.Start - i_motifs.Start_b
                i_motifs = i_motifs.drop(to_drop_cols)
                i_motifs = i_motifs.df
                i_motifs.loc[i_motifs.location == "Intron_upstream_2", "distance_to_donor"] = pd.NA
                i_motifs.loc[i_motifs.location == "Intron_downstream_2", "distance_to_acceptor"] = pd.NA
                i_motifs = pr.PyRanges(i_motifs)
            else:
                return pr.PyRanges()
        else:
            return pr.PyRanges()
        
        return i_motifs
    
    ############################
    #### Generate intervals ####
    ############################
    logger.debug('.. generating intervals fromm ss idx ..')
    exons, exons_cassette, introns = _generate_intervals(ref_ss_idx)
    motifs = PyRanges(info.rename(columns={'seq_id': 'Chromosome',
                                           'start': 'Start',
                                           'end': 'End'}))
    motifs.Strand = '+'

    #############################
    ### Dist to cassette exon ###
    #############################
    logger.debug('.. calculating distances to cassette ss ..')
    motifs = _distance_to_cassette(motifs, exons_cassette)

    ############################
    ### Loc of exonic motifs ###
    ############################
    logger.debug('.. mapping location of exonic motifs ..')
    exonic_motifs = _map_exonic_motifs(motifs, exons)

    ############################
    ## Loc of intronic motifs ##
    ############################
    logger.debug('.. mapping location of intronic motifs ..')
    intronic_motifs = _map_intronic_motifs(motifs, introns)

    out = pr.concat([exonic_motifs, intronic_motifs])
    out.distance_to_acceptor = out.distance_to_acceptor.clip(0)
    out.distance_to_donor = out.distance_to_donor.clip(0)
    return out.sort().as_df().rename(columns={'Chromosome': 'seq_id'})


def _process_ss_idx(seqs: dict, ss_info: Union[str, dict]):
    """
    Processes the splice site indexes of a reference
    sequence

    :param dict seqs: Sequences to scan/mutate

    :param Union[str, dict] ss_info: Information about splice site 
    indexes on the upstream, cassette and downstream exons

    :return Union[List, dict]: Processed list
    with indexes for upstream, cassette
    and downstream exons if sequences represent
    mutations from a reference sequence. Or a dict
    with exon indexes per sequence in the file, if
    sequences do not represent mutations from a
    single reference sequence

    :return str: Coordinates of the non-extended input
    features (e.g. exons) if sequences represent
    mutations from a reference sequence. Or a dict
    target coordinates per sequence in the file, if
    sequences do not represent mutations from a
    single reference sequence
    """

    def _get_idx(_ss: pd.DataFrame, seq_name: str, txId: str):
        """"
        :param _ss pd.DataFrame: Df with all ss_idx per
        original sequence
        :param str seq_name: Sequence name with just the
        spanning coordinates
        :param str txId: Transcript ID for the seq_name

        :return List: List with the indexes of the exons
        present in the given seq_name
        """

        _ss = _ss.loc[(_ss.header == seq_name) & (_ss.tx_id == txId)]

        assert _ss.shape[0] == 1, "More than one row with splice site " \
                                  "indexes for {} seq.".format(seq_name)

        idx = list(chain.from_iterable(zip(_ss.iloc[0].acceptor_idx.split(";"),
                                           _ss.iloc[0].donor_idx.split(";"))))

        idx = [x if x == '<NA>' else int(x) for x in idx]

        upstream = idx[0:2]
        cassette = idx[2:4]
        downstream = idx[4:]

        return [upstream, cassette, downstream], _ss.iloc[0].Name

    if ss_info:

        if isinstance(ss_info, dict):
            return ss_info, None

        out_idx, out_coords = {}, {}
        ss = pd.read_csv(ss_info, sep="\t")

        for i, k in enumerate(seqs.keys()):

            seq_header = k.split("_")[0]
            tx_id = '_'.join(k.replace("_REF_seq", "").split('_')[1:])

            if seq_header not in list(ss.header):
                raise ValueError(
                    '{} seq ID not in splice site idx file.'.format(
                        seq_header))

            # If REF_seq, seqs represent mutated sequences
            # so the ss_idx should be the same for all the
            # sequences. Idx are returned right away

            # REF_seq header comes in:
            # spanningCoordinates_transcriptID_REF_seq
            if "REF_seq" in k and i == 0:

                return _get_idx(ss, seq_header, tx_id)

            elif "REF_seq" in k:
                raise ValueError(
                    'Problem with reference sequence ({}). It should come '
                    'first in a fasta file with mutated sequences.'.format(k))

            else:

                indexes, target_coordinates = _get_idx(ss, seq_header, tx_id)

                out_idx[k] = indexes
                out_coords[k] = target_coordinates

        return out_idx, out_coords

    else:
        return None, None


def _get_flat_ss(info: pd.Series,
                 _level: str,
                 start: int,
                 end: int,
                 full_seqs: bool = True):
    """
    Extracts flat splice site indexes for sequences 
    with surrounding features up to a given level, 
    associated with a given transcript id.

    :param pd.Series info: Information regarding 
    coordinates of surrounding features.
    :param str _level: Max level for which there
    are surrounding features available.
    :param int start: Start coordinate (0-based index) of the flat 
    sequence, after accounting for the extensions.
    :param int end: End coordinate of the flat sequence,
    after accounting for the extensions
    :param bool full_seqs: Whether `start` and `end`
    coordinates represent the true start and end of 
    the sequence at a given surrounding level. If `False`,
    they represent start and end positions up to the 
    limit of spliceAI resolution.

    :return str: Flat acceptor indexes 
    :return str: Flat donor indexes
    """
    _info = info.copy()

    # Check out of scope indexes
    if not full_seqs:
        if _level != 0:
            cols = [
                "Start_upstream" + _level, "End_upstream" + _level, "Start", "End",
                "Start_downstream" + _level, "End_downstream" + _level
            ]

            _info[cols] = _info[cols].apply(lambda x: int(x)
                                            if start <= x <= end else pd.NA)

    # if target is exon, we know
    # what's upstream and downstream
    if _info.Feature == "exon" and _level == "_2":

        if _info.Strand == "+":
            ups_donor = _info["End_upstream" + _level] - start
            target_donor = _info.End - start
            down_donor = _info["End_downstream" + _level] - start
            ups_accept = _info["Start_upstream" + _level] - start
            target_accept = _info.Start - start
            down_accept = _info['Start_downstream' + _level] - start

        else:
            ups_donor = end - _info["Start_upstream" + _level]
            target_donor = end - _info.Start
            down_donor = end - _info["Start_downstream" + _level]
            ups_accept = end - _info["End_upstream" + _level]
            target_accept = end - _info.End
            down_accept = end - _info['End_downstream' + _level]

        # Substract additional 1 from donor idx to represent idx for last exon position, just like spliceai
        donors = [ups_donor - 1, target_donor - 1, down_donor - 1]
        acceptors = [ups_accept, target_accept, down_accept]

    elif _info.Feature == "exon":

        if _info.Strand == "+":
            target_donor = _info.End - start
            target_accept = _info.Start - start

        else:
            target_donor = end - _info.Start
            target_accept = end - _info.End

        donors = [pd.NA, target_donor - 1, pd.NA]
        acceptors = [pd.NA, target_accept, pd.NA]

    acceptor_idx = ';'.join(str(i) for i in acceptors)
    donor_idx = ';'.join(str(i) for i in donors)

    return acceptor_idx, donor_idx


def generate_spliceAI_input_from_neighbour_df(
    df: pd.DataFrame,
    fasta: Union[str, Fasta],
    outbasename: str,
    extend_borders: int = 0,
    full_seqs: bool = True,
    fixed_len_seqs: bool = True,
):
    """
    Generates spliceAI input sequences from a dataframe
    of target features with upstream and downstream
    intervals.

    It will generate the splice site indexes of the features
    surrounding our target feature of interest (e.g. exon) for 
    the associated transcript ID.

    Additionally, two sets of fasta sequences can be written:
    If `full_seqs` is `True`:
        - Complete sequences ranging from the start of the highest 
    upstream level available (by default 2) up to the end of the 
    feature represented by the highest downstream level available. 
    Depending on intron sizes, these sequences may be very large.
    If `extend_borders` is > 0, this number of nucleotides will 
    be additionally extracted on each side.

    If `fixed_len_seqs` is `True`:
    - Sequences surrounding our target feature up to a maximum size 
    of 5000 nucleotides on each side. This is the maximum spliceAI
    resolution we can get to see an long-range effects on splice site
    definition. This set of sequences may be much shorter than the 
    complete one, which has a great impact on spliceAI inference time.
    If the length from the upstream to the downstream features
    is small (less than 10000 + average exon size), this output will be
    actually larger than the complete set. The `extend_borders` argument
    does not apply for this set.    
    Both

    :return: Return the names and numbers of features that were and were
    not discarded.
    """
    assert any(
        strategy for strategy in [full_seqs, fixed_len_seqs]
    ), "At least one strategy must be true to generate input sequences for SpliceAI"

    if list(df.filter(regex='stream')):
        # Extract level so that we know the
        # borders of the genomic sequence to extract
        try:
            level = max([
                int(x.split("_")[-1]) for x in list(df.filter(regex='stream'))
            ])
            _level = "_" + str(level)
        except ValueError:
            # level 1
            level, _level = "", ""

        # For now, rows with NAs in the intervals
        # upstream or downstream are removed,
        # meaning that only exons that are not
        # the first and last are kept (available
        # to be considered as cassette)
        _with_NAs = df[df.filter(regex='^Start.*stream',
                                 axis=1).isna().any(axis=1)]

        df = df[~df.filter(regex='^Start.*stream', axis=1).isna().any(axis=1)]

    else:
        level, _level = 0, 0
        _with_NAs = pd.DataFrame(
            columns=['Chromosome', 'Start', 'End', 'Score', 'Strand'])

        # raise ValueError(
        #     "It seems there are not upstream and downstream intervals. "
        #     "Did you run 'extract_surrounding_features' method with "
        #     "'level' set to 0 ?")

    out_full, out_fixed, out_psi = [], [], []

    for k, v in df.iterrows():

        # Get full sequences
        if full_seqs:

            if v.Strand == "+":
                start = "Start_upstream" + _level if level != 0 else "Start"
                end = "End_downstream" + _level if level != 0 else "End"

            else:
                start = "Start_downstream" + _level if level != 0 else "Start"
                end = "End_upstream" + _level if level != 0 else "End"

            seq = get_fasta_sequences(v,
                                      fasta=fasta,
                                      start_col=start,
                                      end_col=end,
                                      slack=extend_borders)
            spanning_coords = str(v[start] + 1 -
                                  extend_borders) + "-" + str(v[end] +
                                                              extend_borders)
            header = v.Chromosome + ":" + spanning_coords + "({})".format(
                v.Strand)
            header_long = header + "_" + v.transcript_id

            acceptor_idx, donor_idx = _get_flat_ss(v,
                                                   _level,
                                                   start=v[start] -
                                                   extend_borders,
                                                   end=v[end] + extend_borders)
            out_full.append([
                header, header_long, seq, acceptor_idx, donor_idx,
                v.transcript_id, v.Name
            ])

        if fixed_len_seqs:
            # Get sequences fixed at max SpliceAI resolution
            seq = get_fasta_sequences(v,
                                      fasta=fasta,
                                      start_col='Start',
                                      end_col='End',
                                      slack=5000)
            header = v.Chromosome + ":" + str(v.Start + 1 - 5000) + "-" + str(
                v.End + 5000) + "({})".format(v.Strand)
            header_long = header + "_" + v.transcript_id

            start = v.Start - 5000 if v.Strand == "+" else v.End + 5000
            end = v.End + 5000 if v.Strand == "+" else v.Start - 5000
            acceptor_idx, donor_idx = _get_flat_ss(v,
                                                   _level,
                                                   start=min(start, end),
                                                   end=max(start, end),
                                                   full_seqs=False)
            out_fixed.append([
                header, header_long, seq, acceptor_idx, donor_idx,
                v.transcript_id, v.Name
            ])

        try:
            out_psi.append([
                header_long, v.Name, v.gene_name, v.dPSI,
                os.path.basename(outbasename)
            ])
        except AttributeError:
            pass

    ######################
    #### Write output ####
    ######################
    for i, out in enumerate([out_full, out_fixed]):

        out_flag = "" if i == 0 else "_fixed_at_5000bp"
        if len(out) > 0:

            df = pd.DataFrame.from_records(out,
                                           columns=[
                                               'header', 'header_long', 'seq',
                                               'acceptor_idx', 'donor_idx',
                                               'tx_id', 'Name'
                                           ]).drop_duplicates()

            df[['header', 'acceptor_idx', 'donor_idx', 'tx_id',
                'Name']].to_csv(outbasename + "_sequences_ss_idx{}.tsv".format(out_flag),
                                sep="\t",
                                index=False)

            write_fasta_sequences(df,
                                  outname=outbasename +
                                  "_sequences{}.fa".format(out_flag),
                                  seq_col='seq',
                                  header_col='header_long')

    if out_psi:
        pd.DataFrame.from_records(out_psi, columns=['seq_id', 'exon_coord',
                                                    'gene', 'dPSI', 'rbp_name']).to_csv(outbasename +
                                                                                        "_dPSI.tsv",
                                                                                        sep="\t",
                                                                                        index=False)

    write_bed_file(
        _with_NAs,
        name=os.path.join(outbasename + "_first_or_last_exon.tsv"),
        bed6=True,
        additional_fields=[x for x in list(_with_NAs) if 'Col_' in x])

    return df.shape[0], _with_NAs.shape[0]
