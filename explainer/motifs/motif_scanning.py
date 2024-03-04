from explainer.datasets.utils import _process_ss_idx, _get_loc_of_motif, setup_logger
from explainer.motifs.rbp_lists import RBP_SUBSETS
from gtfhandle.utils import fasta_to_dict
from typing import Union, Literal
import os
import re
import pandas as pd
import numpy as np
import pyranges as pr
from Bio import Seq
from itertools import product, chain
import subprocess
from tqdm import tqdm
from loguru import logger
tqdm.pandas()


def process_subset_args(subset: list):
    """
    Process subset RBP argument to return a
    list (RBPs) or str (group of RBPs in a set)
    to be searched in the motif scanning step

    :param list subset: Input argument

    :return Union[None, List, str]:
    """
    #RBPs present in a given database
    if not subset:
        return None

    #RBPs found in a file, one per line
    if len(subset) == 1 and os.path.isfile(subset[0]):
        return [rbp.rstrip() for rbp in open(subset[0], 'r')]
    
    #RBPs present in the set provided
    elif len(subset) == 1 and subset[0] in list(RBP_SUBSETS.keys()):
        return subset[0] 
    
    #RPBs provided in input argument (1 or more)
    return subset

def _call_parallel_motif_scanning(seqs: str,
                                  ss_idx_path: str,
                                  **kwargs):
    """
    Run motif scanning for a subset of innput seqs
    """
    import tempfile
    tmp_outdir = tempfile.mkdtemp()
    setup_logger(kwargs['verbosity'])
      
    Motifs(seqs=seqs,
           outdir=tmp_outdir,
           subset_rbps=kwargs['subset_rbps'],
           source=kwargs['motif_source'],
           search=kwargs['motif_search'],      
           pvalue_threshold=kwargs['pvalue_threshold'],
           logodds_threshold=kwargs['log_odds_threshold'],
           min_motif_length=kwargs['min_motif_length'],
           ss_idx=ss_idx_path,
           ss_idx_extend=kwargs['ss_idx_extend'],
           log_level=kwargs['verbosity'])
    
    return tmp_outdir

def _concat_parallel_output(outdirs: list,
                            final_outdir: str):
    """
    Concatenation of outputs of parallel runs

    :param list outdirs: Path to the output of a single parallel run
    :param str final_outdir: Final output directory
    """
    all_m_c,all_r_c, all_m_m, m_c, r_c, m_m = [], [], [], [], [], []
    for dir in outdirs:
        
        all_m_c.append(os.path.join(dir, 'ALL_MOTIF_COUNTS.tsv'))
        all_r_c.append(os.path.join(dir, 'ALL_RBP_COUNTS.tsv'))
        all_m_m.append(os.path.join(dir, 'ALL_MOTIF_MATCHES.tsv.gz'))
        
        m_c.append(os.path.join(dir, 'MOTIF_COUNTS.tsv'))
        r_c.append(os.path.join(dir, 'RBP_COUNTS.tsv'))
        m_m.append(os.path.join(dir, 'MOTIF_MATCHES.tsv.gz'))
    
    pd.concat(map(lambda file: pd.read_csv(file, sep="\t"), all_m_c)).to_csv('{}/ALL_MOTIF_COUNTS.tsv'.format(final_outdir), sep="\t", index=False)
    pd.concat(map(lambda file: pd.read_csv(file, sep="\t"), all_r_c)).to_csv('{}/ALL_RBP_COUNTS.tsv'.format(final_outdir), sep="\t", index=False)
    pd.concat(map(lambda file: pd.read_csv(file, sep="\t"), all_m_m)).to_csv('{}/ALL_MOTIF_MATCHES.tsv.gz'.format(final_outdir), 
                                                                             compression='gzip', 
                                                                             sep="\t", 
                                                                             index=False)
    
    pd.concat(map(lambda file: pd.read_csv(file, sep="\t"), m_c)).to_csv('{}/MOTIF_COUNTS.tsv'.format(final_outdir), sep="\t", index=False)
    pd.concat(map(lambda file: pd.read_csv(file, sep="\t"), r_c)).to_csv('{}/RBP_COUNTS.tsv'.format(final_outdir), sep="\t", index=False)
    pd.concat(map(lambda file: pd.read_csv(file, sep="\t"), m_m)).to_csv('{}/MOTIF_MATCHES.tsv.gz'.format(final_outdir),
                                                                        compression='gzip', 
                                                                         sep="\t",
                                                                         index=False)
    
class Motifs(object):
    """
    Class with representations of motif sequences
    """

    def __init__(self,
                 seqs: str,
                 outdir: str,
                 source: str = Literal['rosina2017', 'oRNAment', 'ATtRACT'],
                 search: str = Literal['plain', 'fimo'],
                 subset_rbps: Union[str, list] = "encode",
                 pvalue_threshold: float = 0.00001,
                 logodds_threshold: float = 0.15,
                 min_motif_length: int = 5,
                 ss_idx: Union[str, dict] = None,
                 ss_idx_extend: int = 5000,
                 **kwargs):
        """
        :param str seqs: Input sequences in fasta format
        :param str outdir: Output directory
        :param str source: Scan motifs from the given source.
        :param str search: How to do the motif search

        :param Union[str, list] subset_rbps: Subset motif scanning
        for the given list of RBPs, or to the RBPs belonging
        to a specific set. Default: `encode`, list of RBPs from
        ENCODE that were described as involved in splicing and
        we performed differential splicing analysis

        :param float pvalue_threshold: Maximum pvalue allowed
        to consider a hit as valid when motif scanning is
        performed with FIMO.

        :param float logodds_threshold: Minimum log-odds
        value for a nucleotide in a given position for it
        to be considered as relevant

        :param int min_motif_length: Minimum motif length 
        allowed. Default: `5`

        :param Union[str, dict] ss_idx: File/dict with indexes of splice sites
        for each of the input sequences. If set, output will
        include separate motif hits information for the region
        subset that surrounds the splice site of cassette exons

        :param int ss_idx_extend: Number of base-pairs extending
        from cassette exon splice sites that will be presented
        in the output when `ss_idx` is provided. Default: `5000`.
        """
        try:
            logger.log("MAIN", "Scanning motifs..")
        except ValueError:
            setup_logger(kwargs.get('verbosity', 2))
            
        _sources = ['oRNAment', 'ATtRACT', 'rosina2017', 'encode2020_RBNS']
        _search = ['plain', 'fimo']
        assert source in _sources, 'Wrong motif source database provided.Valid values: {}.'.format(
            _sources)

        assert search in _search, 'Wrong motif search strategy provided. Valid values: {}.'.format(
            _search)

        if search == "fimo":
            assert source not in ['rosina2017', 'encode2020_RBNS'], "{} motif source can not be used " \
                                                                    "when the search strategy == 'fimo'.".format(
                                                                        source)

        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        self.seqs = fasta_to_dict(seqs)
        self.fasta = seqs
        self.source = source
        self.search = search
        self.pvalue_threshold = pvalue_threshold
        self.logodds_threshold = logodds_threshold
        self.min_motif_len = min_motif_length
        self.ref_ss_idx_extend = ss_idx_extend

        if ss_idx:
            self.ref_ss_idx, _ = _process_ss_idx(self.seqs, ss_idx)
        else:
            self.ref_ss_idx = None
            
        if isinstance(subset_rbps, list):
            self.subset_rbps = subset_rbps
            
        elif isinstance(subset_rbps, str):
            try:
                self.subset_rbps = RBP_SUBSETS[subset_rbps]

            except KeyError:
                raise KeyError('RBP subset provided ({}) is not valid. '
                               'Please choose from an existiing set {}, or provide RBP names directly.'.format(
                                   subset_rbps, ['encode', 'rosina2017', 'encode_in_rosina2017', 'encode_in_attract']))
        else:
            self.subset_rbps = None

        if self.source in ['rosina2017', 'encode2020_RBNS']:
            self.motifs = self.read_rosina()

        else:

            self.motifs, self.pwm_ids_per_rbp, self.db = self._read_PWMs()

        if self.subset_rbps:
     
            self.motifs = {
                k: v
                for k, v in self.motifs.items() if k in self.subset_rbps
            }
            if len(self.motifs) == 0:
                raise ValueError('None of the RBPs provided in the "--subset_rbps" argument is present in the {} database.'
                    .format(self.source))

            elif len(self.motifs) != len(self.subset_rbps):
                absent = ','.join([
                    x for x in self.subset_rbps if x not in self.motifs.keys()
                ])
                logger.warning(
                    'Some RBPs provided are not in the {} database:\'{}\'.'.
                    format(self.source, absent))

        if len(self.motifs) > 0:
            raw_hits = self.scan_sequences()
            filtered_motifs = self.filter_output(raw_hits)
                
            if isinstance(filtered_motifs, pd.DataFrame):
                self.write_output(filtered_motifs)
  
    def scan_sequences(self):
        """
        Scan motif ocurrences in the set of sequences provided

        :return:
        """
        logger.info("Scanning for {} RBPs motif ocurrences in {} "
                     "sequences using a {} search".format(
                         len(self.motifs), len(self.seqs), self.search))

        if self.search == "plain":
            return self._scan_by_exact_matches()
        else:
            return self._scan_with_fimo()
        
    def _scan_by_exact_matches(self):
        """
        Scan motif occurrences by blind
        substring search in fasta sequence
        strings. It is located outside of the
        class so that parallel searches with 
        multiprocessing can be performed.

        Additionally, if ref_ss_idx is provided and the number 
        of different motifs to scan is higher than 20, 
        sequences to be spanned will be shortened to the max
        length of the SpliceAI resolution

        :param dict seqs: Input sequences to be scanned
        :param dict motifs: Motifs to scan
        :param dict ref_ss_idx: Splice site indexes. Used to 
        restrict motif scanning space within sequences

        :return pd.DataFrame: Df with positions in the
        sequences where each motif was found with 0-based coordinates

        :return pd.DataFrame: Df with the counts of each
        RBP on each sequence
        :return pd.DataFrame: Df with the counts of each
        motif of each RBP on each sequence
        """
        full_info = []

        # for each sequence
        for header, seq in self.seqs.items():

            # for each RBP
            for rbp_name, _motifs in self.motifs.items():

                # matches are 0-based
                matches = [[_ for _ in re.finditer(m, seq)] for m in _motifs]

                # for each motif match
                for motif_seq, positions in zip(_motifs, matches):
    
                    if positions:
                        # Explode matches:
                        [
                            full_info.append([
                                header, rbp_name, motif_seq,
                                p.start(),
                                p.end(), rbp_name + "_" + motif_seq
                            ]) for p in positions
                            ]

        # Generate clean dfs
        full_df = pd.DataFrame.from_records(full_info,
                                            columns=[
                                                "seq_id", "rbp_name",
                                                "rbp_motif", "start", "end",
                                                "rbp_name_motif"
                                            ])

        return full_df
    
    def _scan_with_fimo(self):
        """
        Scan for motif ocurrences using FIMO.

        If subset of RBPs is provided,
        intermediary PWM files will be
        created by selecting only
        the PWMs that belong to the subset
        of RBPs provided.

        :return pd.DataFrame: 0-based motif ocurrences 
        """

        logger.info("Scanning with FIMO")
        fimo_outdir = os.path.join(self.outdir, 'fimo_out')
        os.makedirs(fimo_outdir, exist_ok=True)
        base_cmd = ['fimo', '--norc', '--oc', fimo_outdir, '--thresh', str(self.pvalue_threshold), '--no-qvalue', '--bfile', '--motif--']

        # If subset by at least one RBP, update the arg list
        # of FIMO to just use the PWM belonging to those RBPs.
        if self.subset_rbps:
            subset_RB = list(
                chain(*[
                    self.pwm_ids_per_rbp[rbp_name]
                    for rbp_name in self.motifs.keys()
                ]))
            _aux = ['--motif'] * len(subset_RB)
            add_cmd = list(chain(*zip(_aux, subset_RB)))
            base_cmd.extend(add_cmd)

        fimo_cmd = base_cmd + [self.db, self.fasta]

        log_file = os.path.join(fimo_outdir, 'fimo.log')
        _p = subprocess.run(fimo_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)

        try:
            df_fimo_out = pd.read_csv(os.path.join(fimo_outdir, 'fimo.tsv'),
                                      comment='#',
                                      sep="\t",
                                      low_memory=False)

            rename_cols = {
                'sequence_name': 'seq_id',
                'matched_sequence': 'rbp_motif',
                'motif_alt_id': 'rbp_name',
                'stop': 'end'
            }

            ordered_cols = [
                'seq_id', 'rbp_name', 'rbp_motif', 'start', 'end', 'p-value'
            ]
            df_fimo_out.rename(columns=rename_cols, inplace=True)
            full_df = df_fimo_out[ordered_cols].drop_duplicates(
                ['seq_id', 'rbp_name', 'rbp_motif', 'start', 'end'])

            # Remove matches of motif that include positions with
            # low log-odds score (below the threshold provided)
            all_possible_features = []
            [
                all_possible_features.extend([k + "_" + x for x in v])
                for k, v in self.motifs.items()
            ]

            full_df['rbp_name_motif'] = full_df.rbp_name + \
                "_" + full_df.rbp_motif

            _n = full_df.shape[0]
            full_df = full_df[full_df.rbp_name_motif.isin(
                all_possible_features)]
            logger.debug("Number of hits removed due to the logodds "
                         "threshold set ({}): {}".format(
                             self.logodds_threshold, _n - full_df.shape[0]))

            # Remove matches not passing the p-value threshold
            _n = full_df.shape[0]

            full_df = full_df[full_df['p-value'] <= self.pvalue_threshold]
            logger.debug("Number of hits removed due to the p-value "
                         "threshold set ({}): {}".format(
                             self.pvalue_threshold, _n - full_df.shape[0]))

            full_df.start -= 1
            return full_df.drop(columns=['p-value'])

        except pd.errors.EmptyDataError:
            logger.log("MAIN", "No single match found by FIMO.")
    
    def filter_output(self, raw_hits: pd.DataFrame):
        """
        Filters motif results and adds additional information
        if splice sites information is available

        :param pd.DataFrame df: Results from motif scanning (by FIMO or plain search)

        :return pd.DataFrame: Filtered df with additional information 
        """
        if raw_hits is None or raw_hits.empty:
            raise ValueError('No motifs found given this experimental setup.')
        else:
            logger.info("Filtering motif results:")

            df = _redundancy_and_density_analysis(raw_hits)

            if self.ref_ss_idx is not None:
                logger.info("Mapping location of motifs")
                df = _get_loc_of_motif(df, self.ref_ss_idx)

            return df
        

    def write_output(self, df: pd.DataFrame):
        """
        Writes the output of the motif scanning step
        
        :param bool is_subset: Whether filtered df refers to subset 
        of input sequences, it writes tmp files to be later 
        concatenated
        """

        def _filter_by_scope(df: pd.DataFrame):
            """
            Filtes input df to include motif intervals 
            within the scope provided
            """
            _filt = []
            for seq_name, motifs_per_seq in df.groupby('seq_id'):

                cassette_idx = self.ref_ss_idx[seq_name][1]
                region_from_accept = max(
                    cassette_idx[0] - self.ref_ss_idx_extend, 0)
                region_from_donor = cassette_idx[1] + self.ref_ss_idx_extend

                _df = motifs_per_seq[(motifs_per_seq.Start >= region_from_accept)
                                     & (motifs_per_seq.End <= region_from_donor)]

                _filt.append(_df)

            return pd.concat(_filt)

        logger.info("Writing output")
        
        if 'Strand' in df.columns:
            df = df.drop('Strand', axis=1)
        orig_df = df.copy()

        # Get the names of all possible RBP_based features
        all_rbps = list(self.motifs.keys())
        all_rbps_detailed = [
            rbp_name + "_" + m
            for rbp_name, list_motifs in self.motifs.items()
            for m in list_motifs
        ]

        for i in range(0, 2):

            out_flag = "ALL_" if i == 0 else ""

            if i == 1:
                if self.ref_ss_idx is None:
                    break

                elif self.ref_ss_idx_extend <= 0:
                    logger.debug("There are no sequence regions to restrict motif "
                                 "location ('ref_ss_idx_extend == {}'). ".format(self.ref_ss_idx_extend))
                    break

                else:
                    df = _filter_by_scope(df)
                    logger.debug("Number of outside of scope ({} bp on "
                                 "each side of a cassette exon) motifs: {}".format(self.ref_ss_idx_extend, orig_df.shape[0] - df.shape[0]))

            # Count ocurrences per RBP and per individual RBP motif
            rbp_counts = df.groupby(
                ['seq_id', 'rbp_name']).size().unstack(fill_value=0)
            rbp_counts_detailed = df.groupby(
                ['seq_id', 'rbp_name_motif']).size().unstack(fill_value=0)

            # Fill motif df with absent occurrences
            absent_rbps_hits = [
                x for x in all_rbps if x not in list(rbp_counts)]
            absent_rbps_detailed_hits = [
                x for x in all_rbps_detailed if x not in list(rbp_counts_detailed)]

            d_rbps1 = pd.concat([pd.DataFrame(dict.fromkeys(absent_rbps_hits, 0),
                                              index=[0])] * rbp_counts.shape[0],
                                ignore_index=True).set_index(rbp_counts.index)

            d_rbps2 = pd.concat([pd.DataFrame(dict.fromkeys(absent_rbps_detailed_hits, 0),
                                              index=[0])] * rbp_counts_detailed.shape[0],
                                ignore_index=True).set_index(rbp_counts_detailed.index)

            rbp_counts = pd.concat([rbp_counts, d_rbps1], axis=1)
            rbp_counts_detailed = pd.concat(
                [rbp_counts_detailed, d_rbps2], axis=1)

            # Write outputs
            df.to_csv(self.outdir + "/{}MOTIF_MATCHES.tsv.gz".format(out_flag), compression='gzip', sep="\t", index=False)
            rbp_counts.to_csv(self.outdir + "/{}RBP_COUNTS.tsv".format(out_flag), sep="\t")
            rbp_counts_detailed.to_csv(self.outdir + "/{}MOTIF_COUNTS.tsv".format(out_flag), sep="\t")
        logger.success("Done")

    def _pwm_to_unambiguous(self, pwm: Union[dict, np.array]):
        """
        Reads PWMs as numpy arrays and returns a list
        of unambiguous motif sequences for which all
        their positions have a probability higher than
        a given threshold.

        :return list: List of all possible sequences for
        the given pwm
        """
        nuc_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        ambiguous_dict = {
            ''.join(sorted(v)): k
            for k, v in Seq.IUPAC.IUPACData.ambiguous_dna_values.items()
        }
        ambiguous_seq = ""

        if isinstance(pwm, dict):

            for _, _pwm in pwm.items():
                pwm = _pwm

        if isinstance(pwm, np.ndarray):

            r, c = np.where(pwm >= self.logodds_threshold)

            per_position = np.split(c,
                                    np.searchsorted(r, range(1, pwm.shape[0])))
            ambiguous_seq = ""
            for position in per_position:
                nucs = ''.join([nuc_map[i] for i in position])
                ambiguous_seq += ambiguous_dict[nucs]

        return list(
            map(
                "".join,
                product(*map(Seq.IUPAC.IUPACData.ambiguous_dna_values.get,
                             ambiguous_seq))))

    def _read_PWMs(self, file_format: str = "meme"):
        """
        Reads a database of PWMs in MEME format and
        decomposes each PWM into all possible umambiguous
        sequences

        :param str file_format: Format of the PWM file

        :return dict: Dict with all the valid non-ambiguous
        sequences for each RBP
        :return dict: Dict with all the PWM IDs for each RBP
        """

        from collections import defaultdict
        motifs, final = defaultdict(list), defaultdict(list)

        if self.source == 'oRNAment':
            db = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "db/oRNAment/oRNAment_PWMs_database.txt")

        elif self.source == 'ATtRACT':
            db = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "db/ATtRACT/ATtRACT_PWMs_database.txt")

        # Get PWMs per RBP represented as np.arrays
        if file_format == "meme":

            logger.info(
                "Loading and processing PWM file from {} source".format(
                    self.source))
            motif_id, rbp_name, pwm = "", "", ""
            records = open(db, 'r')
            save_pwm = False
            for line in records:
                line = line.rstrip()

                if line.startswith('MOTIF'):
                    motif_id = line.split()[1]
                    rbp_name = line.split()[2]
                    pwm = []

                if line.startswith("letter-probability"):
                    save_pwm = True

                elif line and line[0].isdigit():
                    pwm.append([float(x) for x in line.split()])

                elif save_pwm:
                    motifs[rbp_name].append({motif_id: np.array(pwm)})
                    save_pwm = False

        logger.info("Generating unambiguous sequences using {} "
                     "as the minimum log-odds score".format(
                         self.logodds_threshold))
        # Convert PWMs to unambiguous sequences
        too_short = []
        for rbp_name, _motifs in motifs.items():
            per_rbp_motifs, _too_short = set(), set()

            for pwm in _motifs:
                _flat = self._pwm_to_unambiguous(pwm)
                flat_good = set()
                for x in _flat:
                    if len(x) >= self.min_motif_len:
                        flat_good.add(x)
                    else:
                        _too_short.add(x)
                per_rbp_motifs.update(flat_good)

            too_short.extend(list(_too_short))
            final[rbp_name] = list(per_rbp_motifs)

        pwd_ids_per_RBP = {
            rbp_name: set().union(*(pwm.keys() for pwm in pwms))
            for rbp_name, pwms in motifs.items()
        }

        logger.debug("Number of motifs removed due to short size (< {}): {}".format(
            self.min_motif_len, len(too_short)))
        return final, pwd_ids_per_RBP, db

    def read_rosina(self):
        """
        Reads additional file 1 from rosina et al 2017
        paper and returns a dictionary with all the non-ambiguous
        motifs for each RBP

        :return dict:
        """
        if self.source == "rosina2017":
            file_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "db/")
            motifs = open(file_path + "rosina2017_motifs.txt", 'r')

        elif self.source == "encode2020_RBNS":
            file_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "db/RBNS_Encode/")
            motifs = open(file_path + "encode2020_RBNS_motifs.txt", 'r')

        out = {}
        too_short = []
        for line in motifs:

            line = line.rstrip()
            if line.startswith('>'):
                rbp_name = line[1:].split("|")[0]
                _too_short = set()
                flat_good = set()

            elif line.startswith('*'):
                if rbp_name in out.keys():
                    raise ValueError(
                        "Repeated RBP name in file ({}).".format(rbp_name))

                for m in line[1:].split("|"):

                    if len(m) >= self.min_motif_len:
                        flat_good.update({m})

                    else:
                        _too_short.update({})

                out[rbp_name] = list(flat_good)
                too_short.extend(list(_too_short))

            if line.startswith("MOUSE"):
                break

        logger.debug("Number of motifs removed due to short size (< {}): {}".format(
            self.min_motif_len, len(too_short)))

        return out
    
    
def _redundancy_and_density_analysis(df: pd.DataFrame):
    """
    Finds overlaps in motif occurrences and do
    several operations:
        - Removes self contained motifs of the
    same RBP so that those occurrences are counted
    once.
        - Flags partially overlapped motifs of
    the same RBP together with motifs in close
    proximity (up to 5bp distant) as high-density
    region for that RBP.
        - Aggregates duplicate hits where
    multiple RBPs share the exact same motif [NOT DONE NOW]
    :param pd.DataFrame df: Df with all the hits
    :return pr.PyRanges: Subset of original df
    """
    df = df.reset_index(drop=True)

    _df = df.copy()
    _df = pr.PyRanges(
        _df.rename(columns={
            'seq_id': 'Chromosome',
            'start': 'Start',
            'end': 'End'
        }))

    logger.info("Self contained hits analysis..")
    _df = _remove_self_contained(_df)

    logger.info("Proximity hits analysis")
    _df = _tag_high_density(_df)

    #logger.info("Flagging duplicate hits across multiple RBPs")
    #_df = self._remove_duplicate_hits(_df)

    if isinstance(_df, pr.PyRanges):
        _df = _df.as_df()

    return _df.rename(columns={'Chromosome': 'seq_id'})


def _remove_duplicate_hits(gr: pr.PyRanges) -> pd.DataFrame:
    """
    Remove duplicate hits when different
    RBPs have the same motif
    """

    exact_cols = ['Chromosome', 'Start', 'End']

    df = gr.df
    g = df.groupby(exact_cols)

    no_dup = df[g['Start'].transform('size') == 1]
    dup = df[g['Start'].transform('size') > 1]

    if dup.empty:
        return pr.PyRanges(no_dup).sort()

    else:
        to_aggregate = ['rbp_name', 'rbp_motif', 'rbp_name_motif',
                        'has_self_submotif', 'has_other_submotif',
                        'is_high_density_region', 'n_at_density_block']

        #g = dup.groupby(exact_cols)
        # x = g.agg("first")
        # x.update(g.agg({"rbp_name": ";".join, "rbp_motif": ";".join}))
        # x = x.reset_index()

        #out = dup.drop_duplicates(subset=exact_cols)
        #out = dup.groupby(exact_cols)['rbp_name'].agg(list)

        out = dup.groupby(exact_cols)[to_aggregate].agg(
            ';'.join).reset_index().dropna()

        if not no_dup.empty:
            out = pd.concat([out, no_dup])

        return pr.PyRanges(out).sort()


def _tag_high_density(gr: pr.PyRanges) -> pr.PyRanges:
    """
    Tags motifs of the same RBP and other RBP located in close
    proximity (motifs that either overlap, or are
    up to 5bp apart)
    :param pd.DataFrame _any_overlaps:
    :return pr.PyRanges:
    """
    logger.debug('.. clustering proximal ..')
    proximal_hits = gr.cluster(slack=5).df

    proximal_hits['Size'] = proximal_hits.groupby(
        'Cluster')['Start'].transform(np.size)

    no_high_density = proximal_hits[proximal_hits.Size == 1].copy()
    no_high_density['is_high_density_region'] = False
    no_high_density['n_at_density_block'] = 1

    high_density = proximal_hits[proximal_hits.Size > 1].copy()
    high_density['is_high_density_region'] = True
    high_density['n_at_density_block'] = high_density.Size

    return pr.PyRanges(pd.concat([no_high_density, high_density]).drop(columns=['Cluster', 'Size']))


def _remove_self_contained(gr: pr.PyRanges) -> pr.PyRanges:
    """
    Flags motif ocurrences that are fully contained 
    within other motif (whether it is the same RBP or not)
    Removes those motif ocurrences of the same RBP 
    that are self contained.
    """
    logger.debug('.. clustering overlaps ..')

    _gr = gr.cluster(by='rbp_name', slack=-4)
    _gr.Length = _gr.lengths()
    df = _gr.df

    longest = df.groupby("Cluster").Length.idxmax()
    l = pr.PyRanges(df.reindex(longest))

    j = gr.join(l)

    to_drop_cols = ['Chromosome', 'Start', 'End',
                    'rbp_name', 'rbp_motif', 'rbp_name_motif']
    to_clean_cols = ['Cluster', 'Length', '_merge']
    df.drop(columns=to_clean_cols[:-1], inplace=True)

    contained = j[((j.Start >= j.Start_b) & (j.End < j.End_b)) |
                  ((j.Start > j.Start_b) & (j.End <= j.End_b))]

    if contained.empty:
        df['has_self_submotif'] = False
        df['has_other_submotif'] = False
        logger.debug('.. no self containments found ..')

    else:
        #####################
        # Per RBP contained #
        #####################
        logger.debug('.. self contained hits ..')

        contained_same_rbp = contained[contained.rbp_name ==
                                       contained.rbp_name_b]

        if contained_same_rbp.empty:
            df['has_self_submotif'] = False
            logger.debug(
                '.. no self contained hits found for the same RBP ..')

        else:
            contained_same_rbp.has_self_submotif = True
            contained_same_rbp = contained_same_rbp.df

            # Remove self.contained rows per RBP
            df = pd.merge(
                df, contained_same_rbp[to_drop_cols], on=to_drop_cols, how='left', indicator=True)
            df = df.loc[df._merge == 'left_only']
            logger.debug('.. {} hits removed ..'.format(
                contained_same_rbp.shape[0]))

            # Add self.contained.tag
            contained_same_rbp.drop(columns=to_drop_cols[1:], inplace=True)
            contained_same_rbp.columns = contained_same_rbp.columns.str.rstrip(
                '_b')

            contained_same_rbp = contained_same_rbp.drop_duplicates()

            df = pd.merge(df, contained_same_rbp, how='left',
                          on=to_drop_cols).drop(columns=to_clean_cols)
            df['has_self_submotif'] = df.has_self_submotif.fillna(False)

        #######################
        # Other RBP contained #
        #######################
        logger.debug('.. other contained hits ..')
        contained_other_rbp = contained[contained.rbp_name !=
                                        contained.rbp_name_b]
        if contained_other_rbp.empty:
            df['has_other_submotif'] = False
            logger.debug(
                '.. no self contained found hits for other RBPs ..')

        else:
            contained_other_rbp.has_other_submotif = True
            contained_other_rbp = contained_other_rbp.df

            # Add other.contained.tag
            contained_other_rbp.drop(
                columns=to_drop_cols[1:], inplace=True)
            contained_other_rbp.columns = contained_other_rbp.columns.str.rstrip(
                '_b')
            contained_other_rbp = contained_other_rbp.drop_duplicates()
            df = pd.merge(df, contained_other_rbp, how='left',
                          on=to_drop_cols).drop(columns=to_clean_cols[:-1])
            df['has_other_submotif'] = df.has_other_submotif.fillna(False)
            logger.debug('.. {} hits flagged ..'.format(
                contained_other_rbp.shape[0]))

    return pr.PyRanges(df)





