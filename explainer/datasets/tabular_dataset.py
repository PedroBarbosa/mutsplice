
from collections import defaultdict
from loguru import logger
from platform import java_ver
from posixpath import join
from natsort.utils import do_decoding
import pandas as pd
import numpy as np
from typing import Union, Dict
import os
import glob
import functools
import shutil
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from explainer.motifs.rbp_lists import RBP_SUBSETS
from explainer.motifs.motif_scanning import Motifs, _call_parallel_motif_scanning, _concat_parallel_output
from explainer.datasets.utils import _get_len, _location_specific_normalization, setup_logger
from gtfhandle.utils import split_fasta_file


class TabularDataset(object):

    def __init__(self,
                 data: Union[pd.DataFrame, Dict],
                 outdir: str,
                 granularity: str = 'motif',
                 normalize_by_length: bool = True,
                 **kwargs):
        """
        Process mutsplice output to generate tabular datasets 
        of motif occurences

        :param Union[pd.DataFrame, Dict]: Input data
        :param str outdir: Output directory
        :param str granularity: Level of detail to generate feature values. Default: 'motif'
            Possible values:
             - motif: Counts motif ocurrences per RBP in the input sequences
             - submotif: Counts submotif ocurrences (each motif sequence is independent) per RBP in the input sequences
             - per_location: Counts motif occurrences per RBP on each discrete location of the input sequence (e.g. exon_upstream, intron_upstream, etc) 

        :param bool normalize_by_length: Normalize motif occurences by the length of the input sequence (if 'granularity' == 'motif' or 'submotif'), or by
        the length of each location (if 'granularity' == 'per_location')

        """
        logger.remove()
        setup_logger(kwargs.get('verbosity', 2))
        logger.log('MAIN', 'Tabular dataset generation')
        assert granularity in [
            'motif', 'submotif', 'per_location'], 'Wrong value for the granularity argument.'
        self.outdir = outdir
        self.granularity = granularity
        self.normalize_by_length = normalize_by_length

        if isinstance(kwargs['subset_rbps'], str):
            self.rbps = RBP_SUBSETS[kwargs['subset_rbps']]
        else:
            self.rbps = kwargs['subset_rbps']

        # Mutagenesis was performed.
        # Each sequence is a neighbour
        # dataset of an original one
        if isinstance(data, dict):
            self.is_from_mutagenesis = True
            dfs = list(data.values())
            self.data = pd.concat(dfs).reset_index(drop=True)

        else:
            if isinstance(data, str):
                if os.path.isfile(data):
                    data = pd.read_csv(data, sep="\t")
                elif os.path.isdir(data):
                    files = glob.glob('{}/*gz'.format(data))
                    data = pd.concat(map(functools.partial(pd.read_csv, sep='\t'), files))
                    
            if len(list(data.filter(like='_effect'))) > 1:
                self.is_from_mutagenesis = True

            else:
                self.is_from_mutagenesis = False

            self.data = data

        input_seqs, motifs_path = self.process_input_sequences(**kwargs)
        self.data = self.data.set_index('seq_id')
        self.data = _get_len(self.data, input_seqs)
        
        if self.is_from_mutagenesis:
            out_dir = os.path.dirname(input_seqs)
            self.ss_idx = self._fix_input_from_mutsplice(out_dir)
            self.run_motif_scanning(input_seqs, out_dir, **kwargs)

        if self.granularity == 'per_location':
            motifs_df = pd.read_csv(os.path.join(
                motifs_path, 'ALL_MOTIF_MATCHES.tsv.gz'), sep="\t")
            self.count_occurrences_per_location(motifs_df)
        else:
            self.count_occurences_full_seq(motifs_path)

        # Remove some cols
        self.data = self.data.drop(columns=list(
            self.data.filter(regex='^start|^end|^other_', axis=1)))

        cass_cols = ['ref_acceptor_cassette', 'ref_donor_cassette']
        self.data['average_cassette_strength'] = self.data[cass_cols].mean(
            axis=1, numeric_only=True)
        self.data.to_csv('{}/TABULAR_DATASET.tsv.gz'.format(self.outdir),
                         sep="\t", index=False, compression='gzip')

    def process_input_sequences(self, **kwargs) -> str:
        """
        Locates input sequences for the given dataset,
        processes their splice sites, and detects path
        for the motif scanning stage

        :return str: Path where the input sequences
        are located 
        :return str: Path where the motifs are located 
        (if not from mutagenesis) or where they
        are going to be written (if from mutagenesis)
        """
        # If perturbation data, create a single fasta file for all sequences
        if self.is_from_mutagenesis:
            # Rename seq_id because in mutation data it refers to the original seq
            self.data['seq_id'] = self.data.id
            
            # If exec called right after Preprocessing
            if 'out_dir' in kwargs:
                mut_seqs_path = os.path.join(
                    kwargs['out_dir'], "3_mutated_seqs")

            # If independent call, requires output directory
            # to be the same as the Preprocessing run before
            else:
                mut_seqs_path = os.path.join(self.outdir, "3_mutated_seqs")

            if not os.path.isdir(mut_seqs_path):
                raise NotADirectoryError(
                    '{} directory not found. Please set the output directory to be the parent directory of the Preprocessing routine output'.format(mut_seqs_path))

            # Mutated fasta seqs is the input
            seqs = glob.glob('{}/*fa'.format(mut_seqs_path))

            out_folder = "perturbation_results"
            motifs_path = os.path.join(self.outdir, out_folder)
            os.makedirs(motifs_path, exist_ok=True)

            # Cat sequences into single file
            input_seqs = os.path.join(motifs_path, '1_input_seqs.fa')
            with open(input_seqs, 'wb') as wf:
                for f in seqs:
                    with open(f, 'rb') as rf:
                        shutil.copyfileobj(rf, wf)
            wf.close()
            
        else:
            motifs_path = os.path.join(self.outdir, "2_motif_scanning")
            input_seqs_path = os.path.join(self.outdir, '1_seq_extraction/')
            assert os.path.exists(
                input_seqs_path), "Input sequences path does not exist ({}). Did you set the 'skip_gtf_queries' flag in the preprocessing?".format(input_seqs_path)

            assert os.path.exists(
                motifs_path), "Motif scanning path does not exist ({}). Did you set the 'skip_motif_scanning' flag in the preprocessing?".format(motifs_path)

            input_seqs = glob.glob('{}/*fa'.format(input_seqs_path))
            if len(input_seqs) > 1:
                raise ValueError('More than one fasta file found (*fa) in {} directory.\
                    Which file points to the sequences represented in this dataset ?'.format(input_seqs_path))

            elif len(input_seqs) == 0:
                raise ValueError(
                    'No input sequences found in {}'.format(input_seqs_path))

            else:
                input_seqs = input_seqs[0]

            assert os.path.exists(
                motifs_path), "Motif scanning path does not exist ({}). Did you set the 'motif_scanning' flag in the preprocessing?".format(motifs_path)

        return input_seqs, motifs_path

    def run_motif_scanning(self,
                           input_seqs,
                           output_dir,
                           **kwargs):
        """
        Perform motif scanning for each of the perturbations 
        to be later aggregated into features of the dataset

        :param str input_seqs: Path to the input sequences
        :param str output_dir: Output directory
        """
        final_motif_path = output_dir   
        n_seqs = sum(1 for line in open(input_seqs) if line.startswith('>'))

        # Parallelize
        if n_seqs > 10:
            logger.info('Spliting fasta files')
            motif_out_path = os.path.join(final_motif_path, '_fasta_chunks')
            if os.path.exists(motif_out_path):
                shutil.rmtree(motif_out_path)
            os.makedirs(motif_out_path)
                
            chunks = split_fasta_file(input_seqs, motif_out_path, 10)
            logger.info('Done')

            with Pool(cpu_count()) as p:
                tmpdirs = p.map(functools.partial(_call_parallel_motif_scanning, 
                                                  ss_idx_path=self.ss_idx,
                                                  **kwargs), tqdm(chunks, total=len(chunks)))
            p.close()
            _concat_parallel_output(tmpdirs, final_motif_path)
            [shutil.rmtree(d) for d in tmpdirs]
            shutil.rmtree(motif_out_path)

        else:
            Motifs(fasta=input_seqs,
                outdir=final_motif_path,
                subset_rbps=self.rbps,
                source=kwargs['motif_source'],
                search=kwargs['motif_search'],
                qvalue_threshold=kwargs['qvalue_threshold'],
                logodds_threshold=kwargs['log_odds_threshold'],
                ss_idx=self.ss_idx,
                ss_idx_extend=kwargs['ss_idx_extend'])

    def count_occurences_full_seq(self, motifs_path: str):
        """
        Generates aggregated motif counts and merges that info
        to the main data

        :param str motifs_path: Path to the input motif
        scanning output, where motif frequencies are 
        already reported
        """
        logger.info("Adding motif ocurrences features")
        flag = 'RBP' if self.granularity == 'motif' else 'MOTIF'
        counts = os.path.join(motifs_path, '{}_COUNTS.tsv'.format(flag))
        motif_counts = pd.read_csv(counts, sep="\t")

        if self.normalize_by_length:
            logger.info("Normalizing by exon/intron length..")
            motif_counts = motif_counts.set_index('seq_id')
            _data = self.data.copy()
            _data = _data[['seq_id', 'len_seq']].drop_duplicates().set_index('seq_id')
            motif_counts = motif_counts.div(_data.len_seq, axis=0).reset_index()
  
        self.data = pd.merge(self.data, motif_counts, on='seq_id')

    def count_occurrences_per_location(self,
                                       motif_details: pd.DataFrame):
        """
        Generates aggregated RBP motif counts per input location and 
        merges that info to the main data

        :param pd.DataFrame motif_details: Detailed informatin about the motif
        scanning step
        """

        def _add_missing_data(df: pd.DataFrame,
                              possible_values: list):
            """
            Adds missing features as 0 counts to the dataframe
            """
            all_features = sorted(
                [rbp + "_" + feat for rbp in self.rbps for feat in possible_values])
            to_add = []
            for f in all_features:
                if f not in df.columns:
                    to_add.append(f)

            return pd.concat([df, pd.DataFrame(0, df.index, to_add)], axis=1)[all_features].copy().reset_index()

        _possible_loc = ["Intron_upstream_2", "Intron_upstream", "Intron_downstream", "Intron_downstream_2",
                         "Exon_upstream_fully_contained", "Exon_upstream_acceptor_region", "Exon_upstream_donor_region",
                         "Exon_cassette_fully_contained", "Exon_cassette_acceptor_region", "Exon_cassette_donor_region",
                         "Exon_downstream_fully_contained", "Exon_downstream_acceptor_region", "Exon_downstream_donor_region"]

        logger.info("Adding location features")
        # Boolean features will be represented as True counts
        # bool_feat = []
        # for feat in ['has_self_submotif', 'has_other_submotif, ''is_high_density_region', 'n_at_density_block']:
        #     if feat in motif_details.columns:
        #         aux = pd.pivot_table(motif_details[['seq_id', motif_col, feat]],
        #                             index='seq_id',
        #                             columns=[motif_col, feat],
        #                             aggfunc=len,
        #                             fill_value=0,
        #                             dropna=False)
        #         aux.columns = [col[0] + "_" + str(col[1]) for col in aux.columns.values]
        #         aux = aux[aux.columns.drop(list(aux.filter(regex='False')))]
        #         aux = aux.rename(columns=lambda x: re.sub('True', feat, x))

        #         bool_feat.append(_add_missing_data(aux, possible_values=[feat]))

        # boolean_loc_features = pd.concat(bool_feat, axis=1)

        # MOTIF LOCATION FEATURES
        location = pd.pivot_table(motif_details[['seq_id', 'rbp_name', 'location']],
                                  index='seq_id',
                                  columns=['rbp_name', 'location'],
                                  aggfunc=len,
                                  fill_value=0,
                                  dropna=False)

        location.columns = ["_".join(col).strip()
                            for col in location.columns.values]
        location = _add_missing_data(location,
                                     possible_values=_possible_loc)

        #motif_loc_feat = pd.merge(location, boolean_loc_features, on="seq_id")

        if self.normalize_by_length:
            logger.info("Normalizing by exon/intron length..")
            location = _location_specific_normalization(location, self.data)
    
        self.data = pd.merge(self.data, location, on="seq_id")

    def _fix_input_from_mutsplice(self, input_seqs_path: str) -> dict:
        """
        Generates file with splice site indexes for each of the perturbed sequences
        and fixes indexes of upstream/downstream exons that were truncated if 
        the pipeline was run with sequences fixed at 5000bp, so that proper 
        downstream mapping of the motifs can be performed.
        
        Additionally cleans some data
        
        :param str input_seqs_path: Path where perturbed input sequences were written
        
        :param dict: Dict with splice 
        """    
        # New ss idx
        cols = ['seq_id', 'rbp_name', 'target_coordinates',
                'start_exon_upstream', 'end_exon_upstream',
                'start_exon_cassette', 'end_exon_cassette',
                'start_exon_downstream', 'end_exon_downstream']
        
        if 'seq_id' not in self.data.columns:
            self.data = self.data.reset_index()

        _aux = self.data[cols]

        ss_idx = dict(zip(_aux.seq_id, zip(_aux[cols[3]], _aux[cols[4]],
                                            _aux[cols[5]], _aux[cols[6]],
                                            _aux[cols[7]], _aux[cols[8]])))

        ss_idx_f = os.path.join(input_seqs_path, '1_input_seqs_ss_idx.tsv')
        f = open(ss_idx_f, 'w')
        f.write('header\tacceptor_idx\tdonor_idx\ttx_id\tName\trbp_name\n')

        for k, v in ss_idx.items():
            v = list(v)

            ss_idx[k] = [v[0:2], v[2:4], v[4:6]]

            acc_idx = ';'.join([str(x) for x in v[0::2]])
            don_idx = ';'.join([str(x) for x in v[1::2]])

            _dup_seqs = _aux[_aux.seq_id == k]
            _dup_seqs.apply(lambda x: f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(k, acc_idx, don_idx,
                                                                                k.split("_")[
                                                                                    1],
                                                                                x.target_coordinates,
                                                                                x.rbp_name)), axis=1)
        f.close()
        
        # Fix <NA> upstream/downstream idx for later location mapping
        _ss_idx = {}
        for header, idx in ss_idx.items():
            seq_len = self.data[self.data.seq_id == header].len_seq.iloc[0]
   
            ups = idx[0]
            cass = idx[1]
            down = idx[2]
            
            ups = pd.Series(ups).fillna(0).astype('Int64').tolist()
            down = pd.Series(down).fillna(seq_len - 1).astype('Int64').tolist()
            _ss_idx[header] = [ups, cass, down]
        ss_idx = _ss_idx

        # Convert ref ss indexes and len to int
        to_int = ['start_exon_upstream', 'end_exon_upstream',
                  'start_exon_cassette', 'end_exon_cassette',
                  'start_exon_downstream', 'end_exon_downstream',
                  'len_intron_upstream2', 'len_intron_upstream',
                  'len_intron_downstream2', 'len_intron_downstream',
                  'len_exon_cassette', 'len_exon_upstream', 'len_exon_downstream']
        
        self.data[to_int] = self.data[to_int].astype('Int64')

        # Drop mutation metadata cols and mutation effects cols
        to_drop_cols = ['id', 'start', 'end', 'mutation', 'type',
                        'motif_start',  'motif_end', 'has_self_submotif', 'has_other_submotif',
                        'is_high_density_region', 'n_at_density_block',
                        'is_in_exon', 'location', 'distance_to_donor', 'distance_to_acceptor',
                        'other_acceptor_distance_to_mutation', 'other_donor_distance_to_mutation']

        self.data = self.data.drop(columns=to_drop_cols)
        self.data = self.data.drop(columns=self.data.filter(regex="_effect"))

        return ss_idx


# class LocalDataset(object):
#     """"
#     Representation of a local dataset
#     """

#     def __init__(self,
#                  data: pd.DataFrame,
#                  module: str = "exons",
#                  problem: str = "regression"):
#         """
#         Constructs an hypothetical dataset
#         from the output of the spliceAI
#         results processing pipeline

#         :param pd.DataFrame data: SpliceAI results
#         for each of the mutated sequences.

#         :param str input_dir: Directory where the
#         mutated sequences (the dataset) are located

#         :param str module: What do we want to find
#         explanations for ? Splicing of an exon (exons)
#         or effect of a genetic variant (variants)

#         :param str problem: How to treat the problem.
#         E.g. regression, classification
#         """
#         assert data.groupby('seq_id').ngroups == 1, "Input data must represent mutations " \
#                                                     "originated from the same original sequence."

#         self.name = data.iloc[0].target_coordinates
#         self.module = module
#         self.data = self.remove_useless_cols(data)

#         if problem == "classification":
#             self.assign_class()
#         elif problem == "regression":
#             self.assign_value()

#     def __str__(self):
#         return self.name

#     @property
#     def strength(self):
#         """
#         Returns the reference strength
#         of the cassette exon by calculating
#         the average between the donor and
#         acceptor score

#         :return float: Strength of the exon
#         """
#         return round(self.data.iloc[0][['Ref_Acceptor_Cassette',
#                                         'Ref_Donor_Cassette']].mean(), 3)

#     @property
#     def label(self):
#         _s = self.strength
#         if 0 < _s < 0.1:
#             return "No_exon"
#         elif 0.1 <= _s < 0.4:
#             return "Weak"
#         elif 0.4 <= _s < 0.8:
#             return "Intermediate"
#         else:
#             return "Strong"

#     def remove_useless_cols(self, data: pd.DataFrame):
#         """
#         Removes the columns that won't be used
#         as features in the training stage.

#         :param pd.DataFrame data: Input data
#         :return pd.DataFrame: Filtered data
#         """
#         cols = ['id', 'rbp_name', 'start', 'end', 'mutation', 'type', 'motif_start', 'motif_end',
#                 'target_coordinates', 'Ref_Acceptor_Cassette', 'Ref_Donor_Cassette', 'Is_in_exon',
#                 'Location', 'Distance_to_donor', 'Distance_to_acceptor', 'Acceptor_Upstream_effect',
#                 'Donor_Upstream_effect', 'Acceptor_Cassette_effect', 'Donor_Cassette_effect',
#                 'Acceptor_Downstream_effect', 'Donor_Downstream_effect', 'Other_Acceptor_effect',
#                 'Other_Acceptor_distance_to_mutation', 'Other_Acceptor_position', 'Other_Donor_effect',
#                 'Other_Donor_distance_to_mutation', 'Other_Donor_position']

#         return data[cols]

#     def assign_value(self):
#         """
#         Assign a single value for the
#         exon strength to each of the
#         sequence perturbation in the
#         dataset based on the average
#         of the splice site probabilities
#         for the donor and acceptor.

#         :return
#         """
#         if self.module == "exons":
#             _acceptor_scores = self.data['Ref_Acceptor_Cassette'] + \
#                 self.data['Acceptor_Cassette_effect']
#             _donor_scores = self.data['Ref_Donor_Cassette'] + \
#                 self.data['Acceptor_Cassette_effect']

#             _score = pd.concat(
#                 [_acceptor_scores, _donor_scores], axis=1).mean(axis=1)
#             self.data['Average_Cassette_Strength'] = _score

#         # print(self.data.Average_Cassette_Strength.value_counts())

#     def assign_class(self):
#         """
#         Assign categorical class to each of the
#         sequence perturbations in the dataset.

#         If module == `exons`, it will assign a
#         category to the exon under study based
#         on the average of the splice site probabilities
#         for the acceptor and donor.

#         :return:
#         """
#         self.assign_value()
#         self.data['label'] = pd.cut(self.data['Average_Cassette_Strength'],
#                                     bins=[0, 0.1, 0.4, 0.8, 1],
#                                     labels=["No_exon", "Weak", "Intermediate", "Strong"])
#         print(self.data.label.value_counts())
