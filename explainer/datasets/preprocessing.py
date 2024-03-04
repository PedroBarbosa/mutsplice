import argparse
from audioop import tomono
from math import log
import os
from pathlib import Path
import glob
from time import sleep
from typing import Union, Literal, List
import pandas as pd
import pyranges as pr
import numpy as np
import shutil
import sys
from multiprocessing import cpu_count, Pool
import functools
from tqdm import tqdm
from vcfhandle.utils import validate_vcf
from gtfhandle.utils import file_to_bed_df, bed_is_ok, split_fasta_file
from gtfhandle.extract_surrounding_features import getFeaturesFromCache, write_output
from explainer.motifs.motif_scanning import Motifs, _call_parallel_motif_scanning, _concat_parallel_output, process_subset_args
from explainer.mutations.mutate_seqs import MotifsHits, MutateAtMotifLocation, MutateOverSequences
from explainer.datasets.manage_spliceai import SpliceAI
from explainer.datasets.utils import setup_logger
from explainer.datasets.tabular_dataset import TabularDataset
from loguru import logger

def _generate_default_parameters(**kwargs):
    """
    Generates default parameter values in case
    they were not present in the main class 
    constructor. 
    """

    kwargs['input_feature_type'] = kwargs.get('input_feature_type', 'exons')
    kwargs['input_extend_borders'] = kwargs.get('input_extend_borders', 100)
    kwargs['use_full_sequence'] = kwargs.get('use_full_sequence', True)

    kwargs['subset_rbps'] = kwargs.get('subset_rbps', 'encode')
    kwargs['motif_source'] = kwargs.get('motif_source', 'rosina2017')
    kwargs['motif_search'] = kwargs.get('motif_search', 'plain')
    kwargs['pvalue_threshold'] = kwargs.get('pvalue_threshold', 0.00005)
    kwargs['log_odds_threshold'] = kwargs.get('log_odds_threshold', 0.15)
    kwargs['min_motif_length'] = kwargs.get('min_motif_length', 5)

    kwargs['ss_idx_extend'] = kwargs.get('ss_idx_extend', 5000)
    kwargs['skip_ss_region'] = kwargs.get('skip_ss_region', False)

    kwargs['no_batch_predictions'] = kwargs.get('no_batch_predictions', False)
    kwargs['batch_size'] = kwargs.get('batch_size', 64)
    kwargs['save_spliceai_raw'] = kwargs.get('save_spliceai_raw', False)
    kwargs['spliceai_raw_results'] = kwargs.get('spliceai_raw_results', None)
    kwargs['verbosity'] = kwargs.get('verbosity', 0)
    return kwargs

  
class Preprocessing(object):
    """
    Creates the structure to do all the preprocessing routines
    """

    def __init__(self, data: Union[pr.PyRanges, pd.DataFrame],
                 do_gtf_queries: bool = True,
                 do_motif_scanning: bool = True,
                 do_mutations: bool = True,
                 run_spliceai: bool = False,
                 **kwargs):
        """
        Data preprocessing routines to generate
        datasets to explain spliceAI predictons

        :param Union[pr.PyRanges, pd.DataFrame] data: Input data to run the
        preprocessing pipeline. Depending on the flags provided, this input
        may be different

        :param bool do_gtf_queries
        :param bool do_motif_scanning
        :param bool do_mutations
        :param bool disable_mutsplice
        """
        
        kwargs = _generate_default_parameters(**kwargs)
        setup_logger(kwargs['verbosity'])
        
        os.makedirs(kwargs['out_dir'], exist_ok=True)
        self.out_dir = kwargs['out_dir']
 
        if kwargs['use_full_sequence']:
            seq_file = kwargs['outbasename'] + "_sequences.fa"
            ss_idx_file = kwargs['outbasename'] + "_sequences_ss_idx.tsv"
        else:
            seq_file = kwargs['outbasename'] + "_sequences_fixed_at_5000bp.fa"
            ss_idx_file = kwargs['outbasename'] + \
                "_sequences_ss_idx_fixed_at_5000bp.tsv"

        self.SEQS_PATH = os.path.join(os.path.join(
            self.out_dir, "1_seq_extraction"), seq_file)
        self.SS_IDX_PATH = os.path.join(os.path.join(
            self.out_dir, "1_seq_extraction"), ss_idx_file)

        if do_gtf_queries:
            self.generate_input_sequences_from_GTF(data, **kwargs)
    
        else:
            assert os.path.isfile(
                self.SEQS_PATH), "Input sequences ({}) do not exist. Please initialize Preprocessing class with the 'do_gtf_queries' flag set to true.".format(self.SEQS_PATH)

        if do_motif_scanning:

            self.motif_scanning(**kwargs)

        if do_mutations:
            self.datasets = self.mutagenesis(**kwargs)

        else:
            self.motif_hits = None
            self.datasets = self.SEQS_PATH

        if run_spliceai:
            logger.log('MAIN', 'SpliceAI')
            assert len(
                self.datasets) > 0, "No mutated sequences found. Please generate the putative datasets."

            # If mutsplice pipeline, each element is a fasta
            # with seq perturbations in relation to a single ref seq
            if isinstance(self.datasets, list):

                results = {}
                for _dataset in self.datasets:

                    results[Path(_dataset).stem] = self.run_spliceAI(seqs=_dataset,
                                                                     outname_=Path(
                                                                         _dataset).stem,
                                                                     **kwargs)

            else:

                results = self.run_spliceAI(seqs=self.datasets,
                                            outname_=Path(self.datasets).stem,
                                            seqs_are_mutated=False,
                                            **kwargs)

        elif kwargs['spliceai_final_results'] is None:
            logger.log('MAIN', 'SpliceAI will not be run and previous spliceai predictions were not provided. We are done here.')
            results = None

        else:
            if os.path.isdir(kwargs['spliceai_final_results']):
                files = glob.glob(
                    '{}/*output.tsv.gz'.format(kwargs['spliceai_final_results']))
                results = {}
                for f in files:
                    results[Path(f).stem] = pd.read_csv(f, sep="\t")

            elif os.path.isfile(kwargs['spliceai_final_results']):
                results = pd.read_csv(
                    kwargs['spliceai_final_results'], sep="\t")

            else:
                raise ValueError('{} must be a valid file or directory'.format(
                    kwargs['spliceai_final_results']))

        self.results = results

    def generate_input_sequences_from_GTF(self, data, **kwargs):
        """
        Process input intervals and generate spliceAI input taking
        into account splice site indexes and transcript structure
        """
        df_surrounding, _known, _absent_in_gtf, _discarded = getFeaturesFromCache(
            data, **kwargs)

        _kwargs = kwargs
        _kwargs['out_dir'] = os.path.join(
            _kwargs['out_dir'], '1_seq_extraction')
        os.makedirs(_kwargs['out_dir'], exist_ok=True)
        write_output(data, df_surrounding, _known,
                     _absent_in_gtf, _discarded, **_kwargs)
        logger.success('Done.')

    def motif_scanning(self, **kwargs):
        """
        Scan input sequence for the ocurrence of known motifs
        """
        final_motif_path = os.path.join(self.out_dir, '2_motif_scanning')      
        n_seqs = sum(1 for line in open(self.SEQS_PATH)
                     if line.startswith('>'))

        # If few seqs, or few motifs in the database to scan
        if (n_seqs < 2000) or (isinstance(kwargs['subset_rbps'], list) and len(kwargs['subset_rbps']) < 5):

            Motifs(seqs=self.SEQS_PATH,
                   outdir=final_motif_path,
                   subset_rbps=kwargs['subset_rbps'],
                   source=kwargs['motif_source'],
                   search=kwargs['motif_search'],
                   pvalue_threshold=kwargs['pvalue_threshold'],
                   logodds_threshold=kwargs['log_odds_threshold'],
                   min_motif_length=kwargs['min_motif_length'],
                   ss_idx=self.SS_IDX_PATH,
                   ss_idx_extend=kwargs['ss_idx_extend'],
                   log_level=kwargs['verbosity'])

        # Parallelize
        else:
            logger.info('Spliting fasta files for motif scanning parallelization')
            motif_out_path = os.path.join(final_motif_path, '_fasta_chunks')
            if os.path.exists(motif_out_path):
                shutil.rmtree(motif_out_path)
            os.makedirs(motif_out_path)
                
            chunks = split_fasta_file(self.SEQS_PATH, motif_out_path, 10)

            with Pool(cpu_count()) as p:
                tmpdirs = p.map(functools.partial(_call_parallel_motif_scanning, 
                                                  ss_idx_path=self.SS_IDX_PATH,
                                                  **kwargs), tqdm(chunks, total=len(chunks)))
            p.close()

            _concat_parallel_output(tmpdirs, final_motif_path)
            sleep(1)
            #[shutil.rmtree(d) for d in tmpdirs]
            #shutil.rmtree(motif_out_path)
            
    def mutagenesis(self, **kwargs) -> List:
        """
        Read motif hits and perform mutations at motif positions

        :return List: List with the path to all mutated sequences
        """

        hits_path = os.path.join(self.out_dir, '2_motif_scanning')
        mutated_path = os.path.join(self.out_dir, '3_mutated_seqs')

        if 'ss_idx_extend' in kwargs and kwargs['ss_idx_extend'] <= 0:
            restrict = False
            f = "ALL_MOTIF_MATCHES.tsv.gz"
        else:
            restrict = True
            f = "MOTIF_MATCHES.tsv.gz"

        self.motif_hits = MotifsHits(file=os.path.join(hits_path, f),
                                     file_format='plain')

        m = MutateAtMotifLocation(fasta=self.SEQS_PATH,
                                  motifs=self.motif_hits,
                                  outdir=mutated_path,
                                  outbasename=kwargs['outbasename'],
                                  abrogate=True,
                                  ss_idx=self.SS_IDX_PATH if restrict else None,
                                  ss_idx_extend=kwargs['ss_idx_extend'],
                                  ss_idx_skip_ssRegion=kwargs['skip_ss_region'])

        m.mutateMotifs()
        logger.success("Done")
        return glob.glob('{}/*fa'.format(mutated_path))

    def run_spliceAI(self, seqs: str,
                     outname_: str,
                     seqs_are_mutated: bool = True,
                     **kwargs):
        """
        Run spliceAI for a given set of mutated
        sequences

        :param str seqs: Fasta file with mutated sequences
        originated from the same original sequence
        :param bool seqs_are_mutated: Flag to indicate whether
        input seqs represent perturbation on a original reference
        sequences. Default: `True`. If `False`, runs spliceAI
        for a set of independent sequences.

        :param str outname_ : Basename for the output file. It is
        different from the overall outbasename present in kwargs

        :return pd.DataFrame: Df with all the spliceAI results
        """
        dataset_path = os.path.join(self.out_dir, '4_datasets')

        if seqs_are_mutated:
            mutated_path = os.path.join(self.out_dir, '3_mutated_seqs')
            metadata = os.path.join(
                mutated_path, kwargs['outbasename'] + '_all_metadata.tsv')
        else:
            metadata = None

        sp = SpliceAI(seqs,
                      metadata=metadata,
                      outbasename=outname_,
                      outdir=dataset_path,
                      extend_context=0,  # kwargs['input_extend_borders'],
                      ref_genome=kwargs['fasta'],
                      splice_site_idx=self.SS_IDX_PATH,
                      no_batch_predictions=kwargs['no_batch_predictions'],
                      batch_size=kwargs['batch_size'],
                      save_spliceai_raw=kwargs['spliceai_save_raw'],
                      raw_preds_path=kwargs['spliceai_raw_results'])

        logger.log('MAIN', 'Processing predictions ({})'.format(seqs))
        logger.info("Getting high-scoring positions on reference sequence")
        ref_ss = sp.get_preds_on_reference_seq()

        if seqs_are_mutated:
            logger.info("Extracting positions that differ upon mutations")
            donor_high, acceptor_high = sp.get_differing_positions(
                minimum_difference=0.01)

            logger.info("Generating summary")
            summary = sp.generate_summary(donor_high,
                                          acceptor_high,
                                          motif_hits=self.motif_hits.motif_matches)

            logger.info("Creating plots (max of 50 motifs displayed)")
            logger.debug(".. All ..")
            sp.plot_predictions(ref_ss, donor_high,
                                acceptor_high, max_motifs_to_display=50)

            logger.debug(".. Signif ..")
            sp.plot_significant_predictions(summary,
                                            ref_ss,
                                            sign_thresh=0.05,
                                            discard_opposite_effects=True,
                                            use_max_effect=True,
                                            max_motifs_to_display=50)

        else:
            logger.info("Generating summary")
            summary = sp.generate_summary_no_Mutsplice()
            
            logger.info("Creating plot")
            sp.plot_predictions_no_Mutsplice()

        logger.success("All done")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Generate a local dataset for explaining spliceAI "
                                                 "predictions, one per each row of the input.")
    parser.add_argument(
        dest='input', help='Input data that we want to explain.')
    parser.add_argument(
        dest='cache', help='Path to the directory where cache files are located.')
    parser.add_argument(
        dest='out_dir', help='Path where output files will be written')

    # parser.add_argument('-m', '--mode', required=True, choices=('local', 'global'), help='Running mode. Local mode will generate a '
    #                     'dataset from each sequence in the input by generating mutations in the neighborhood of the input. Global mode '
    #                     'will generate a single dataset based on all the sequences present in the input. Those sequences may or may not '
    #                     'be additionally mutated.')

    parser.add_argument('--fasta', type=str, help="Reference genome in fasta")
    parser.add_argument('--outbasename', help="Output basename")

    parser.add_argument('--input_has_header', action='store_true',
                        help="Whether input file has header row "
                             "when input is not in VCF or BED format.")
    parser.add_argument('--input_is_0_based', action='store_true',
                        help="Whether coordinates represented in 'input_coordinates_col' "
                             "are in 0-based coordinate system when input is not in "
                             "VCF or BED format. Default: 1-based system is assumed.")
    parser.add_argument('--input_coordinates_col', metavar="", type=int, default=0,
                        help="Column (0-based index) in input data where "
                             "coordinates of exons are located when input "
                             "is not in VCF or BED format. Default: '0'")
    parser.add_argument('--input_feature_type', choices=('exons', 'introns'),
                        default="exons", help="Which features the coordinates in "
                                              "the input data represent. Default: 'exons'.")

    parser.add_argument('--input_drop_overlaps', action='store_true', help="Whether overlaps in input features should be\
                        removed. Useful to avoid highly correlated instances in the final dataset (e.g. two very similar exons\
                        will have the same feature values since the surrounding context will be same (motif counts and sequence\
                        mutations.")

    parser.add_argument('--input_context_level', default=2, choices=[
                        0, 1, 2], type=int, help="Restrict context extraction to the given level of upstream and downstream features. Default: 2.")

    parser.add_argument('--input_extend_borders', type=int, default=100, help="Extend input sequences on each side by this number when\
                        --use_full_sequence is set. If set to 0, sequences will be extracted exactly up to the context level provided.\
                        Default: 100.")

    parser.add_argument('--skip_gtf_queries', action='store_true',
                        help='Skip GTF querying to generate sequences based on surrounding intervals.')
    parser.add_argument('--skip_motif_scanning',
                        action='store_true', help='Skip motif scanning stage.')
    parser.add_argument('--skip_mutagenesis',
                        action='store_true', help='Skip mutagenesis stage.')
    parser.add_argument('--skip_spliceai', action='store_true',
                        help='Skip running spliceAI and all subsequent analysis (if splice_raw_results or splice_final_results are set).')

    parser.add_argument('--gene_names', help="Restrict context extraction to the given "
                        "gene_names, one per line.")

    parser.add_argument('--gene_ids', help="Restrict context extraction to the given "
                        "gene_ids, one per line.")
    
    parser.add_argument('--transcript_ids', help="Restrict context extraction to the given "
                                                 "transcript IDs, one per line. Useful if not "
                                                 "interested on assigning transcript ID to exons "
                                                 "based on the rank score present in the GTF cache.")

    parser.add_argument('--motif_source', choices=['encode2020_RBNS', 'rosina2017',
                                                   'oRNAment', 'ATtRACT'], default='rosina2017',
                        help="Motif source, either in PWM or plain text format. Default: 'rosina2017'")
    parser.add_argument('--motif_search', choices=['plain', 'fimo'],
                        default='plain',
                        help="How to search motifs in the sequences. Default: 'plain'")

    parser.add_argument('--subset_rbps', nargs='+', default='encode',
                        help='Subset motif scanning for the given list '
                             'of RBPs, or to the RBPs belonging to a specific '
                             'set. It also acceptos a file as input, one RBP name '
                             'per line. Default: `encode` set, which is composed by '
                             'a list of RBPs from ENCODE with RNA-Seq kD data, '
                             'were described as involved in splicing, and I performed '
                             'differential splicing analysis')

    parser.add_argument('-p', '--pvalue_threshold', default=0.00005, type=float,
                        help='Maximum pvalue threshold from FIMO output to consider '
                             'a motif ocurrence as valid. Only relevant when "motif_search" '
                             '== "fimo". Default: "0.00005".')

    parser.add_argument('-l', '--log_odds_threshold', default=0.15, type=float,
                        help='Minimum log-odds value for a nucleotide in a given '
                             'position of the PWM for it to be considered as relevant. Default: 0.15')

    parser.add_argument('--min_motif_length', default=5, type=int,
                        help='Minimum length of a sequence motif to search in the sequences. Default. 5')

    parser.add_argument('--use_full_sequence', action='store_true', help='Whether to extract and predict the full sequence '
                        'from the start coordinate of the upstream feature to he end coordinate of the '
                        'downstream feature. Default: `False`: Use restricted sequence regions up to the resolution limit '
                        'of spliceAI (5000bp) on each side of the target feature.')

    parser.add_argument('--ss_idx_extend', type=int, default=5000, help='Max number of nucleotides to extend '
                        'from splice sites of cassette exons when scanning motifs and mutating '
                        'sequences. Default: 5000 (spliceAI resolution). If negative number, does not put any '
                        'restriction and sequences are mutated at their full length')

    parser.add_argument('--skip_ss_region', action='store_true', help='Do not mutate sequences at near splice site '
                        '(5bp upstream of acceptor, 5bp downstream of donor) intronic regions. '
                        'Default: False. Motifs locating at those regions will be eventually mutated')

    parser.add_argument('--no_batch_predictions', action='store_true', help='Do not make spliceAI inferences in batches. If set, '
                        'each sequence will be predicted individually.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to make spliceAI inferences.')

    parser.add_argument('--spliceai_save_raw', action='store_true',
                        help='Save spliceAI raw predictions for later use')
    parser.add_argument('--spliceai_raw_results',
                        help='SpliceAI raw results if it was previously run on the input data. Must be a directory where pickle files exist (automatically created when spliceAI is actually run).')

    parser.add_argument('--spliceai_final_results',
                        help='SpliceAI final results if it was previously run on the input data. Can be a directory (if results are stored in multiple files; e.g. multiple input exons) or the path\
                            to a file (e.g. if perturbations were perfomed on a single input sequence).')
    parser.add_argument('--verbosity', choices=[0, 1, 2], type=int, default=2, help='Verbosity level. Default: 0. Options: [0, 1, 2].')
    args = parser.parse_args()

    if args.motif_source == 'rosina2017' or args.motif_source == 'encode2020_RBNS':
        assert args.motif_search == "plain", "Motif source set ({}) is in plain format 'plain' search must be set.".format(
            args.motif_source)

    if any(args.input.endswith(x) for x in ['vcf.gz', 'vcf', 'vcf.bgz']):
        logger.info("Input file seems to be in VCF format. Validating it.")
        validate_vcf(args.input)

    elif any(args.input.endswith(x) for x in ['bed', 'bed.gz', 'bed.bgz']):
        logger.info("Input file seems to be in bed format. Validating it.")
        bed_is_ok(args.input)

    else:
        h = 0 if args.input_has_header else None

        df = file_to_bed_df(args.input, header=h,
                            col_index=args.input_coordinates_col,
                            is_0_based=args.input_is_0_based)

        kwargs = {'input_feature_type': args.input_feature_type,
                  'input_drop_overlaps': args.input_drop_overlaps,
                  'input_context_level': args.input_context_level,
                  'input_extend_borders': args.input_extend_borders,
                  'out_dir': args.out_dir,
                  'outbasename': Path(args.input).stem if args.outbasename is None else args.outbasename,
                  'gtf_cache': args.cache,
                  'fasta': args.fasta,
                  'transcript_ids': args.transcript_ids,
                  'gene_ids': args.gene_ids,
                  'gene_names': args.gene_names,
                  'motif_source': args.motif_source,
                  'motif_search': args.motif_search,
                  'subset_rbps': process_subset_args(args.subset_rbps),
                  'pvalue_threshold': args.pvalue_threshold,
                  'log_odds_threshold': args.log_odds_threshold,
                  'min_motif_length': args.min_motif_length,
                  'use_full_sequence': args.use_full_sequence,
                  'ss_idx_extend': args.ss_idx_extend,
                  'skip_ss_region': args.skip_ss_region,
                  'no_batch_predictions': args.no_batch_predictions,
                  'batch_size': args.batch_size,
                  'spliceai_final_results': args.spliceai_final_results,
                  'spliceai_save_raw': args.spliceai_save_raw,
                  'spliceai_raw_results': args.spliceai_raw_results,
                  'verbosity': args.verbosity}

        preprocess = Preprocessing(df,
                                   do_gtf_queries=not args.skip_gtf_queries,
                                   do_motif_scanning=not args.skip_motif_scanning,
                                   do_mutations=not args.skip_mutagenesis,
                                   run_spliceai=False if kwargs['spliceai_final_results'] or args.skip_spliceai else True,
                                   **kwargs)

        # TabularDataset(preprocess.results,
        #                args.out_dir,
        #                granularity='per_location',
        #                normalize_by_length=True,
        #                **kwargs)


if __name__ == "__main__":
    main()
