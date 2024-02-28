import argparse
from cmath import log
from loguru import logger
from gtfhandle.utils import dict_to_fasta
from explainer.datasets.utils import setup_logger
from explainer.datasets.manage_spliceai import SpliceAI
from explainer.mutations.mutate_seqs import MotifsHits

def main():
    parser = argparse.ArgumentParser(description="Run spliceAI directly from a set of fasta sequences")
    parser.add_argument(dest='seqs', help='Input sequences to make inferences')
    parser.add_argument(dest='outbasename', help="It can be the name of a splicing factor analyzed")
    parser.add_argument(dest='outdir', help='Output directory.')
    
    parser.add_argument('--fasta', type=str, help="Reference genome in fasta")
    parser.add_argument('--is_flat_input', action='store_true', help='Whether input sequences/intervals reflect flat regions from which we'
                        ' want a SpliceAI result, regardless of known splice site information. This is useful when the input is not related with cassette exons.')
    parser.add_argument('--report_all_positions', action='store_true', help='Whether to report in bedgraph predictions for all positions in the input')
    parser.add_argument('--splice_site_idx', help='Auxiliar file with info about where known splice sites occur in the fasta sequences.')
    parser.add_argument('--extend_context', type=int, default=0, help="Extend input sequences/intervals on each side by this number. Default: 0")
    parser.add_argument('--seqs_are_mutated', action='store_true', help='Whether input sequences are mutated (downstream processing will differ)')
    parser.add_argument('--motif_hits', help='Path to the output of motif scanning step.')
    parser.add_argument('--metadata', help='Metadata about each sequence in the input file when they are mutated')
    
    parser.add_argument('--no_batch_predictions', action='store_true', help='Do not make spliceAI inferences in batches. If set, '
                        'each sequence will be predicted individually.' )
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size to make spliceAI inferences.' )
    parser.add_argument('--spliceai_save_raw', action='store_true', help='Save spliceAI raw predictions for later use')
    parser.add_argument('--spliceai_raw_results',
                        help='SpliceAI raw results if it was previously run on the input data. Must be a directory where pickle files exist (automatically created when spliceAI is actually run).')
    parser.add_argument('--skip_plotting', action='store_true',
                        help='Skip plotting of spliceAI predictions.')
    args = parser.parse_args()

    if args.seqs_are_mutated:
        assert args.metadata, "Metadata file is required when input sequences are mutated"
        assert args.motif_hits, "Motif hits file is required when input sequences are mutated"
        m_obj = MotifsHits(file=args.motif_hits, file_format='plain')
    
    setup_logger(2)
    logger.info("Running spliceAI")

    sp = SpliceAI(args.seqs,
                  args.metadata,
                  args.outbasename,
                  args.outdir,
                  extend_context=args.extend_context,
                  ref_genome=args.fasta,
                  is_cassette_exon=not args.is_flat_input,
                  splice_site_idx=args.splice_site_idx,
                  no_batch_predictions=args.no_batch_predictions,
                  batch_size=args.batch_size,
                  save_spliceai_raw=args.spliceai_save_raw,
                  raw_preds_path=args.spliceai_raw_results)
    
    logger.log('MAIN', 'Processing predictions ({})'.format(args.seqs))
    logger.info("Getting high-scoring positions on reference sequence")
    ref_ss = sp.get_preds_on_reference_seq()

    if args.report_all_positions:
        logger.info("Generating bedgraph with predictions for all positions in the input")
        sp.report_all_predictions()

    if args.seqs_are_mutated:
        logger.info("Extracting positions that differ upon mutations")
        donor_high, acceptor_high = sp.get_differing_positions(minimum_difference=0.01)
        
        logger.info("Generating summary")
        summary = sp.generate_summary(donor_high, 
                                      acceptor_high, 
                                      motif_hits=m_obj.motif_matches)
        
        if not args.skip_plotting:
            
            logger.info("Creating plots (max of 50 motifs displayed)")

            logger.debug(".. All ..")
            sp.plot_predictions(ref_ss,
                                donor_high,
                                acceptor_high, 
                                max_motifs_to_display=50)

            logger.debug(".. Signif ..")
            sp.plot_significant_predictions(summary,
                                            ref_ss, 
                                            sign_thresh = 0.05,
                                            discard_opposite_effects=True,
                                            use_max_effect=True,
                                            max_motifs_to_display=50)
            
    else:
    
        if not args.skip_plotting:
            logger.info("Creating plot")
            sp.plot_predictions_no_Mutsplice()

        if sp.ref_ss_idx:
            logger.info("Generating summary")
            sp.generate_summary_no_Mutsplice()
    
    logger.success('All done')
    
if __name__ == '__main__':
    main()
