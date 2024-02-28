## InterpretSplicing
Set of modules to aid on the interpretation of SpliceAI, a neural network that models RNA splicing.

Starting from a list of exons in a file, one can easily construct a large dataset with thousands of features that may explain spliceAI predictions for those same exons.

A file with a list of exons is necessary:

```
chr10:100250248-100250332
chr10:100256262-100256476
chr10:100243698-100243772
```

Then, simply call:

```
from gtfhandle.utils import file_to_bed_df
from explainer.datasets.generate_dataset import Preprocessing
from explainer.datasets.global_explain import GlobalDataset

df = file_to_bed_df(bed_file)
preprocess = Preprocessing(df,
                        do_gtf_queries=True,
                        do_motif_scanning=True,
                        do_mutations=True,
                        run_spliceai=True
                        **kwargs)

d = GlobalDataset(preprocess.results, out_dir, **kwargs)
```

There are many custom settings one can add. Particularly `gtf_cache`, `out_dir`, `fasta` are mandatory:

```
kwargs = {'input_feature_type': args.input_feature_type,
          'out_dir': args.out_dir,
          'outbasename': Path(args.input).stem if args.outbasename is None else args.outbasename,
          'gtf_cache': args.cache,
          'fasta': args.fasta,
          'transcript_ids': args.transcript_ids,
          'motif_source': args.motif_source,
          'motif_search': args.motif_search,
          'subset_rbps': args.subset_rbps,
          'qvalue_threshold': args.qvalue_threshold,
          'log_odds_threshold': args.log_odds_threshold,
          'min_motif_length': args.min_motif_length,
          'use_full_sequence': args.use_full_sequence,
          'ss_idx_extend': args.ss_idx_extend,
          'no_batch_predictions': args.no_batch_predictions,
          'batch_size': args.batch_size,
          'spliceai_final_results': args.spliceai_final_results,
          'spliceai_raw_results': args.spliceai_raw_results}
```

To run spliceAI on flat sequences (regardless of gene structure information, and exon definition), just call directly the spliceAI script (which accepts fasta and bed is input):

```
python make_inferences.py --fasta ref_genome.fa --is_flat_input --outbasename test intervals.bed out_dir
```
