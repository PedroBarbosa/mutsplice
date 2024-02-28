import argparse
from dis import dis
import os
from loguru import logger
from re import L
from sys import path
import pandas as pd
import pyranges
from typing import Union
from gtfhandle.features import Exons, Transcripts, compute_gc_and_hexamers, compute_length, insert_exons, extract_surrounding_features
from gtfhandle.utils import file_to_bed_df, open_fasta, write_bed_file, read_features_file, remove_overlaps
from explainer.datasets.utils import generate_spliceAI_input_from_neighbour_df


def getFeaturesFromCache(data: Union[pyranges.PyRanges, pd.DataFrame], **kwargs):

    if isinstance(data, pyranges.PyRanges):
        data = data.as_df()

    assert kwargs['gtf_cache'] is not None, "GTF cache required to extract genomic information."
    assert kwargs['fasta'] is not None, "Genome fasta file is required to extract genomic information"

    logger.log('MAIN', 'Reading cache file to extract genomics features..')
    if 'input_context_level' in kwargs and kwargs['input_context_level'] != 2:
        level = kwargs['input_context_level']
    else:
        level = 2

    if kwargs['input_feature_type'] == "exons":
        _f = os.path.join(kwargs['gtf_cache'],
                          "Exons_level_{}.tsv.gz".format(level))

    elif kwargs['input_feature_type'] == "introns":
        _f = os.path.join(kwargs['gtf_cache'],
                          "Introns_level_{}.tsv.gz".format(level))

    # Process cache
    features = pd.read_csv(_f, sep="\t")
 
    int_cols = [x for x in features.columns if
                any(j in x for j in ['upstream', 'downstream'])
                and all(j not in x for j in ['GC', 'Feature'])]

    features[int_cols] = features[int_cols].astype('Int64', errors='ignore')

    # Subset cache by gene_names|gene_ids, if they exists in the input
    subset_by = ['gene_names', 'gene_ids', 'transcript_ids']
    for i, filter_by in enumerate(subset_by):

        if filter_by in kwargs and kwargs[filter_by] is not None:
            if i == 0:
                col = "gene_name"
            elif i == 1:
                col = "gene_id"
            else:
                col = "trascript_id"

            _values = read_features_file(kwargs[filter_by])
            features = features[features[col].isin(_values)]
    
    # Merge data with cache
    cols = ['Chromosome', 'Start', 'End']
    for c in ['Strand', 'gene_name', 'gene_id', 'transcript_id']:
        if c in data.columns:
            cols.append(c)

    df = pd.merge(data, features,
                  how='left',
                  on=cols,
                  suffixes=('_repeat', ''),
                  indicator=True)
    df = df[[c for c in df.columns if not c.endswith('_repeat')]]

    # Split by known/unknown
    known = df[df._merge == "both"].drop(columns='_merge')

    absent_in_gtf = df[df._merge == "left_only"].drop(columns='_merge')[
        data.columns]

    # If there are known exons
    if known.shape[0] > 0:
        # Select the top ranked one
        surrounding = known.groupby(cols, group_keys=False).apply(
            lambda x: x.nlargest(1, "rank_score")).reset_index(drop=True)
        if 'just_intervals' in kwargs and kwargs['just_intervals']:
            surrounding.drop(list(surrounding.filter(
                regex='Length_|GC_')), axis=1, inplace=True)

    else:
        absent_in_gtf = data
        surrounding = []

    discarded_exons = ""
    # If all known, or there is not gene_name col (hence, impossible to process pseudoexons), we can return
    if absent_in_gtf.shape[0] == 0 or all(x not in absent_in_gtf.columns for x in ['Strand', 'gene_name']):
        return surrounding, known, absent_in_gtf, discarded_exons

    else:
    
        # Strand is needed to insert new exons
        if "Strand" not in absent_in_gtf:
            
            gene_name_col = [x for x in absent_in_gtf.columns if "gene_name" in x][0]
           
            absent_in_gtf = pd.merge(absent_in_gtf,
                                     features[['Strand', 'gene_name']],
                                     how='left',
                                     left_on=gene_name_col,
                                     right_on='gene_name').drop_duplicates()
            if gene_name_col != "gene_name":
                absent_in_gtf.drop(columns="gene_name", inplace=True)

        logger.info(
            "Loading transcript subfeatures from cache (to process new pseudoexons) ..")
        _f = os.path.join(kwargs['gtf_cache'],
                          "Transcripts_subfeatures_exploded.tsv.gz")
        subfeatures = pd.read_csv(_f, sep="\t")

        # Insert new exons withn a new transcript structure
    
        if all(x not in absent_in_gtf.columns for x in ['gene_name', 'gene_id']):
            logger.info("No gene_name or gene_id column found in the list of events regarded as pseudoexons. "
                        "Without this information, it is not possible to insert pseudoexons into a transcript structure. ")

        else:
            look_for = 'gene_name' if 'gene_name' in absent_in_gtf.columns else 'gene_id'
            new_exons, new_subfeatures, discarded_exons = insert_exons(
                absent_in_gtf, subfeatures, look_for=look_for)

            # Compute feature values for new transcripts generated from pseudoexons
            if 'just_intervals' not in kwargs or kwargs['just_intervals'] is False:
                compute_gc_and_hexamers(new_subfeatures, kwargs['fasta'])
                compute_length(new_subfeatures)
                new_subfeatures.drop(
                    columns=['cg_dinuc', 'hexamers'], inplace=True)

            # Extract surrounding features
            surrounding_pseudoexons = extract_surrounding_features(new_exons,
                                                                new_subfeatures,
                                                                level=2)

            if type(surrounding) == pd.DataFrame:
                surrounding = pd.concat(
                    [surrounding, surrounding_pseudoexons]).reset_index(drop=True)

            else:
                surrounding = surrounding_pseudoexons

    return surrounding, known, absent_in_gtf, discarded_exons


def computeFeaturesNow(data: pyranges.PyRanges, **kwargs):

    tx_obj = Transcripts(kwargs['gtf_cache'],
                         kwargs['gtf_is_processed'],
                         kwargs['out_dir'],
                         fasta=kwargs['fasta'],
                         gene_names=kwargs['gene_names'],
                         gene_ids=kwargs['gene_ids'],
                         select_top=kwargs['select_top'])

    tx_obj.compute_genomic_attributes(per_subfeature=True,
                                      just_get_intervals=kwargs['just_intervals'])

    subfeatures = tx_obj.explode_transcripts_subfeatures(
        extra_features=not kwargs['just_intervals'])

    if kwargs['feature_type'] == "exons":

        exons_obj = Exons(tx_obj.transcripts_and_subfeatures,
                          kwargs['gtf_is_processed'],
                          kwargs['out_dir'],
                          kwargs['fasta'],
                          exon_coordinates=data)

    # Known exons
    if exons_obj.exons.shape[0] > 0:
        surrounding = extract_surrounding_features(exons_obj.exons,
                                                   subfeatures,
                                                   level=2)
    else:
        surrounding = []

    # New exons
    discarded_exons = ""

    if exons_obj.absent_in_gtf.shape[0] > 0:

        if "Strand" not in exons_obj.absent_in_gtf:
            gene_name_col = [
                x for x in exons_obj.absent_in_gtf.columns if "gene_name" in x][0]
            exons_obj.absent_in_gtf = pd.merge(exons_obj.absent_in_gtf,
                                               exons_obj.exons[[
                                                   'Strand', 'gene_name']],
                                               how='left',
                                               left_on=gene_name_col,
                                               right_on='gene_name').drop_duplicates()
            if gene_name_col != "gene_name":
                exons_obj.absent_in_gtf.drop(columns="gene_name", inplace=True)

        # Insert new exons withn a new transcript structure
        new_exons, new_subfeatures, discarded_exons = insert_exons(
            exons_obj.absent_in_gtf, subfeatures)

        # Compute feature values for new transcripts generated from pseudoexons
        if kwargs['just_intervals'] is False:
            compute_gc_and_hexamers(new_subfeatures, exons_obj.fasta)
            compute_length(new_subfeatures)
            new_subfeatures.drop(
                columns=['cg_dinuc', 'hexamers'], inplace=True)

        # Extract surrounding features
        surrounding_pseudoexons = extract_surrounding_features(new_exons,
                                                               new_subfeatures,
                                                               level=2)

        if type(surrounding) == pd.DataFrame:
            surrounding = pd.concat(
                [surrounding, surrounding_pseudoexons]).reset_index()
        else:
            surrounding = surrounding_pseudoexons

    return surrounding, exons_obj.exons, exons_obj.absent_in_gtf, discarded_exons


def write_output(data, surrounding, known_exons, absent_in_gtf, discarded_exons, **kwargs):

    if 'input_drop_overlaps' in kwargs and kwargs['input_drop_overlaps'] is True:

        _surrounding = remove_overlaps(
            surrounding, keep='mutsplice', slack=0).reset_index()
        discard_cols = ['Chromosome', 'Start', 'End', 'Name', 'Strand']

        removed = _surrounding.merge(
            surrounding, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'right_only'][discard_cols]
        discarded_exons = pd.concat([discarded_exons, removed]) if not isinstance(
            discarded_exons, str) else removed

        surrounding = _surrounding

    to_write = {"_new_pseudoexons.bed": absent_in_gtf,
                "_discarded_exons.bed": discarded_exons}

    outpath = os.path.join(kwargs['out_dir'], kwargs['outbasename'])

    for name, _df in to_write.items():
        if isinstance(_df, pd.DataFrame) and _df.shape[0] > 0:
            bed6 = True if 'Strand' in _df.columns else False
            write_bed_file(_df,
                           name=outpath + name,
                           bed6=bed6,
                           additional_fields=[x for x in list(_df) if 'Col_' in x])

    if 'use_full_sequence' in kwargs:
        full_seqs = kwargs['use_full_sequence']
        fixed_len_seqs = not kwargs['use_full_sequence']
    else:
        full_seqs = True
        fixed_len_seqs = True

    n_seqs_written, n_1st_or_last_exon = generate_spliceAI_input_from_neighbour_df(surrounding,
                                                                                   fasta=open_fasta(
                                                                                       kwargs['fasta']),
                                                                                   outbasename=outpath,
                                                                                   extend_borders=kwargs['input_extend_borders'],
                                                                                   full_seqs=full_seqs,
                                                                                   fixed_len_seqs=fixed_len_seqs)

    stats = {'#Events': data.as_df().shape[0],
             '#Events_annotated (known_exons)': data.as_df().shape[0] - absent_in_gtf.shape[0],
             '#Events_unknown (new_exons)': absent_in_gtf.shape[0],
             '#Unique_events (unique_exons)': data.as_df().Name.nunique(),
             '#Unique_annotated_exons': known_exons.Name.nunique(),
             '#Unique_unknown_exons': absent_in_gtf.Name.nunique(),
             '#Written_seqs': n_seqs_written,
             '#1st_or_last_exon': n_1st_or_last_exon,
             '#Discarded_new_exons': discarded_exons.shape[0] if isinstance(discarded_exons, pd.DataFrame) else 0}

    pd.DataFrame.from_dict(stats, orient='index').to_csv(outpath + "_stats.tsv",
                                                         sep="\t",
                                                         header=False)


def main():
    parser = argparse.ArgumentParser(
        description="Utilities to extract info surrounding a given target sequence (e.g. exon).")
    parser.add_argument(
        dest='features', help='Path to the file with the target features to analyse')
    parser.add_argument(
        dest='gtf', help='Path to the GTF file or directory where the cache is located.')
    parser.add_argument(
        dest='out_dir', help='Path where output files will be written')
    parser.add_argument('--outbasename', required=True, help='Output basename')
    parser.add_argument('--fasta', required=True, type=str,
                        help="Reference genome in fasta")

    parser.add_argument('--feature_type', default="exons",
                        help="Feature type to be analysed")
    parser.add_argument('--features_have_header', action='store_true',
                        help="Whether --features files has a header"),
    parser.add_argument('--just_intervals', action='store_true',
                        help='Compute just intervals-based features (e.g do not compute GC content)')

    # Args to speed up gtf processing
    parser.add_argument('--gtf_is_processed', action='store_true',
                        help='If \'gtf\' is a file, this flag tells that it represents '
                             'the processed dataframe from an original gtf file.')

    parser.add_argument('--select_top', action='store_true', help="Select top transcript per gene "
                                                                  "so that only exons from top transcripts "
                                                                  "are kept")
    parser.add_argument(
        '--gene_names', help="Filter gtf by provided gene names")
    parser.add_argument('--gene_ids', help="Filter gtf by provided gene IDs")
    args = parser.parse_args()

    h = 0 if args.features_have_header else None
    logger.info("Parsing input features.")
    data = file_to_bed_df(args.features, header=h, is_0_based=False)
    logger.info("Done")

    kwargs = {'cache': args.gtf,
              'out_dir': args.out_dir,
              'outbasename': args.outbasename,
              'fasta': args.fasta,
              'input_feature_type': args.feature_type,
              'just_intervals': args.just_intervals,
              'gtf_is_processed': args.gtf_is_processed,
              'select_top': args.select_top,
              'gene_names': args.gene_names,
              'gene_ids': args.gene_ids}

    if os.path.isdir(args.gtf):
        surrounding, known_exons, absent_in_gtf, discarded_exons = getFeaturesFromCache(
            data, **kwargs)

    elif os.path.isfile(args.gtf):
        surrounding, known_exons, absent_in_gtf, discarded_exons = computeFeaturesNow(
            data, **kwargs)

    else:
        raise ValueError(
            'Please set a valid file or directory for the input data')

    # WRITE OUTPUT #
    write_output(data, surrounding, known_exons,
                 absent_in_gtf, discarded_exons, **kwargs)


if __name__ == "__main__":
    main()
