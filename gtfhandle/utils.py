import pandas as pd
from pandas.core.frame import DataFrame
import pyranges as pr
import gzip
from typing import Union, Optional, TextIO, List
from pyfaidx import Fasta
import os
from loguru import logger
#import pyBigWig
import numpy as np
from collections import Counter
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
#from Bio.Alphabet import IUPAC
from Bio import SeqIO
from itertools import islice


def read_features_file(file: Union[TextIO, str, List, pd.Series, np.ndarray]):
    """
    Reads a file with feature IDs (exon, transcripts, etc)
    :param file: File with one feature ID per line
    :return list: List with IDs
    """
    if isinstance(file, (List, pd.Series, np.ndarray)):
        return file
    
    try:
        with open(file, 'r') as f:
            feature_ids = [line.rstrip() for line in f]
        f.close()
        return feature_ids
    except FileNotFoundError:
        raise ValueError("{} is not a valid file".format(file))


def bed_is_ok(file: Union[str, TextIO], kill: bool = True):
    """
    Reads a bed file and checks if it's properly
    formatted by creating a `BedTool` object
    :param file: Bed file
    :param bool kill: Whether to raise Exception and exit.
    If false, a boolean is returned.
    """
    try:
        if file.endswith(('.gz', '.bgz')):
            l = gzip.open(file).readline()
        else:
            l = open(file).readline()

        start = l.split()[1]
        end = l.split()[2]
        int(start)
        int(end)

    except IndexError:
        if kill:
            raise IndexError("'{}' does not seem to be a valid bed file.".format(file))
        else:
            return False

    except ValueError:
        if kill:
            raise ValueError("'{}' file does not seem to have a proper bed format. Does it have"
                             " an header? If so, remove it".format(file))
        else:
            return False

    except UnicodeDecodeError:
        if kill:
            raise UnicodeDecodeError("'{}' file may be in bigwig format? If not, make sure the"
                                     " compressed bed filename ends with .gz or .bgz".format(file))
        else:
            return False

    return True


def write_ids(input_ids: List, output_file: str):
    """
    Writes a list of IDs to file, one per line.
    :param List input_ids: List of IDs
    :param str output_file: Path to the output file
    """
    with open(output_file, 'w') as out:
        [out.write("{}\n".format(elem)) for elem in input_ids]
    out.close()


def write_bed_file(input_data: pd.DataFrame,
                   name: str,
                   bed6: bool = False,
                   compression: str = None,
                   is_1_based: bool = False,
                   use_as_name_or_score: Optional[dict] = None,
                   additional_fields: Optional[List] = None):
    """
    Write bed from pandas Dataframe
    :param pd.DataFrame input_data: Data to write (Must contain Chromosome,
     Start and End) columns
    :param str name: Filename to write output
    :param bool bed6: Whether 6 column bed file should be written. Default: False
    :param str compression: How to compress file. Default: gzip
    :param bool is_1_based: Whether input `data` owns 1-based coordinates.
        Default: `False`, `input_data` is 0-based
    :param use_as_name_or_score: Mapping of columns from `data` to be used as the
        name and/or score column when `bed6` is set to `True`.
        E.g. {'gene_name':'Name', 'exon_number:'Score'}
    :param additional_fields: Add additional columns to the output.

    """
    data = input_data.copy()
    assert all(x in data.columns for x in ['Chromosome', 'Start', 'End']), \
        'Dataframe must contain required columns'

    if any(v is not None for v in [use_as_name_or_score, use_as_name_or_score]):
        assert bed6 is True, "'bed6' should be enabled to use 'use_as_name_or_score' or 'additional_fields' args"

    if additional_fields:
        assert isinstance(additional_fields, list), "additional_fields argument requires a list"
        assert all(x in data.columns for x in additional_fields), \
            'Dataframe must contain all additional columns passed'

    if is_1_based:
        data['Start'] -= 1

    if bed6:
        cols = ['Chromosome', 'Start', 'End', 'Name', 'Score', 'Strand']
        assert 'Strand' in data.columns, 'When "bed6" is set to True, "Strand" must exist'
        _df = data.copy()

        if isinstance(use_as_name_or_score, dict):
            assert all(x in ['Name', 'Score'] for x in use_as_name_or_score.values()), \
                "'use_as_name_or_score' only accepts 'Name' and 'Score' as values."
            assert all(x in data.columns for x in use_as_name_or_score.keys()), \
                "Columns set to be used as 'Name' and/or 'Score' do not exist in the data."

            # if Score and Name column already existed
            if 'Score' in use_as_name_or_score.values() and 'Score' in data.columns:
                _df.drop("Score", axis=1, inplace=True)
            if 'Name' in use_as_name_or_score.values() and 'Name' in data.columns:
                _df.drop("Name", axis=1, inplace=True)
            _df = _df.rename(columns=use_as_name_or_score)

        if additional_fields is not None:
            cols = cols + additional_fields

        try:
            _df = _df[cols]

        except KeyError:
            if 'Name' not in _df.columns:
                _df['Name'] = "."
            if 'Score' not in _df.columns:
                _df['Score'] = "."

            _df = _df[cols]

    else:
        _df = data[['Chromosome', 'Start', 'End']]

    _df.to_csv(name,
               compression=compression,
               sep='\t',
               index=False,
               header=False)


def write_bedgraph_file(data: pd.DataFrame,
                        name: str,
                        value_column: str = "Score"):
    """
    Write bedgraph format from pandas Dataframe
    :param data: Data to write (Must contain Chromosome, Start and End) columns
    :param name: Filename to write output
    :param str value_column: Column to use as the score column. Default: Score
    """
    assert all(x in data.columns for x in ['Chromosome', 'Start', 'End', value_column]), \
        'Dataframe must contain required columns'

    assert name.endswith(".bedgraph"), 'Name of the output file must have the "bedgraph" extension'
    _df = data[['Chromosome', 'Start', 'End', value_column]]
    _df.to_csv(name,
               sep='\t',
               index=False,
               header=False)


def open_fasta(fasta: Union[str, TextIO, Fasta]):
    """
    Creates faidx index for fast random access.

    :param str fasta: Original Fasta file
    """
    if isinstance(fasta, Fasta):
        return fasta

    if fasta is not None:
        try:
            open(fasta).readline()
        except UnicodeDecodeError:
            raise ValueError("Make sure you provide an uncompressed fasta for pyfaidx")

        return Fasta(fasta)
    return None


# def _get_bigwig_signal(x: pd.Series, bw: pyBigWig.pyBigWig, bins: int, stat: 'str' = 'mean'):
#     """
#     :param pd.Series x: Df row with genomic coordinates
#     :param pyBigWig.pyBigWig bw: Opened pybigwig file
#     :param int bins: Number of bins to split each interval
#     :param str stat: Type of metric to compute. Default: 'mean'

#     :return np.array: 1D np.array with the bigwig signal: the
#         values retrieved for the given interval considering
#         the `bins` and the `stat` provided
#     """
#     out = bw.stats(x.Chromosome,
#                    x.Start,
#                    x.End,
#                    nBins=bins,
#                    type=stat,
#                    exact=True)

#     out = out[::-1] if x.Strand == "-" else out
#     return np.array(out).astype('float16')


def dict_to_fasta(d: dict, outfile: str):
    """
    Writes a dict (keys are the header, values are the seqs)
    to a fasta file

    :param dict d: Input dictionary with sequences
    :param str outfile: Output file
    """
    f = open(outfile, "w")
    for k, v in d.items():
        f.write(">{}\n{}\n".format(k, v))


def fasta_to_dict(fasta: str):
    """
    Reads a fasta file into a dictionary
    :param str fasta: Input fasta
    :return dict: Processed dict where headers are the keys,
    and the values correspond to the sequences
    """
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta, "fasta")}


def file_to_bed_df(data: Union[str, pd.DataFrame, pd.Series],
                   col_index: int = 0,
                   is_0_based: bool = True,
                   header: int = None,
                   c_map: dict = None):
    """
    Reads a tab-delimited file where
    one column contains genomic
    intervals (chr:start-end) and creates
    a bed df

    :param Union[str, pd.DataFrame] file: Input file/df
    :param int col_index: Col index where
    coordinates are located. Default: `0`
    :param bool is_0_based: Whether input file
    has coordinates in 0-based format.
    Default: `True`, coordinates are 0-based.
    Coordinates will be returned in 0-based half
    open intervals
    :param int header: Row index where header
     is contained. Default: `None`, no header
    :param dict c_map: Dictionary mapping col
    indexes to col names when `header` is `None`.

    :return pd.DataFrame:
    """

    if isinstance(data, str):
       df = pd.read_csv(data, sep="\t", header=header)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, pd.Series):
        df = data.to_frame().T
    else:
        raise ValueError('Wrong input')
    
    df_intervals = df.iloc[:, col_index].str.split(':|-', expand=True)

    df_intervals.columns = ['Chromosome', 'Start', 'End']
    df_intervals[['Start', 'End']] = df_intervals[['Start', 'End']].apply(pd.to_numeric)

    if header is None:
        cols = []
        if not c_map:
            c_map = {col_index: 'Name', 1: 'Score', 2: 'Strand'}

        for i in range(len(list(df))):
            c = c_map.get(i, 'Col')
            c = c + "_{}".format(i) if c == 'Col' else c
            cols.append(c)
        df.columns = cols
        df_intervals.columns = ['Chromosome', 'Start', 'End']

  
    c_map = {'strand': 'Strand', 
             df.columns[col_index]: 'Name'}
    df = df.rename(columns=c_map)

    if "Strand" in df.columns:
        assert all(x in ['+', '-'] for x in df.Strand), "Strand column must contain only '+' or '-' values. "\
                                                    "Please set c_map argument to manually assign column names "\
                                                    "to specific column indexes"                                                    
    if not is_0_based:
        df_intervals['Start'] -= 1

    return pr.PyRanges(pd.concat([df_intervals, df], axis=1))


def fasta_to_bed_df(fasta: Union[dict, str]):
    """
    Reads a fasta file where the header
    can be parsed into a bed interval
    and generates bed dataframe out of
    it
    :param Union[dict, str] fasta: Fasta sequences
    :return pd.DataFrame: Dataframe with proper colnames
    """
    if isinstance(fasta, str):
        seqs = fasta_to_dict(fasta)

    intervals = []
    for k, v in seqs.items():
        try:
            chrom = k.split(":")[0][0:]
            start = k.split(":")[1].split("-")[0]
            end = k.split("-")[1].split("(")[0]
            strand = k.split("(")[1][0]
            intervals.append([chrom, start, end, strand, k, v])

        except IndexError:
            logger.info("No bed intervals extracted from header of sequence {}. Will run sequence blindly.".format(k))
            #raise IndexError("Make sure that bed intervals can be extracted from fasta headers.")
            intervals.append([None, None, None, None, k, v])
  
    return pd.DataFrame.from_records(intervals, columns=['Chromosome', 'Start', 'End', 'Strand', 'id', 'seqs'])


def get_fasta_sequences(x: pd.Series,
                        fasta: Fasta,
                        slack: Optional[int] = None,
                        one_based: bool = False,
                        extend: str = None,
                        chrom_sizes: dict = None,
                        start_col: str = "Start",
                        end_col: str = "End"):
    """
    Retrieve Fasta sequence from a set of genomic coordinates of a
    given feature using an indexed reference genome

    :param pd.Series x: Single row of a Feature dataframe
        (e.g. Transcript, exon)
    :param Fasta fasta: pyFaidx object representing the
        reference genome
    :param int slack: How many additional bp to extract
        fasta sequence from each interval side.
        Default: `None`. Retrieve fasta from exact intervals
        given
    :param bool one_based: Whether start coordinates of
        `x` are 1-based. Default: `False`

    :param str extend: If slack is provided, and genomic
    sequence of each interval is already present in the
    column provided in this argument, it extends the sequence
    without changing the existing sequence (e.g. useful if
    artificial mutations were inserted within the intervals)

    :param dict chrom_sizes: Chromosome sizes of the genome
        to ensure that intervals are in range. Default: `None`

    :param str start_col: Colname in `x` where end coordinate is
    located. Default: `Start`
    :param str end_col: Colname in `x` where end coordinate is
    located. Default: `End`

    :return pd.Series: Additional column with the fasta sequence
        of a given feature
    """

    if chrom_sizes and x.Chromosome not in chrom_sizes.keys():
        logger.info('Chrom of interval not found in chromosome '
                     'sizes dict. Not possible to extract fasta seq.')
        return

    if extend is not None:
        assert slack is not None, "Set the slack argument when " \
                                  "'extend' argument is provided."

        #assert extend in x.index, "{} is not in the columns names.".format(extend)

    # missing assert that checks whether
    # intervals comply with chrom sizes

    try:
        start = x[start_col] - 1 if one_based else x[start_col]
        end = x[end_col]

        if slack is not None:
            start -= slack
            end += slack

        int_id = x.Chromosome + "_" + str(start) + "_" + str(end)
        if chrom_sizes and end > chrom_sizes[x.Chromosome]:
            logger.info('Interval {} exceeds chromosome '
                         'boundaries ({})'.format(int_id, chrom_sizes[x.Chromosome]))
            return

        if extend is None or extend not in x.index:
            # start attributes in pyFaidx are 0-based
            if x['Strand'] == "-":
                out_seq = fasta[x['Chromosome']][start:end].reverse.complement.seq
            else:
                out_seq = fasta[x['Chromosome']][start:end].seq

        else:
            if x['Strand'] == "-":
                left = fasta[x['Chromosome']][start:start + slack].reverse.complement.seq
                right = fasta[x['Chromosome']][end - slack:end].reverse.complement.seq
                out_seq = right + x[extend] + left
            else:
                left = fasta[x['Chromosome']][start:start + slack].seq
                right = fasta[x['Chromosome']][end - slack:end].seq
                out_seq = left + x[extend] + right

        if len(out_seq) == end - start:
            # If seq is all composed by Ns
            if not out_seq.replace('N', ''):
                _s = chrom_sizes[x.Chromosome] if chrom_sizes else None
                logger.info('Sequence for the interval {} '
                             'is just composed by Ns. Chromosome '
                             'len in Fasta obj: {}. Chromosome '
                             'len in the chrom dict, if provided: {}'.format(int_id,
                                                                             len(fasta[x.Chromosome]),
                                                                             _s))
                return
            return out_seq

        else:
            logger.info("Problem extracting Fasta sequence for "
                         "the given interval: {}. Length of "
                         "chromosome in Fasta where interval is "
                         "located: {}".format(int_id,
                                              len(fasta[x.Chromosome])))
            return

    except KeyError:
        raise KeyError("{} chromosome is not in Fasta header. Additionally, "
                       "make sure that the apply function is performed rowwise "
                       "(axis=1), when extracting fasta sequences.".format(x.Chromosome))


def get_chromosome_sizes(genome_build: str,
                         primary: list = None):
    """
    Returns a dict with the chromosome sizes
    for the given genome build
    :param str genome_build: Genome build
    :param list primary: List with the available
    primary chromosomes for a given Fasta object.
    Useful if Fasta comes with 'chr' string, then
    the dict will be adjusted for the primary
    chromosomes

    :return dict: Chromosome sizes
    """
    assert genome_build in ['hg19', 'hg38']

    raise NotImplementedError
    # # chromosomes have the 'chr' notation
    # _sizes = pr_db.ucsc.chromosome_sizes(genome_build)
    # d = {k: v for k, v in zip(_sizes.Chromosome,
    #                           _sizes.End)}
    # if primary:
    #     # primary chrom in Fasta have 'chr'
    #     if all([p in d.keys() for p in primary]):
    #         return d
    #     # remove 'chr' from dict
    #     else:
    #         return {k.replace('chr', ''): v for k, v in d.items()}

    # return d


# def df_to_seqrecord(sequences: Union[pd.DataFrame, pr.PyRanges, pd.Series],
#                     seq_col: str = 'Fasta',
#                     header_col: str = 'ID',
#                     iupac: IUPAC = IUPAC.unambiguous_dna) -> List[SeqRecord]:
#     """
#     Creates a list of Bio.SeqRecord from a set
#     of Fasta sequences stored in a pd.Series

#     :param pd.DataFrame sequences: Df with Fasta sequences
#         to transform
#     :param str seq_col: Column where Fasta is located.
#         Default: 'Fasta'
#     :param str header_col: Column to serve as fasta header
#         Default: 'ID'
#     :param IUPAC iupac: Alphabet to use in the creation of
#         SeqRecords. Default: 'IUPAC.unambiguous_dna'

#     :return List: List of SeqRecord without Ns
#     """
#     if not isinstance(sequences, pd.Series):
#         assert seq_col in sequences.columns, "Input data must contain a {} column".format(seq_col)
#         assert header_col in sequences.columns, "Input data must contain a {} column to serve as " \
#                                                 "fasta header".format(header_col)

#         if isinstance(sequences, pr.PyRanges):
#             sequences = sequences.as_df()

#         return list(sequences.apply(lambda x: SeqRecord(Seq(x[seq_col].replace('N', ''), iupac),
#                                                         id=x[header_col]) if 'N' in x[seq_col]
#         else SeqRecord(Seq(x[seq_col], iupac), id=x[header_col]), axis=1))

#     else:
#         return list(sequences.apply(lambda x: SeqRecord(Seq(x))))


def compute_sequence_based_features(sequence: str, all_hexamers: list):
    """
    Computes GC content, CG dinucleotide
    and hexamer counts for the given input
    sequence
    :param str sequence: Fasta sequence
    :param list all_hexamers: List
    with all possible sequence hexamers (4096)
    :return tuple: Tuple with GC, CG dinuc
    and a dict with k-mer counts
    """
    # GC content
    try:
        gc = round(sum([1 for nucl in sequence if nucl in ['G', 'C']]) / len(sequence) * 100, 2)
    except TypeError:
        gc = None
    # CG dinucleotides
    # cg_dinucl = sequence.count('GC')

    # Hexamer counts
    # kmers = []
    # n_kmers = len(sequence) - 6 + 1
    # for i in range(n_kmers):
    #     kmer = sequence[i:i + 6]
    #     kmers.append(kmer)
    #
    # kmer_counts = Counter(kmers)
    # hexamers = {}
    # for x in all_hexamers:
    #     if x not in kmer_counts:
    #         hexamers[x] = 0
    #     else:
    #         hexamers[x] = kmer_counts[x]
    # hexamers = tuple([(k, v) for k, v in hexamers.items()])
    return pd.Series([gc, [], []])

def generate_chunks(data: Union[dict, list], size=200):
    """
    Split input sequences/dfs in chunks of 100 seqs
    """
    if isinstance(data, dict):
        it = iter(data)
        for i in range(0, len(data), size):
            yield {k: data[k] for k in islice(it, size)}
            
    elif isinstance(data, list):
        for i in range(0, len(data), size):
            yield data[i:i + size]
                            
def _chunk_iterator(iterator, batch_size):
    """
    Adapted from https://biopython.org/wiki/Split_large_file
    
    Returns lists of length batch_size.

    Can be used on any iterator

    This is a generator function, and it returns lists of the
    entries from the supplied iterator. Each list will have
    batch_size entries, although the final list may be shorter.
    """
    entry = True  # Make sure we loop once
    while entry:
        batch = []
        while len(batch) < batch_size:
            try:
                entry = next(iterator)
            except StopIteration:
                entry = None
            if entry is None:
                # End of file
                break
            batch.append(entry)
        if batch:
            yield batch
            
def split_fasta_file(fasta: str, out_dir: str, batch_size: int):
    """
    Splits fasta file in several files 
    with 'size' sequences each of them
    """    
    chunks = []
    record_iter = SeqIO.parse(open(fasta), "fasta")
    for i, batch in enumerate(_chunk_iterator(record_iter, batch_size)):
        filename = "{}/chunk_{}.fa".format(out_dir, i)
        with open(filename, "w") as handle:
            SeqIO.write(batch, handle, "fasta")
        chunks.append(filename)
    return chunks
    
def write_fasta_sequences(df: Union[pr.PyRanges, pd.DataFrame],
                          outname: str,
                          seq_col: str = 'Fasta',
                          header_col: Optional[str] = None):
    """
    Write fasta file from a dataframe of features
    :param pd.Dataframe df: Df with a set of sequences
        to write to a file
    :param str outname: Filename to write output
    :param str seq_col: Column where Fasta is located.
        Default: 'Fasta'
    :param str header_col: Column to serve as fasta header.
        Default: Df index.
    """

    assert seq_col in df.columns, "Dataframe with genomic features must " \
                                  "contain a {} column".format(seq_col)
    assert not os.path.isdir(outname), "A directory was provided as the output file"

    if isinstance(df, pr.PyRanges):
        df = df.as_df()

    out = open(outname, "w")
    if header_col:
        assert header_col in df.columns, "{} is not in the list of columns".format(header_col)
        _df = df.copy()

    else:
        header_col = df.index.name
        _df = df.reset_index()

    _df[header_col] = _df[header_col].apply(lambda x: "{}{}".format('>', x))
    for record in list(zip(_df[header_col], _df[seq_col])):
        out.write('\n'.join(str(s) for s in record) + '\n')


def remove_overlaps(df: pd.DataFrame,
                    keep: Union[str, None] = "longest",
                    ignore_strand: bool = False,
                    slack: int = -1,
                    invert: bool = False):
    """
    Compute interval overlaps within a dataframe
    with features (e.g. exons, transcripts, etc)

    :param pd.DataFrame df: Df to compute overlaps
    :param str keep: What to do with the features
        that overlap. Default: 'longest', keep only
        the longest feature. Possible values: 'longest',
        'first', 'None', 'mutsplice'. If mutsplice is set,
        returns intervals filtered  by first: a) be a 
        known transcript and b ) by rank_score of that
        transcript
        
    :param bool ignore_strand: Whether to ignore
        strand information when looking for overlaps.
        Default: False: Overlaps in different strands
        are counted
    :param int slack: Consider intervals separated
        by less than slack to be in the same cluster.
        If slack is negative, intervals overlapping
        less than slack are not considered to be in
        the same cluster.
    :param bool invert: Invert output to return just
    the intervals that overlap. Default: False

    :return pd.Dataframe: Df with features that overlap
    removed according to the `keep` argument
    """
    assert keep in ['longest', 'first', None, 'mutsplice'], 'keep parameter must longest, first or None'
    assert all(x in df.columns for x in ['Chromosome', 'Start', 'End']), 'Df does not contain required cols'

    try:
        _tmp = pr.PyRanges(df.rename_axis('Name').reset_index(drop=True)).cluster(strand=ignore_strand,
                                                                      slack=slack,
                                                                      nb_cpu=5,
                                                                      count=True)
    except ValueError:
        _tmp = pr.PyRanges(df).cluster(strand=ignore_strand,
                                    slack=slack,
                                    nb_cpu=5,
                                    count=True)

    if invert:
        return _tmp.as_df().groupby('Cluster', sort=False).filter(lambda x: len(x) > 1)

    _overlaps_removed = _tmp.as_df().groupby('Cluster', sort=False).filter(lambda x: len(x) == 1)
    if keep is None:
        return _overlaps_removed.set_index('Name').drop(['Cluster', 'Count'], axis=1)
    else:
        _with_overlaps = _tmp.as_df().groupby('Cluster', sort=False).filter(lambda x: len(x) > 1)
        logger.info("Number of features that have at least one overlap: {}".format(_with_overlaps.shape[0]))
        if keep == 'longest':
            idx_longest = _with_overlaps.groupby('Cluster', sort=False).apply(lambda x: (x.End - x.Start).idxmax())
            _filtered = _with_overlaps.loc[idx_longest]

        elif keep == 'first':
            _filtered = _with_overlaps.groupby('Cluster', sort=False).first().reset_index(drop=True)

        elif keep == 'mutsplice':
            _filtered = _with_overlaps.groupby('Cluster', sort=False).apply(lambda x: x[~x.transcript_id.str.contains('_')]).reset_index(drop=True)

            if _filtered.empty:
                _filtered = _with_overlaps

            _filtered = _filtered.groupby('Cluster', sort=False).apply(lambda x: x.nlargest(1, 'rank_score'))
            
        return pd.concat([_overlaps_removed, _filtered]). \
            sort_values(['Chromosome', 'Start', 'End']). \
            set_index('Name').drop(['Cluster', 'Count'], axis=1)


def remove_version_from_ensembl_ids(df: pd.DataFrame):
    """
    Removes version number from ensembl IDs
    from genes, transcripts and exons.
    Be default, it looks for 'gene_id',
    'transcript_id' and 'exon_id' columns

    :param pd.DataFrame df: Input df
    :return pd.DataFrame: df with version
        removed
    """
    id_cols = ['gene_id', 'transcript_id', 'exon_id']
    cols = [col for col in df.columns if col in id_cols]
    for c in cols:
        df[c] = df[c].str.split('.').str[0]


def filter_redundancy_in_sexual_chromosomes(df: pd.DataFrame, chr_to_remove: List = ["chrY", "Y"]):
    """
    Filters duplicate genes found in sexual chromosomes.
    Those duplicates (same gene and transcript ID) have \
    influence on downstream functions, such as mapping
    regions to unique transcripts, or training sequence-based
    models based on mRNA sequences (duplicate instances).
    :param pd.DataFrame df: Df to filter
    :param str chr_to_remove: Remove features from the given
        sexual chromosome. Default: `chrY`
    :return:
    """
    logger.info("Filtering out duplicate genes in sexual chromosomes.")

    sexual_chroms = ["chrX", "chrY", "X", "Y"]
    assert all([chrom in sexual_chroms for chrom in chr_to_remove]), "Wrong sexual chromosome provided."

    gene_groups = df.groupby('gene_id')
    duplicates = gene_groups.filter(lambda x: (x.Feature.values == 'gene').sum() > 1)
    _to_filter = duplicates[duplicates.Chromosome.isin(chr_to_remove)].index
    logger.info("{} genes filtered out.".format(len(_to_filter)))
    df.drop(_to_filter, inplace=True)
