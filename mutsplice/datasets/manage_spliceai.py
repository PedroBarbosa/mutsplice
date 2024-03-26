import os
from loguru import logger
import itertools
from typing import List, Tuple
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import pyranges as pr
import time
from tqdm import tqdm
import tensorflow as tf
from matplotlib.patches import Patch
from dna_features_viewer import GraphicFeature, GraphicRecord
from spliceai.utils import *
from gtfhandle.utils import *
from mutsplice.datasets.utils import _process_ss_idx, _adjust_ss_idx_on_mutated_seqs


class SpliceAI(object):
    """
    Creates a SpliceAI object
    """

    def __init__(self,
                 infile: str,
                 metadata: str,
                 outbasename: str,
                 outdir: str = os.getcwd(),
                 extend_context: int = 0,
                 ref_genome: str = None,
                 is_cassette_exon: bool = True,
                 splice_site_idx: str = None,
                 no_batch_predictions: bool = False,
                 batch_size: int = 64,
                 save_spliceai_raw: bool = False,
                 raw_preds_path: str = None):
        """
        Initializes object with main parameters and runs
        spliceAI on the input sequences

        :param str infile: Input file in fasta or bed format
        :param str metadata: Metadata about each sequence/interval
        represented in the `infile`
        :param str outbasename: Basename to the output files
        :param str outdir: Output directory
        :param int extend_context: Extend sequences/intervals on
        each side by this value. Default: `0`
        :param str ref_genome: Reference genome. Needed if `infile`
        is in bed format
        :param bool is_cassette_exon: Whether sequences represent
        cassette exons.
        :param str splice_site_idx: Splice site indexes within the
        input sequences/intervals
        :param bool no_batch_predictions: Do not make inferences in batches of 
        size `batch_size`. If set to `True`, sequences will be predicted
        one at a time. Default: `False`, runs inferences in batches of
        sequences padded to have the same size.
        :param int batch_size: Size of each batch when `no_batch_predictions`
        is `False`
        :param bool save_spliceai_raw: Whether raw spliceAI predictions should be
        saved.
        :param str raw_preds_path: Directory where raw spliceAI preds in pickle 
        format are located. If set, spliceAI is not run. Useful for testing purposes
        """
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        self.outbasename = outbasename
        self.is_cassette_exon = is_cassette_exon
        self.no_batch_predictions = no_batch_predictions
        self.batch_size = batch_size

        if self.is_cassette_exon:
            assert splice_site_idx, "Splice site indexes must be provided when " \
                                    "sequences represent cassette exons with surrounding " \
                                    "introns."

        extension = os.path.splitext(infile)[1]

        assert any(x == extension for x in ['.fa', '.bed']), "Input file must be a fasta (*.fa) " \
                                                             "or bed (*.bed) file."

        logger.info("Processing {} file".format(infile))

        if extension == '.fa':
            df = fasta_to_bed_df(infile)
            try:
                self.df = pr.PyRanges(df).as_df()
            except TypeError:
                self.df = df

        elif extension == '.bed':
            assert ref_genome, "bed file provided, please set the reference genome argument."

            self.df = pd.read_csv(infile,
                             sep="\t",
                             names=['Chromosome', 'Start', 'End', 'score', 'name', 'Strand'])

        if extend_context > 0 or extension == '.bed':
            assert ref_genome, "To extend sequences or extract sequence from intervals, " \
                               "the reference genome argument is necessary."

            genome = open_fasta(ref_genome)

            self.df['seqs'] = self.df.apply(get_fasta_sequences,
                                  fasta=genome,
                                  slack=extend_context,
                                  one_based=False,
                                  extend="seqs",
                                  axis=1)

        if all(x is None for x in self.df.Chromosome):
            self.df['scanned_interval'] = self.df['id']
        else:
            self.df['scanned_interval'] = self.df['Chromosome'].astype(str) + ":" + \
            (self.df['Start'].astype(int) - extend_context).astype(str) + \
            "-" + (self.df['End'].astype(int) + extend_context).astype(str) + \
            "(" + self.df['Strand'].astype(str) + ")"

        if not 'id' in self.df.columns:
            self.df['id'] = self.df.scanned_interval
        self.seqs = dict(zip(self.df['id'], self.df['seqs']))

        if self.is_cassette_exon:
            self.ref_ss_idx, self.name = _process_ss_idx(self.seqs,
                                                         splice_site_idx)
        else:
            self.ref_ss_idx = None
            
        if metadata is None:
            self.metadata = metadata
        else:
            self.metadata = pd.read_csv(metadata, sep="\t")

        if raw_preds_path:
            if os.path.isdir(raw_preds_path):

                raw_path = os.path.join(raw_preds_path, self.outbasename)
                donor_path = raw_path + '_donors_raw.pickle'
                acceptors_path = raw_path + '_acceptors_raw.pickle'
                self.donor_preds = pd.read_pickle(donor_path)
                self.acceptor_preds = pd.read_pickle(acceptors_path)
            else:
                raise ValueError(
                    '{} is not a valid dir'.format(raw_preds_path))

        else:
            self.acceptor_preds, self.donor_preds = self.run()
            self.fix_coordinates()
            if save_spliceai_raw:
                raw_preds_path = os.path.join(self.outdir, self.outbasename)
                self.acceptor_preds.to_pickle(
                    raw_preds_path + '_acceptors_raw.pickle')
                self.donor_preds.to_pickle(
                    raw_preds_path + '_donors_raw.pickle')

    def run(self, context: int = 10000):
        """
        Predicts splice site probabilities
        for each position in a set of input
        sequences
        """
        logger.debug("Loading the models..")
        paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
        models = [
            load_model(resource_filename('spliceai', x), compile=False)
            for x in paths
        ]

        donor_preds = {}
        acceptor_preds = {}

        logger.debug(
            "Sorting sequences by their length and performing one hot encoding..")
        _df = pd.DataFrame.from_dict(
            self.seqs, orient='index').reset_index().rename(columns={
                'index': 'header',
                0: 'fasta'
            })

        _df['len'] = _df.fasta.str.len()
        _df = _df.set_index('len').sort_index()
        x = _df.fasta.apply(one_hot_encode).to_numpy()

        logger.debug(
            "Padding sequences so that every position in the input is predicted.."
        )

        npad = ((context // 2, context // 2), (0, 0))
        for i in range(len(x)):
            x[i] = np.pad(x[i], pad_width=npad)

        # Predict one sequence at a time
        # All seqs are resorted to have the same length
        logger.info("Total of {} predictions will be made".format(x.shape[0]))
        if self.no_batch_predictions is True:
            donor_preds, acceptor_preds = {}, {}

            max_len = x[-1].shape[0]
            logger.info(
                "Extra padding to max seq length ({} bp)..".format(max_len))

            x = tf.keras.preprocessing.sequence.pad_sequences(
                x, maxlen=max_len)

            logger.info("Running spliceAI..")

            start = time.time()
            for i in tqdm(range(x.shape[0])):
                _x = x[i][None, :]

                y_pred = np.mean([model.predict(_x)
                                 for model in models], axis=0)

                header = _df.iloc[i].header
                to_unpad = y_pred.shape[1] - _df.index[i]

                acceptor_preds[header] = y_pred[0, :, 1][to_unpad:]
                donor_preds[header] = y_pred[0, :, 2][to_unpad:]

            end = time.time()
            logger.log("MAIN", "Done. Time elapsed: {} min.".format(
                (end - start) / 60))

        # Predict in batches
        else:

            @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 4], dtype=tf.int32),))
            def predict_batch(batch):

                return tf.reduce_mean([model(batch) for model in models], axis=0)

            logger.info(
                'Splitting data in batches of {} seqs..'.format(self.batch_size))
            batches = np.split(x, np.arange(
                self.batch_size, len(x), self.batch_size))
            logger.info(
                'Number of batches: {}. Making predictions..'.format(len(batches)))

            start = time.time()
            preds = []

            for batch in tqdm(batches):

                seq_lengths = [seq.shape[0] for seq in batch]
                max_len = max(seq_lengths)

                batch = tf.keras.preprocessing.sequence.pad_sequences(
                    batch, maxlen=max_len)
                batch_tf = tf.convert_to_tensor(batch, dtype=tf.int32)

                raw_preds = predict_batch(batch_tf)

                batch_preds = [x.numpy()[max_len - seq_lengths[i]:]
                               for i, x in enumerate(raw_preds)]
                preds.extend(batch_preds)

            end = time.time()
            logger.info("Done. Time elapsed: {} min.".format(
                (end - start) / 60))

            for i, seq_id in enumerate(_df.header):

                _pred = preds[i]
                acceptor_preds[seq_id] = _pred[:, 1]
                donor_preds[seq_id] = _pred[:, 2]

        return pd.DataFrame.from_dict(acceptor_preds, orient='index').transpose(), \
            pd.DataFrame.from_dict(donor_preds, orient='index').transpose()

    def get_preds_on_reference_seq(self,
                                   splice_site_threshold: float = 0.1) -> List:
        """
        Returns per-position predictions
        on the reference sequence
        (non-mutated). Additionally,
        it can can write down seq indexes that
        were predicted to be splice sites

        :param float splice_site_threshold: Minimum prediction
        value so that a given position is considered to be a
        splice site. Default: `0.1`

        :return List: List of donor and acceptor predictions in
        the reference sequence
        """
        ref_col_name = [
            c for c in self.donor_preds.columns
            if 'REF_seq' in c or self.metadata is None
        ]
  
        donors = self.donor_preds[ref_col_name].to_numpy().squeeze()
        acceptors = self.acceptor_preds[ref_col_name].to_numpy().squeeze()

        pred_ss = open(
            os.path.join(self.outdir,
                         self.outbasename + "_SpliceAI_predicted_ss.txt"),
            "w")
        pred_ss.write("{}\t{}\t{}\t{}\t{}\n".format("header", "acceptor_idx",
                                                    "donor_idx", "acceptor_preds", "donor_preds"))

        _d = np.argwhere(donors > splice_site_threshold)
        _a = np.argwhere(acceptors > splice_site_threshold)

        for i, _name in enumerate(ref_col_name):

            # Only one ref seq, mutsplice is ON
            if _d.shape[1] == 1:
                idx_d = list(_d.flatten())
                idx_a = list(_a.flatten())

                d = [str(x) for x in idx_d]
                a = [str(x) for x in idx_a]

                d_p = [str(round(x, 3)) for x in donors[idx_d]]
                a_p = [str(round(x, 3)) for x in acceptors[idx_a]]

            # Multiple seqs, mutsplice is OFF
            elif _d.shape[1] == 2:
                idx_d = list(_d[_d[:, 1] == i][:, 0])
                idx_a = list(_a[_a[:, 1] == i][:, 0])

                d = [str(x) for x in idx_d]
                a = [str(x) for x in idx_a]

                d_p = [str(round(x, 3)) for x in donors[idx_d, i]]
                a_p = [str(round(x, 3)) for x in acceptors[idx_a, i]]

            pred_ss.write("{}\t{}\t{}\t{}\t{}\n".format(_name,
                                                        ';'.join(a),
                                                        ';'.join(d),
                                                        ';'.join(a_p),
                                                        ';'.join(d_p)))

        return [donors, acceptors]

    def report_all_predictions(self):
        """
        Report predictions for all positions in the input
        """
        for i, ss in enumerate([self.acceptor_preds, self.donor_preds]):
            
            dfs = []
            for seq_id in ss.columns:
                
                _df = self.df[self.df.id == seq_id]
                p = ss[seq_id].to_numpy().squeeze()
                p = p[~np.isnan(p)]
                
                if _df.iloc[0].Chromosome is None:
                    chrom = _df.iloc[0].id
                    start = 0
                    end = len(p)
                    strand = '+'
                else:
                    chrom = _df.iloc[0].Chromosome
                    start = _df.iloc[0].Start
                    end = _df.iloc[0].End
                    strand = _df.iloc[0].Strand

                assert _df.shape[0] == 1, "More than one sequence for seq_id: {}".format(seq_id)
                assert len(p) == end - start 
                
                ranges = [(i, i+1) for i in range(start, end)]
                aux = pd.DataFrame(ranges, columns=['Start', 'End'])
                aux['Chromosome'] = chrom
                aux['SpliceAI'] = np.round(p, decimals=5) if strand == '+' else np.round(p[::-1], decimals=5)
    
                dfs.append(aux)
            
            out = pd.concat(dfs)
            tag = "acceptor" if i == 0 else "donor"
            f  = os.path.join(self.outdir, self.outbasename + "_SpliceAI_all_preds_{}.bedgraph".format(tag))
            out[['Chromosome', 'Start', 'End', 'SpliceAI']].to_csv(f, sep='\t', index=False, header=False)
        
    def get_differing_positions(self,
                                minimum_difference: float = 0.05) -> Tuple:
        """
        Return positions in the sequence
        that differ by any mutation by a
        given amount

        :param float minimum_difference: Min difference in
        prediction to consider the change as relevant.
        Default: `0.05`

        :return Tuple: Tuple with 2 dfs containing the
        predictions on variable positions in the sequence
        (donors and acceptors, respectively)
        """
        # Mask to get positions that differ from the ref sequence by a given ammount
        # mask_donor = (np.abs(np.subtract(preds_donor, ref_col_val, dtype='float')) > 0.1).any(axis=1)

        # Mask to get positions that differ by any two mutations
        mask_donor = (self.donor_preds.max(axis=1) -
                      self.donor_preds.min(axis=1)) > minimum_difference
        mask_acceptor = (self.acceptor_preds.max(axis=1) -
                         self.acceptor_preds.min(axis=1)) > minimum_difference

        _donor_high = self.donor_preds[mask_donor].T
        _acceptor_high = self.acceptor_preds[mask_acceptor].T

        # Check positions with changing scores for both donor and acceptor
        # Likely to happen when minimum difference is very tiny.
        # Keeps position in df with highest median difference
        common_pos = list(set(_donor_high.columns).intersection(
            set(_acceptor_high.columns)))
        if common_pos:
            _d = _donor_high[common_pos].median(axis=0)
            _a = _acceptor_high[common_pos].median(axis=0)
            s = pd.concat([_a, _d], axis=1)
            s.columns = ['a', 'd']

            d_to_drop = s[s.d <= s.a].index.tolist()
            a_to_drop = s[s.a < s.d].index.tolist()

            _donor_high = _donor_high.drop(columns=d_to_drop)
            _acceptor_high = _acceptor_high.drop(columns=a_to_drop)

        return _donor_high, _acceptor_high

    def fix_coordinates(self):
        """
        Fix predictions indexes when the reference sequence was
        mutated with indels so that the prediction indexes match
        between sequences

        :return:
        """
        def _fix(x: pd.Series, replace_offset: bool = True):
            """
            Fix individual sequence

            :param bool replace_offset: Replace
            absent predictions caused by the
            sequence offset created by a deletion.
            If `True`, replaces the NAs by 0,
            meaning that a deletion occurring in
            a highly probable splice site will
            cause its disruption.
            """

            if "REF_seq" in x.name:
                return x

            _metadata = self.metadata[self.metadata.id == x.name]

            for mut in _metadata.itertuples():

                if mut.type == "DEL":
                    # Get NAs stored in the last indexes
                    del_ = x.tail(mut.end - (mut.start))

                    # Drop them from the main seq
                    x.drop(del_.index, inplace=True)

                    # Insert DEL indexes in the proper place
                    seq_preds = pd.concat(
                        [x.iloc[:mut.start], del_,
                         x.iloc[mut.start:]]).reset_index(drop=True)

                    # Replace by 0 if replace_offset is set
                    return seq_preds.fillna(0) if replace_offset else seq_preds

                else:
                    return x

        if self.metadata is not None:
            self.acceptor_preds = self.acceptor_preds.apply(_fix)
            self.donor_preds = self.donor_preds.apply(_fix)

    def plot_predictions(self,
                         ref_ss: list,
                         donor_high: pd.DataFrame,
                         acceptor_high: pd.DataFrame,
                         splicing_factor: str = None,
                         max_motifs_to_display: int = 50):
        """
        Plot spliceAI predictions based on
        known genomic coordinates of exons/introns

        :param list ref_ss: List with the predicted
            splice sites in the reference sequence.
            First element: donors, Second element: acceptors
        :param pd.DataFrame donor_high: Positions whose
            donors predictions vary by more than 0.2 in
            at least one mutated sequence
        :param pd.DataFrame acceptor_high: Positions whose
            acceptor predictions vary by more than 0.2 in
            at least one mutated sequence
        :param str splicing_factor: Splicing factor name
        :param int max_motifs_to_display: Max number of motif
        locations to display
        """
        ref_col = [x for x in self.donor_preds if "_REF_seq" in x][0]
        title = ref_col.split("_REF")[0]

        if "motif_start" in self.metadata.columns:
            seqs = self.metadata[self.metadata.seq_id == title]
            motif_pos = seqs[['motif_start', 'motif_end',
                              'rbp_name']].drop_duplicates().values.tolist()

        if not donor_high.empty:
            _donor_high = donor_high.copy()
            _donor_high = _donor_high.rename_axis('id').reset_index()
            _donor = _donor_high.melt(id_vars=['id'],
                                      var_name='Sequence position',
                                      value_name='Prediction upon mutation')
            _donor['class'] = 'donor'

        if not acceptor_high.empty:
            _acceptor_high = acceptor_high.copy()
            _acceptor_high = _acceptor_high.rename_axis('id').reset_index()

            _accept = _acceptor_high.melt(
                id_vars=['id'],
                var_name='Sequence position',
                value_name='Prediction upon mutation')
            _accept['class'] = 'acceptor'

        if donor_high.empty and acceptor_high.empty:
            diff_df_melted = None

        elif donor_high.empty:
            diff_df_melted = _accept

        elif acceptor_high.empty:
            diff_df_melted = _donor

        else:
            diff_df_melted = pd.concat([_donor, _accept])

        ref_seq_len = len(ref_ss[0])

        if self.is_cassette_exon:

            fig = plt.figure(figsize=(10, 7))
            spec = gridspec.GridSpec(nrows=3,
                                     ncols=1,
                                     height_ratios=[2, 2, 2],
                                     figure=fig)
            ax1 = fig.add_subplot(spec[0])
            ax1.title.set_text(title)
            ax2 = fig.add_subplot(spec[1], sharex=ax1)
            ax3 = fig.add_subplot(spec[2], sharex=ax1)
            # ax4 = fig.add_subplot(spec[3])

            # PLOT THE GENE STRUCTURE
            features = []
            upstream = self.ref_ss_idx[0]
            if all(coord in ['<NA>', np.nan] for coord in upstream):
                features.append(GraphicFeature(
                    start=0, end=0, color="#ffd700"))

            elif any(coord in ['<NA>', np.nan] for coord in upstream):
                features.append(GraphicFeature(
                    start=0, end=upstream[1], color="#ffd700", open_left=True))

            else:
                features.append(GraphicFeature(
                    start=upstream[0], end=upstream[1], color="#ffd700"))

            cassette = self.ref_ss_idx[1]
            features.append(GraphicFeature(start=cassette[0],
                                           end=cassette[1],
                                           color="#ffcccc"))

            downstream = self.ref_ss_idx[2]
            if all(coord in ['<NA>', np.nan] for coord in downstream):
                features.append(GraphicFeature(
                    start=ref_seq_len, end=ref_seq_len, color="#ffd700"))

            elif any(coord in ['<NA>', np.nan] for coord in downstream):
                features.append(GraphicFeature(
                    start=downstream[0], end=ref_seq_len, color="#ffd700", open_right=True))

            else:
                features.append(GraphicFeature(
                    start=downstream[0], end=downstream[1], color="#ffd700"))

            single_motif = True if len(motif_pos) == 1 else False
            label_d = {}

            if len(motif_pos) < max_motifs_to_display:
                for i, m_pos in enumerate(motif_pos):
                    if splicing_factor:
                        label = splicing_factor if single_motif else "{}_{}".format(
                            splicing_factor, i)

                    else:
                        label = m_pos[2]

                    label_d[str(m_pos[0]) + "_" + str(m_pos[1])] = label
                    features.append(
                        GraphicFeature(start=m_pos[0],
                                       end=m_pos[1],
                                       color="grey",
                                       label=label))

            record = GraphicRecord(sequence_length=ref_seq_len,
                                   features=features)
            record.plot(ax=ax1, figure_width=10)

            # PLOT THE SPLICEAI PREDICTION ON THE REFERENCE SEQUENCE
            plt.sca(ax2)
            plt.plot(ref_ss[0], color='maroon', label='Donor')
            plt.plot(ref_ss[1], color='mediumturquoise', label='Acceptor')
            plt.legend(bbox_to_anchor=(1, 1))
            # ax2.set_ylim(bottom=0)
            ax2.set_ylabel("SpliceAI")

            # PLOT THE DISPERSION IN PREDICTIONS AFTER MUTATING
            if diff_df_melted is not None:
                plt.sca(ax3)

                no_ref_preds = diff_df_melted[~diff_df_melted.id.str.
                                              contains('REF')]
                ref_preds = diff_df_melted[diff_df_melted.id.str.contains(
                    'REF')]

                n_alt_seqs = no_ref_preds.groupby('id').ngroups
                if n_alt_seqs <= 3:
                    c = ['tan', 'slategrey', 'lightblue', 'purple']
                    i = 0

                    for k, d in no_ref_preds.groupby('id'):
                        label = "DEL_" + k.split(
                            "DEL_")[1] if "DEL" in k else k.split(
                                "_motif_at_")[1]
                        plt.scatter(d["Sequence position"],
                                    d["Prediction upon mutation"],
                                    label=label,
                                    c=c[i],
                                    edgecolors='black')
                        i += 1
                else:
                    plt.scatter(no_ref_preds["Sequence position"],
                                no_ref_preds["Prediction upon mutation"],
                                c="grey",
                                alpha=0.8,
                                edgecolor='black')

                plt.scatter(ref_preds["Sequence position"],
                            ref_preds["Prediction upon mutation"],
                            c="darkred",
                            marker='x',
                            s=100,
                            label='Reference seq')

                ax3.set_ylim(0, 1)
                ax3.set_ylabel('Prediction upon mutation')
                ax3.set_xlabel('Sequence position')
                plt.legend(bbox_to_anchor=(1.17, 1))

                # # PLOT THE DISPERSION IN MORE DETAIL
                # plt.sca(ax4)
                # sns.stripplot(x="Sequence position",
                #               y="Prediction upon mutation",
                #               hue="mutation_pos_within_motif",
                #               data=no_ref_preds,
                #               size=4,
                #               linewidth=0.5,
                #               edgecolor='black',
                #               palette="Purples")
                #
                # ax4.set_ylabel('Prediction upon mutation')
                # ax4.set_xlabel('Differing positions')
                # plt.legend(title="Pos mutated in motif", bbox_to_anchor=(1.19, 1))

            plt.tight_layout(pad=1)
            plt.savefig(
                os.path.join(self.outdir,
                             self.outbasename + "_predictions.pdf"))

    def plot_significant_predictions(self,
                                     summary: pd.DataFrame,
                                     ref_ss: list,
                                     sign_thresh: float = 0.05,
                                     discard_opposite_effects: bool = True,
                                     use_max_effect: bool = True,
                                     max_motifs_to_display: int = 50):
        """
        Plot spliceAI perturbations that led to a significant 
        difference in the cassette exon predictions. Top 50 
        perturbations are displayed

        :param pd.DataFrame: Df with results of spliceAI perturbations

        :param list ref_ss: List with the predicted
            splice sites in the reference sequence.
            First element: donors, Second element: acceptors

        :param flot sign_thresh: Threshold to consider a spliceAI 
        difference in prediction as significant. Default: 0.05

        :param bool discard_opposite_effects: Discard perturbations
        that have opposing effects in acceptor or donor (e.g. positive 
        in donor, negative in acceptor). Default: `True`

        :param bool use_max_effect: Use max value between donor and 
        acceptor predictions. Default: `True`. If `False`, it uses
        the average effect between the two splice sites of the exon

        :param int max_motifs_to_display: Max number of motif
        locations to display
        """
        # Get significant perturbations
        effect_cols = ['acceptor_cassette_effect',
                       'donor_cassette_effect']

        sign_eff = summary[(summary.acceptor_cassette_effect.abs() >= sign_thresh) |
                           (summary.donor_cassette_effect.abs() >= sign_thresh)]

        sign_eff = sign_eff[['id', 'seq_id'] + effect_cols]

        if discard_opposite_effects:
            sign_eff = sign_eff.set_index(['id', 'seq_id'])
            sign_eff = sign_eff[(sign_eff > 0).all(
                axis=1) | (sign_eff < 0).all(axis=1)]
            sign_eff = sign_eff.reset_index()

        mut_metadata = self.metadata.copy()
        mut_metadata = pd.merge(mut_metadata,
                                sign_eff,
                                how='left',
                                indicator=True)
        mut_metadata = mut_metadata[mut_metadata["_merge"] == "both"]

        if mut_metadata.shape[0] == 0:
            logger.info(
                'No perturbations with significant effect ({}) on cassette exon.'.format(sign_thresh))
            return

        if use_max_effect:
            def max_eff(x): return max(x.acceptor_cassette_effect,
                                       x.donor_cassette_effect, key=abs)
            mut_metadata['effect'] = mut_metadata[effect_cols].apply(
                max_eff, axis=1)
        else:
            mut_metadata['effect'] = mut_metadata[effect_cols].mean(axis=1)

        # Top perturbations
        mut_metadata = mut_metadata.sort_values(
            'effect', ascending=False).head(max_motifs_to_display)

        ref_col = [x for x in self.donor_preds if "_REF_seq" in x][0]
        title = ref_col.split("_REF")[0]

        seqs = mut_metadata[mut_metadata.seq_id == title]
        seqs['effect_bin'] = pd.cut(x=seqs.effect,
                                    bins=[-1, -0.6, -0.2, 0, 0.2, 0.6, 1],
                                    labels=['Very_Strong_Neg', 'Strong_Neg', 'Moderate_Neg',
                                            'Moderate_Pos', 'Strong_Pos', 'Very_Strong_Pos'])

        motif_pos = seqs[['motif_start', 'motif_end', 'type',
                          'rbp_name', 'effect', 'effect_bin']].drop_duplicates().values.tolist()

        ref_seq_len = len(ref_ss[0])

        if self.is_cassette_exon:

            fig = plt.figure(figsize=(10, 7))
            spec = gridspec.GridSpec(nrows=2,
                                     ncols=1,
                                     height_ratios=[2, 2],
                                     figure=fig)
            ax1 = fig.add_subplot(spec[0])
            ax1.title.set_text(title)
            ax2 = fig.add_subplot(spec[1], sharex=ax1)

            # PLOT THE GENE STRUCTURE
            features = []
            upstream = self.ref_ss_idx[0]
            if all(coord in ['<NA>', np.nan] for coord in upstream):
                features.append(GraphicFeature(
                    start=0, end=0, color="#ffd700"))

            elif any(coord in ['<NA>', np.nan] for coord in upstream):
                features.append(GraphicFeature(
                    start=0, end=upstream[1], color="#ffd700", open_left=True))

            else:
                features.append(GraphicFeature(
                    start=upstream[0], end=upstream[1], color="#ffd700"))

            cassette = self.ref_ss_idx[1]
            features.append(GraphicFeature(start=cassette[0],
                                           end=cassette[1],
                                           color="#ffcccc"))

            downstream = self.ref_ss_idx[2]
            if all(coord in ['<NA>', np.nan] for coord in downstream):
                features.append(GraphicFeature(
                    start=ref_seq_len, end=ref_seq_len, color="#ffd700"))

            elif any(coord in ['<NA>', np.nan] for coord in downstream):
                features.append(GraphicFeature(
                    start=downstream[0], end=ref_seq_len, color="#ffd700", open_right=True))

            else:
                features.append(GraphicFeature(
                    start=downstream[0], end=downstream[1], color="#ffd700"))

            # CREATE DICT MAPPING COLOR TO RANGE OF VALUES
            bin_map = {'Very_Strong_Neg': 'darkred',
                       'Strong_Neg': 'salmon',
                       'Moderate_Neg': 'peachpuff',
                       'Moderate_Pos': 'powderblue',
                       'Strong_Pos': 'steelblue',
                       'Very_Strong_Pos': 'darkblue'}

            for i, m_pos in enumerate(motif_pos):
                mut_type = m_pos[2]
                label = m_pos[3]
                effect_bin = m_pos[5]
                try:
                    _color = bin_map[effect_bin]

                    features.append(
                        GraphicFeature(start=m_pos[0],
                                    end=m_pos[1],
                                    color=_color,
                                    label=label))
                except KeyError:
                    continue

            record = GraphicRecord(sequence_length=ref_seq_len,
                                   features=features)
            record.plot(ax=ax1, figure_width=10)

            legend_elements = [Patch(facecolor='darkred', edgecolor='black', label='Very_Strong_Neg'),
                               Patch(facecolor='salmon',
                                     edgecolor='black', label='Strong_Neg'),
                               Patch(facecolor='peachpuff',
                                     edgecolor='black', label='Moderate_Neg'),
                               Patch(facecolor='powderblue',
                                     edgecolor='black', label='Moderate_Pos'),
                               Patch(facecolor='steelblue',
                                     edgecolor='black', label='Strong_Pos'),
                               Patch(facecolor='darkblue', edgecolor='black', label='Very_Strong_Pos')]
            ax1.legend(legend_elements, ['Very_Strong_Neg', 'Strong_Neg', 'Moderate_Neg',
                                            'Moderate_Pos', 'Strong_Pos', 'Very_Strong_Pos'], title='Perturbation effect')

            # PLOT THE SPLICEAI PREDICTION ON THE REFERENCE SEQUENCE
            plt.sca(ax2)
            plt.plot(ref_ss[0], color='maroon', label='Donor')
            plt.plot(ref_ss[1], color='mediumturquoise', label='Acceptor')
            plt.legend(bbox_to_anchor=(1, 1))
            # ax2.set_ylim(bottom=0)
            ax2.set_ylabel("SpliceAI")

            plt.tight_layout(pad=1)
            plt.savefig(
                os.path.join(self.outdir,
                             self.outbasename + "_predictions_significant.pdf"))

    def plot_predictions_no_Mutsplice(self):
        """
        Plot spliceAI predictions based on
        known genomic coordinates of exons/introns

        :param list ref_ss: List with the predicted
            splice sites in the reference sequence.
            First element: donors, Second element: acceptors
        :param str splicing_factor: Splicing factor name
        """
        donors = self.donor_preds.copy()
        acceptors = self.acceptor_preds.copy()

        for seq_name in donors.columns:
            _donors = donors[seq_name].dropna()
            _acceptors = acceptors[seq_name].dropna()

            ref_seq_len = _donors.size

            fig = plt.figure(figsize=(7, 3))
            spec = gridspec.GridSpec(nrows=2,
                                     ncols=1,
                                     height_ratios=[0.4, 1],
                                     figure=fig)

            ax1 = fig.add_subplot(spec[0])
            ax1.title.set_text(seq_name)
            ax2 = fig.add_subplot(spec[1], sharex=ax1)

            # PLOT THE GENE STRUCTURE
            features = []
            if self.is_cassette_exon:

                ref_ss_idx = self.ref_ss_idx[seq_name]

                upstream = ref_ss_idx[0]
                if all(coord in ['<NA>', np.nan] for coord in upstream):
                    features.append(GraphicFeature(
                        start=0, end=0, color="#ffd700"))

                elif any(coord in ['<NA>', np.nan] for coord in upstream):
                    features.append(GraphicFeature(
                        start=0, end=upstream[1], color="#ffd700", open_left=True))

                else:
                    features.append(GraphicFeature(
                        start=upstream[0], end=upstream[1], color="#ffd700"))

                cassette = ref_ss_idx[1]
                features.append(GraphicFeature(start=cassette[0],
                                               end=cassette[1],
                                               color="#ffcccc"))

                downstream = ref_ss_idx[2]
                if all(coord in ['<NA>', np.nan] for coord in downstream):
                    features.append(GraphicFeature(
                        start=ref_seq_len, end=ref_seq_len, color="#ffd700"))

                elif any(coord in ['<NA>', np.nan] for coord in downstream):
                    features.append(GraphicFeature(
                        start=downstream[0], end=ref_seq_len, color="#ffd700", open_right=True))

                else:
                    features.append(GraphicFeature(
                        start=downstream[0], end=downstream[1], color="#ffd700"))

            record = GraphicRecord(sequence_length=ref_seq_len,
                                   features=features)

            record.plot(ax=ax1, figure_width=10)

            # PLOT THE SPLICEAI PREDICTION ON THE REFERENCE SEQUENCE
            plt.sca(ax2)
            plt.plot(_donors, color='maroon', label='Donor')
            plt.plot(_acceptors, color='mediumturquoise', label='Acceptor')
            plt.legend(bbox_to_anchor=(1, 1))

            ax2.set_ylabel("SpliceAI")

            plt.tight_layout(pad=1)
            plt.savefig(os.path.join(self.outdir,
                                     seq_name.replace("(", "_").replace(")", "_").replace(":", "_") + "_predictions.pdf"))
            plt.close()

    def generate_summary_no_Mutsplice(self):
        """
        Create a report of spliceAI runs
        where each sequence is independent

        :return:
        """
        def _get_high_preds(seq_p: pd.Series, donor: bool = True):
            seq_name = seq_p.name

            if self.ref_ss_idx:
                if donor:
                    exon_ups = self.ref_ss_idx[seq_name][0][1]
                    exon_cass = self.ref_ss_idx[seq_name][1][1]
                    exon_down = self.ref_ss_idx[seq_name][2][1]
                else:
                    exon_ups = self.ref_ss_idx[seq_name][0][0]
                    exon_cass = self.ref_ss_idx[seq_name][1][0]
                    exon_down = self.ref_ss_idx[seq_name][2][0]

                try:
                    pred_ups = seq_p[exon_ups]
                except KeyError:
                    pred_ups = np.nan

                pred_cass = seq_p[exon_cass]

                try:
                    pred_down = seq_p[exon_down]
                except KeyError:
                    pred_down = np.nan

                other_idx = [
                    x for x in seq_p.index[seq_p > 0.2].tolist()
                    if x not in [exon_ups, exon_cass, exon_down]
                ]
                other_preds = seq_p[other_idx].tolist()
                return [
                    exon_ups, exon_cass, exon_down, pred_ups, pred_cass, pred_down,
                    ';'.join([str(round(x, 2)) for x in other_idx]),
                    ';'.join([str(round(x, 2)) for x in other_preds])
                ]

        donors = self.donor_preds.copy()
        acceptors = self.acceptor_preds.copy()

        d = donors.apply(_get_high_preds)
        a = acceptors.apply(_get_high_preds, donor=False)

        accept_cols = [
            'start_exon_upstream', 'start_exon_cassette',
            'start_exon_downstream', 'ref_acceptor_upstream',
            'ref_acceptor_cassette', 'ref_acceptor_downstream',
            'other_acceptor_position', 'other_acceptor_preds'
        ]

        donor_cols = [
            'end_exon_upstream', 'end_exon_cassette', 'end_exon_downstream',
            'ref_donor_upstream', 'ref_donor_cassette', 'ref_donor_downstream',
            'other_donor_position', 'other_donor_preds'
        ]
        d.index = donor_cols
        a.index = accept_cols

        summary = pd.merge(d.T, a.T, right_index=True, left_index=True)
        summary['target_coordinates'] = summary.apply(
            lambda x: self.name[x.name], axis=1)
        summary = summary.reset_index().rename(columns={'index': 'seq_id'})

        summary.to_csv(os.path.join(self.outdir, self.outbasename + "_output.tsv.gz"),
                       sep="\t",
                       compression='gzip',
                       index=False)
        return summary

    def generate_summary(self,
                         donor_high: pd.DataFrame,
                         acceptor_high: pd.DataFrame,
                         motif_hits: dict,
                         splicing_factor: str = None):
        """
        Create a report of the run

        :param list ref_ss: List with the predicted
        splice sites in the reference sequence.
        First element: donors, Second element: acceptors

        :param pd.DataFrame donor_high: Indexes with a relevant
        donor diff in spliceAI prediction after mutating the
        reference sequence

        :param pd.DataFrame acceptor_high: Indexes with a relevant
        acceptor diff in spliceAI prediction after mutating the
        reference sequence

        :param dict motif_hits: Dict with the info the location 
        of motif ocurrences

        :param str splicing_factor: Name of the splicing factor
        that was mutated
        """

        def _get_preds_at_splice_sites(input: Union[dict, pd.Series],
                                       donor_preds: np.ndarray,
                                       acceptor_preds: np.ndarray):
            """
            Save indexes and predictions on the splice sites
            of upstream, cassette and downstream exons.

            If input is dict, refers to the REF sequence
            If input is df, refers to a mutated sequence
            """

            input = _adjust_ss_idx_on_mutated_seqs(input, self.ref_ss_idx)

            # SS indexes in the predictions arrays
            # are normalized to the reference seq
            input['ref_acceptor_cassette'] = acceptor_preds[self.ref_ss_idx[1][0]]
            input['ref_donor_cassette'] = donor_preds[self.ref_ss_idx[1][1]]

            map_ss = {'ref_acceptor_upstream': self.ref_ss_idx[0][0],
                      'ref_donor_upstream': self.ref_ss_idx[0][1],
                      'ref_acceptor_downstream': self.ref_ss_idx[2][0],
                      'ref_donor_downstream': self.ref_ss_idx[2][1]}

            for k, v in map_ss.items():
                aux = donor_preds if "donor" in k else acceptor_preds
                try:
                    input[k] = aux[v]
                except IndexError:
                    input[k] = pd.NA

            return input

        def _get_high_preds_at_other_pos(input: Union[dict, pd.Series],
                                         donor_preds: np.ndarray,
                                         acceptor_preds: np.ndarray):
            """
            Save indexes and predictions of the positions
            predicted to be splice sites that are different
            than the annotated upstream, cassette and downstream
            exons
            """
            flat_ss = list(itertools.chain(*self.ref_ss_idx))

            other_acceptor_idx = [x for x in np.argwhere(
                acceptor_preds > 0.2).flatten() if x not in flat_ss]
            other_acceptor_preds = acceptor_preds[other_acceptor_idx].tolist()

            other_donor_idx = [x for x in np.argwhere(
                donor_preds > 0.2).flatten() if x not in flat_ss]
            other_donor_preds = donor_preds[other_donor_idx].tolist()

            # If ref seq
            if isinstance(input, dict):
                other_acc_pos = ';'.join([str(int(x))
                                         for x in other_acceptor_idx])
                other_don_pos = ';'.join([str(int(x))
                                         for x in other_donor_idx])
            # If mutated seq
            else:
                other_acc_pos = ';'.join(
                    [str(_adjust_ss_idx_on_mutated_seqs(input, [], x)) for x in other_acceptor_idx])
                other_don_pos = ';'.join(
                    [str(_adjust_ss_idx_on_mutated_seqs(input, [], x)) for x in other_donor_idx])

            input['other_acceptor_preds'] = ';'.join(
                [str(round(x, 3)) for x in other_acceptor_preds])
            input['other_acceptor_position'] = other_acc_pos
            input['other_donor_preds'] = ';'.join(
                [str(round(x, 3)) for x in other_donor_preds])
            input['other_donor_position'] = other_don_pos

            return input

        ref_donor, ref_acceptor = "", ""

        high_preds = pd.merge(acceptor_high,
                              donor_high,
                              left_index=True,
                              right_index=True)

        if len(list(self.donor_preds)) == 1:
            logger.info('No alternative sequences.')
            return

        ref_info, summary = {}, []
        for header, _ in self.seqs.items():

            donor_preds_at_seq = self.donor_preds[header].to_numpy().squeeze()
            acceptor_preds_at_seq = self.acceptor_preds[header].to_numpy(
            ).squeeze()

            ############################
            ##### INFO ON REF SEQS #####
            ############################
            if "REF_seq" in header:
                ref_donor = donor_high.loc[header]
                ref_acceptor = acceptor_high.loc[header]

                ref_info['target_coordinates'] = self.name
                ref_info = _get_preds_at_splice_sites(ref_info,
                                                      donor_preds_at_seq,
                                                      acceptor_preds_at_seq)

                ref_info = _get_high_preds_at_other_pos(ref_info,
                                                        donor_preds_at_seq,
                                                        acceptor_preds_at_seq)
                ref_info['id'] = header
                ref_info['seq_id'] = header.replace("_REF_seq", "")
                ref_info = pd.DataFrame(ref_info, index=[0])
                continue

            ###############################
            ##### INFO ON MUTATED SEQS ####
            ###############################
            id_match = self.metadata[self.metadata.id == header].copy()
            duplicate_motifs = True if id_match.shape[0] > 1 else False

            _metadata = id_match.iloc[0].copy()
            _metadata['target_coordinates'] = self.name

            _metadata = _get_preds_at_splice_sites(_metadata,
                                                   donor_preds_at_seq,
                                                   acceptor_preds_at_seq)

            _metadata = _get_high_preds_at_other_pos(_metadata,
                                                     donor_preds_at_seq,
                                                     acceptor_preds_at_seq)

            # Get location info
            new_cols = ['rbp_name', 'rbp_motif', 'Start', 'End', 'rbp_name_motif', 'has_self_submotif',
                        'has_other_submotif', 'is_high_density_region',
                        'n_at_density_block', 'distance_to_cassette_acceptor', 'distance_to_cassette_donor',
                        'is_in_exon', 'location', 'distance_to_acceptor', 'distance_to_donor']

            _loc_df = pd.DataFrame(
                motif_hits[_metadata.seq_id], columns=new_cols)

            loc = _loc_df[(_loc_df.Start == _metadata.motif_start) &
                          (_loc_df.End == _metadata.motif_end) &
                          (_loc_df.rbp_name == _metadata.rbp_name)].iloc[0, 5:]
            _metadata = pd.concat([_metadata, loc])

            # Measure mutation effects
            accept_ups_effect, donor_ups_effect, accept_cassette_effect, donor_cassette_effect, \
                accept_down_effect, donor_down_effect = (0,) * 6

            other_donor_affected, other_donor_pos, other_donor_dist_to_mutation = [], [], []
            other_acceptor_affected, other_acceptor_pos, other_acceptor_dist_to_mutation = [], [], []

            # Iterate over high scoring differences
            for i, pred_class in enumerate([ref_acceptor, ref_donor]):

                for pos, pred in pred_class.items():

                    diff = round(high_preds.loc[header, pos] - pred, 3)

                    # Record all values
                    # if abs(diff) > 0.2:
                    if pos == self.ref_ss_idx[0][0]:
                        accept_ups_effect = diff

                    elif pos == self.ref_ss_idx[0][1]:
                        donor_ups_effect = diff

                    elif pos == self.ref_ss_idx[1][0]:
                        accept_cassette_effect = diff

                    elif pos == self.ref_ss_idx[1][1]:
                        donor_cassette_effect = diff

                    elif pos == self.ref_ss_idx[2][0]:
                        accept_down_effect = diff

                    elif pos == self.ref_ss_idx[2][1]:
                        donor_down_effect = diff

                    elif abs(diff) > 0.2:
                        if _metadata.start < pos < _metadata.end:
                            dist_to_mut = 0
                        else:
                            dist_to_mut = min(abs(pos - _metadata.start),
                                              abs(pos - _metadata.end))

                        pos = _adjust_ss_idx_on_mutated_seqs(_metadata,
                                                             self.ref_ss_idx,
                                                             other_idx=pos)
                        if i == 0:
                            other_acceptor_pos.append(pos)
                            other_acceptor_dist_to_mutation.append(dist_to_mut)
                            other_acceptor_affected.append(diff)

                        else:
                            other_donor_pos.append(pos)
                            other_donor_dist_to_mutation.append(dist_to_mut)
                            other_donor_affected.append(diff)

            new_cols = {
                'acceptor_upstream_effect': accept_ups_effect,
                'donor_upstream_effect': donor_ups_effect,
                'acceptor_cassette_effect': accept_cassette_effect,
                'donor_cassette_effect': donor_cassette_effect,
                'acceptor_downstream_effect': accept_down_effect,
                'donor_downstream_effect': donor_down_effect,
                'other_acceptor_effect': other_acceptor_affected,
                'other_acceptor_distance_to_mutation':
                other_acceptor_dist_to_mutation,
                'other_acceptor_affected_position': other_acceptor_pos,
                'other_donor_effect': other_donor_affected,
                'other_donor_distance_to_mutation':
                other_donor_dist_to_mutation,
                'other_donor_affected_position': other_donor_pos
            }

            for k, v in new_cols.items():
                _v = ';'.join([str(x)
                               for x in v]) if isinstance(v, list) else v
                _metadata[k] = _v

            if splicing_factor:
                _metadata['rbp_name'] = splicing_factor

            _metadata = pd.DataFrame([_metadata])
            if duplicate_motifs:
                # Remove processed row and replace by the new
                id_match = id_match.iloc[1:, :]
                _metadata = pd.concat([_metadata, id_match])
                _metadata.fillna(method='ffill', inplace=True)

            summary.append(_metadata)

        summary = pd.concat(summary)
        final = pd.concat([ref_info, summary])
        final = final[list(summary)]

        final.to_csv(os.path.join(self.outdir,
                                  self.outbasename + "_output.tsv.gz"),
                     sep="\t",
                     compression='gzip',
                     index=False)
        return final
