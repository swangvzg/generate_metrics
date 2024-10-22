import pandas as pd
import numpy as np
import squidpy as sq
import scanpy as sc

import os
import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import rc_context
import scipy
import piecewise_regression
from skimage import filters
from sklearn.mixture import GaussianMixture
from matplotlib.backends.backend_pdf import PdfPages
from scipy.sparse import csr_matrix
from scipy.stats import ttest_ind
import datetime
import metrics
from scipy.optimize import curve_fit

def calc_qc_metrics(
    path_detected_trans,
    path_cell_by_gene,
    path_cell_metadata,
    fov_size = 0.202,  # 0.202 mm, M1.7 will be 0.298
    exp_table=None,
    path_to_codebook=None,
    outputfolderpath=None,
    expid=None,
    portion = 0.0001,
    plot=False,
    skip_spatial=True
    ):
    """
    Calculate quality control (QC) metrics for single-cell transcriptomics data.

    Parameters:
    -----------
    path_detected_trans : str
        Path to the file detected_transcript.csv file, can be local path or s3 path.

    path_cell_by_gene : str
        Path to the file cell_by_gene.csv file, can be local path or s3 path.

    path_cell_metadata : str
        Path to the file cell_metadata.csv file, can be local path or s3 path.

    exp_table : pandas DataFrame or None, optional
        Bulk RNA-seq expression data containing 'transcript_id' and 'count' columns.. Default is None.

    path_to_codebook : str or None, optional
        Path to the codebook file. Default is None.

    outputfolderpath : str or None, optional
        Path to the output folder where analysis results and plots will be saved. Default is None.

    expid : str or None, optional
        An experiment identifier for naming output files. Default is None.
     
    fov_size: float = 0.202,  
        M1.0: 0.202 mm, M1.7 will be 0.298

    plot : bool, optional
        Whether to generate and save plots. Set to True to generate plots. Default is True.

    Returns:
    --------
    qc_metrics : dict
        A dictionary containing calculated quality control metrics for the data.
    Saved files:
        if outputfolderpath exists
        expid_bycounts.jpg : histogram of total_counts, n_gene_by_counts, n_cell_by_counts
        expid_cb.jpg : checkerboard plot
        expid_moranI_1.jpg : Top three moran I genes spatial plot
        expid_moranI_2.jpg : Bottom three moran I genes spatial plot
        expid_piterror.jpg : Error plot per pit count
        expid_r2_bulk.jpg : correlation plot with bulk seq
        expid_res.csv : All output qc metrics
        expid_spatial.jpg : spatial plot of all cells
        expid_spillover.csv : spillover metrcs table
        expid_spillover_gene.csv : spillover metrics per gene table
        expid_umap.jpg : umap plot

    Notes:
    ------
    This function calculates a variety of quality control (QC) metrics for single-cell transcriptomics data.
    The metrics include transcript and cell-related statistics, gene expression dispersion, cell filtering,
    and gene enrichment. Plots and analysis results can be saved to the specified output folder.

    Example:
    --------
    path_detected_trans = 'detected_trans.csv'
    path_cell_by_gene = 'cell_by_gene.csv'
    path_cell_metadata = 'cell_metadata.csv'
    exp_table = pd.DataFrame(...)  # Experimental metadata
    path_to_codebook = 'codebook.csv'
    outputfolderpath = 'output_folder'
    expid = 'experiment_1'
    plot = True
    qc_results = calc_qc_metrics(path_detected_trans, path_cell_by_gene, path_cell_metadata,
                                 exp_table, path_to_codebook, outputfolderpath, expid, plot)
    """

    #print("current time:- ", datetime.datetime.now())
    #print("Reading the detected transcript file")
    
    qc_metrics, gene_count_dic, trs_count_dic, barcode_count_dic, data_picked_df, z_error_rates, z_counts, global_x_min, global_x_max, global_y_min, global_y_max, fov_count_dic, grid_10um = metrics.detected_trs_metrics(path_detected_trans, portion=portion, fov_size = fov_size)
    print("Portion: ", portion)
    # output gene_count_dic
    #print("save gene count to csv: ", f"{outputfolderpath}/{expid}_gene_count.csv")
    pd.DataFrame.from_dict(gene_count_dic, orient="index").to_csv(f"{outputfolderpath}/{expid}_gene_count.csv")
    
    print('data_picked_df.shape, ', data_picked_df.shape)
    # check if the detected_trasnscript.csv is too big #line > 1e8
    if qc_metrics['Total counts'] < 1e8: # not too big to fit in memory, read in the whole csv file
        print('Total count: ', qc_metrics['Total counts'], " < 1e8 ")
        print(f"Using the whole csv file: {path_detected_trans} in stead of a subset.")
        data_picked_df = pd.read_csv(path_detected_trans)
        #print('data_picked_df.shape, ', data_picked_df.shape)
        
    num_unique_barcode_id = qc_metrics['unique_coding_barcodes'] + qc_metrics['unique_blank_barcodes']
    
    if path_to_codebook is not None:
        #print("####  Calculate the bit error metrics ####")
        #print("current time:- ", datetime.datetime.now())
        codebook = pd.read_csv(path_to_codebook, index_col=False, comment='#')
        subset = codebook.copy()
        if "barcodeType" in subset.columns:
            subset = subset[subset["barcodeType"] == 'merfish']
            subset.dropna(axis=1, how="all", inplace=True)
        bitNames = [s for s in subset.columns if s not in ["name", "id", "barcodeType"]]
        codewords = np.array([[x[n] for n in bitNames] for i, x in subset.iterrows()])
        
        bcCounts = pd.DataFrame.from_dict(barcode_count_dic, orient="index").reset_index()
        bcCounts.columns = ['bc', 'count']
        bcCounts['bc'] = [int(i) for i in bcCounts['bc']]
        bcCounts.index = bcCounts['bc']
        for i in range(len(subset)):
            if i not in bcCounts.index:
                bcCounts.loc[i, 'bc'] = i
                bcCounts.loc[i, 'count'] = 0

        bitCounts = codewords.copy()

        for i in range(len(codewords)):
            bitCounts[i] = bitCounts[i] * bcCounts.loc[i, 'count']

        codingBitCounts = codewords.copy()
        coding_index = subset[~subset["name"].str.contains("Blank", case=False)].index
        for i in range(len(codewords)):
            if i in coding_index:
                codingBitCounts[i] = codingBitCounts[i] * bcCounts.loc[i, 'count']
            else:
                codingBitCounts[i] = codingBitCounts[i]*0

        blankBitCounts = codewords.copy()
        blank_index = subset[subset["name"].str.contains("Blank", case=False)].index
        for i in range(len(codewords)):
            if i in blank_index:
                blankBitCounts[i] = blankBitCounts[i] * bcCounts.loc[i, 'count']
            else:
                blankBitCounts[i] = blankBitCounts[i]*0

        perBitErrors = (blankBitCounts.sum(axis=0)/len(blank_index))/(codingBitCounts.sum(axis=0)/len(coding_index))
        qc_metrics['max_per_bit_error'] = np.max(perBitErrors)
        qc_metrics['min_per_bit_error'] = np.min(perBitErrors)
        qc_metrics['mean_per_bit_error'] = np.mean(perBitErrors)
        qc_metrics['median_per_bit_error'] = np.median(perBitErrors)
        qc_metrics['std_per_bit_error'] = np.std(perBitErrors)

        qc_metrics['fraction_genes_above_max_blank'] = len(bcCounts[bcCounts['bc'].isin(
            coding_index) & (bcCounts['count'] > bcCounts[bcCounts['bc'].isin(blank_index)]['count'].max())])/len(coding_index)
        qc_metrics['fraction_genes_above_mean_blank'] = len(bcCounts[bcCounts['bc'].isin(
            coding_index) & (bcCounts['count'] > bcCounts[bcCounts['bc'].isin(blank_index)]['count'].mean())])/len(coding_index)
        qc_metrics['fraction_genes_above_median_blank'] = len(bcCounts[bcCounts['bc'].isin(
            coding_index) & (bcCounts['count'] > bcCounts[bcCounts['bc'].isin(blank_index)]['count'].median())])/len(coding_index)

    #if 'transcript_confidence' in dt.columns:
    #    #print('#### Calculating transcript_confidence related metrics ####')
    #    #print("current time:- ", datetime.datetime.now())
    #    mean_confi_blank = coding_bar['transcript_confidence'].mean().compute()
    #    mean_confi_trs =  blank_bar['transcript_confidence'].mean().compute()
    #    std_confi_blank = blank_bar['transcript_confidence'].std().compute()
    #    max_confi_blank = blank_bar['transcript_confidence'].max().compute()
    #    num_trs_abv_max_blank = sum(coding_bar['transcript_confidence'] >= max_confi_blank)
    #    num_trs_abv_mean_blank = sum(coding_bar['transcript_confidence'] >= mean_confi_blank)
    #    qc_metrics['Mean confidence score of blank'] = mean_confi_blank
    #    qc_metrics['Mean confidence score of transcripts'] = mean_confi_trs
    #    qc_metrics['Num transcript abv max blank confv'] = num_trs_abv_max_blank
    #    qc_metrics['Num transcript abv mean blank confv'] = num_trs_abv_mean_blank
    #    qc_metrics['perc transcript abv max blank confv to total transcript'] = f"{(num_trs_abv_max_blank / total_transcripts)*100:.2f}%"
    #    qc_metrics['perc transcript abv mean blank confv to total transcript'] = f"{(num_trs_abv_mean_blank / total_transcripts)*100:.2f}%"



    if exp_table is not None:
        #print("####  Calculate the bulk-cor metrics ####")
        #print("current time:- ", datetime.datetime.now())
        trs_count_df = pd.DataFrame.from_dict(trs_count_dic, orient='index').reset_index()
        trs_count_df.columns = ['transcript_id', 'count']
        print('trs_count_df: ', trs_count_df.loc[0:10])
        merged_df, bulk_metrics, plot_metrics = segmented_regression(trs_count_df, exp_table)
        qc_metrics.update(bulk_metrics)


    # calculate the checkerboard
    #print("####  Calculate the checkerboard metrics ####")
    #print("current time:- ", datetime.datetime.now())
    cbresults = generate_metrics(data_picked_df, global_x_min, global_x_max, global_y_min, global_y_max, qc_metrics['unique_coding_barcodes'], fov_size)

    qc_metrics['cb_mean'] = cbresults[0]
    qc_metrics['cb_min'] = cbresults[1]
    qc_metrics['plane6_plane0_transcript_ratio'] = cbresults[2]
    qc_metrics['plane3_plane0_transcript_ratio'] = cbresults[3]
    qc_metrics['transcript_per_fov_per_gene'] = cbresults[4]


    # contain the blank
    #print("####  Calculate the per cell metrics ####")
    #print("current time:- ", datetime.datetime.now())
    adata_defined = False
    print('path_cell_by_gene: ', path_cell_by_gene)
    print('path_cell_metadata: ', path_cell_metadata)
    if path_cell_by_gene and path_cell_metadata and not skip_spatial:
        adata = make_AnnData(path_cell_by_gene, path_cell_metadata)
        adata_defined = True
        adata_all = adata.copy()

        # remove the blank
        adata = adata[:, ~adata.var.index.str.contains("Blank")]

        qc_metrics['Num unique cells'] = len(adata.obs)
        qc_metrics['Num unique genes'] = len(adata.var)

        # calculate QC metrics
        sc.pp.calculate_qc_metrics(adata, expr_type='counts', var_type='genes',
                               qc_vars=(), percent_top=(50, 100),
                               layer=None, use_raw=False, inplace=True, log1p=False,
                               parallel=None)

        expressed_genes_per_cell = np.sum(adata.X > 0, axis=1)
        #print(f"Mean transcript / cell: {adata.obs['barcodeCount'].mean()}")
        #print(f"Mean genes / cell: {np.mean(expressed_genes_per_cell)}")
        #print(f"Median transcript / cell: {adata.obs['barcodeCount'].median()}")
        #print(f"Median genes / cell: {np.median(expressed_genes_per_cell)}")
        qc_metrics['Mean transcript / cell'] = adata.obs['barcodeCount'].mean()
        qc_metrics['Mean genes / cell'] = np.mean(expressed_genes_per_cell)
        qc_metrics['Median transcript / cell'] = adata.obs['barcodeCount'].median()
        qc_metrics['Median genes / cell'] = np.median(expressed_genes_per_cell)

        #filter cells based on counts number and volume of the cell
        #if no filter,the number calculated cannot reflect real value of a cell
        # do minimum filtering by 10 counts per cell
        min_counts = 50
        min_volume = 50

        #calculate the percent of cells within this
        qc_metrics['Num filtered cells'] = ((adata.obs['barcodeCount'] >= min_counts) &
                 (adata.obs['volume'] >= min_volume)).sum()
        qc_metrics['perc filtered cells'] = f"{(qc_metrics['Num filtered cells']/qc_metrics['Num unique cells'])*100:.2f}%"

        adata=adata[(adata.obs['barcodeCount'] >= min_counts) & (adata.obs['volume'] >= min_volume)]

        # calculate again
        try:
            sc.pp.calculate_qc_metrics(adata, expr_type='counts', var_type='genes',
                               qc_vars=(), percent_top=(50, 100),
                               layer=None, use_raw=False, inplace=True, log1p=False,
                               parallel=None)
            qc_metrics['pct_dropout_by_counts_median'] = adata.var['pct_dropout_by_counts'].median()
            qc_metrics['n_cells_by_counts_median'] = adata.var['n_cells_by_counts'].median()
            qc_metrics['n_genes_by_counts_median'] = adata.obs['n_genes_by_counts'].median()
            qc_metrics['pct_dropout_by_counts_mean'] = adata.var['pct_dropout_by_counts'].mean()
            qc_metrics['n_cells_by_counts_mean'] = adata.var['n_cells_by_counts'].mean()
            qc_metrics['n_genes_by_counts_mean'] = adata.obs['n_genes_by_counts'].mean()
        except Exception as e:
            print("sc.pp.calculate_qc_metrics erro: ", e)


        ## calculate the highly variable genes, and dispersion of each gene across all cells
        #print("####  Calculate the pca, clustering metrics ####")
        #print("current time:- ", datetime.datetime.now())
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
        sc.pp.highly_variable_genes(adata)

        qc_metrics['Num highly variable genes'] = len(adata.var[adata.var.highly_variable])

        qc_metrics['gene_expression_dispersion_median'] = adata.var['dispersions'].median()
        qc_metrics['gene_expression_dispersion_mean'] = adata.var['dispersions'].mean()

        try:
            sc.tl.pca(adata, svd_solver='arpack')
            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=10)
            sc.tl.umap(adata)
            sc.tl.leiden(adata, resolution=1)
            qc_metrics['Num clusters'] = adata.obs['leiden'].nunique()
            # filtering cluster based its size
            cluster_sizes = adata.obs.groupby(['leiden']).size()
            size_cutoff = 100
            picked_leiden_clusters = cluster_sizes[cluster_sizes > size_cutoff].index.to_list()
            qc_metrics['Num clusters after filtering (> 100)'] = len(picked_leiden_clusters)
            if len(cluster_sizes) > 30:
                adata = adata[[i for i in adata.obs.index if adata.obs.loc[i, 'leiden'] in picked_leiden_clusters]].copy()
        except Exception as e:
            print('pca-umap-leiden step: ', e)
        #print("####  Calculate the moranI metrics ####")
        #print("current time:- ", datetime.datetime.now())

        # downsample to 0.5M for moranI analysis
        adata_down = adata_all.copy()
        if qc_metrics['Num unique cells'] > 500000:
            adata_down = adata_all[np.random.choice(list(adata_all.obs.index), 500000, replace=False)].copy()
        sq.gr.spatial_neighbors(adata_down, radius=20, coord_type="generic", delaunay=True)
        sq.gr.spatial_autocorr(adata_down,mode="moran",n_perms=100,n_jobs=5)
        moran = adata_down.uns['moranI']

        group_blank = moran[moran.index.str.contains('Blank')]
        group_others = moran[~moran.index.str.contains('Blank')]


        top_blank_genes = group_blank.nlargest(3, 'I').index
        top_other_genes = group_others.nlargest(3, 'I').index

        qc_metrics['mean_moranI_blank'] = group_blank['I'].mean()
        qc_metrics['max_moranI_blank'] = group_blank['I'].max()
        qc_metrics['mean_moranI_coding'] = group_others['I'].mean()
        qc_metrics['max_moranI_coding'] = group_others['I'].max()

    # calculate spillover
#    #print("####  Calculate spillover metrics ####")
#    cbg = pd.read_csv(path_cell_by_gene, index_col=0)
#    cbg.index = [str(x) for x in cbg.index.tolist()]

#    sq.gr.spatial_neighbors(adata, radius=20, coord_type="generic")
#    spillover_sum, spillover_worst, df_spillover, df_spillover_gene = calc_spillover_gene(adata, cbg)
#    qc_metrics['overall_spillover_ratio'] = spillover_sum
#    qc_metrics['worst_spillover_ratio'] = spillover_worst

#    if outputfolderpath is not None:
#        df_spillover.to_csv(os.path.join(outputfolderpath,expid+'_spillover.csv'))
#        df_spillover_gene.to_csv(os.path.join(outputfolderpath,expid+'_spillover_gene.csv'))

    # calculate thickness
    thickness_metrics = process_dataset(expid, qc_metrics, fov_count_dic, grid_10um, z_counts)
    qc_metrics = {**qc_metrics, **thickness_metrics}

    if plot:
        print("####  make plots ####")
        print("current time:- ", datetime.datetime.now())

        if exp_table is not None:

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            if plot_metrics['xx_plot'] is not None:

                ax[0].loglog(plot_metrics['x'],plot_metrics['y'],'.')
                ax[0].loglog(np.power(10, plot_metrics['xx_plot']), np.power(10,plot_metrics['yy_plot']))
                ax[0].vlines(10**plot_metrics['breakpoints'][0], np.min(np.log10(plot_metrics['x'])), 10**6, linestyle = '--', color = 'orange')

                textstr = '\n'.join((
                "inflection point = {:.2f} FPKM".format(10**plot_metrics['breakpoints'][0]),
                "low transcript correlation = {:.3f}, n = {}".format(qc_metrics['low_abundance_corr'], qc_metrics['low_abundance_count']),
                "high transcript correlation = {:.3f}, n = {}".format(qc_metrics['high_abundance_corr'], qc_metrics['high_abundance_count'])
                ))

                # these are matplotlib.patch.Patch properties
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

                # place a text box in upper left in axes coords
                ax[0].text(0.03, 0.96, textstr, fontsize=8, transform=ax[0].transAxes, ha='left', va='top',
                         bbox=props)
                ax[0].set_xlabel('FPKM')
                ax[0].set_ylabel('Copy Number, MERFISH Counts')
                ax[0].set_title('Segmented Regression')

            ax[1].loglog(merged_df['bulk'], merged_df['merfish'],'.')
            ax[1].set_xlabel('bulk')
            ax[1].set_ylabel('merfish')
            ax[1].set_title(f'r2:{qc_metrics["bulk_r2"]}')


            fig.tight_layout()
            if outputfolderpath is not None:
                plt.savefig(os.path.join(outputfolderpath,expid+'_r2_bulk.jpg'))
            plt.show()
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        axes[0,0].scatter(data_picked_df['global_x'],data_picked_df['global_y'],s=0.1,alpha=0.5)
        axes[0,0].set_xlabel('X (um)')
        axes[0,0].set_ylabel('Y (um)')

        axes[0,1].hist(data_picked_df['global_x'], bins=np.arange(data_picked_df['global_x'].min(), data_picked_df['global_x'].max(), 5))
        axes[0,1].set_xlabel('X (um)')
        axes[0,1].set_ylabel('Transcript count')

        axes[0,2].hist(data_picked_df['global_y'], bins=np.arange(data_picked_df['global_y'].min(), data_picked_df['global_y'].max(), 5))
        axes[0,2].set_xlabel('Y (um)')
        axes[0,2].set_ylabel('Transcript count')

        axes[1,0].bar(z_error_rates.index, z_error_rates.iloc[:, 0])
        axes[1,0].set_ylabel('Error rate')
        axes[1,0].set_xlabel('Z index')

        axes[1,1].bar(z_counts.index, z_counts.iloc[:, 0])
        axes[1,1].set_ylabel('Transcripts')
        axes[1,1].set_xlabel('Z index')

        #axes[1,2].hist(coding_bar[coding_bar['r'] < maxRadius]['r'], bins=np.arange(0, maxRadius, 50))
        #axes[1,2].hist(blank_bar[coding_bar['r'] < maxRadius]['r'], bins=np.arange(0, maxRadius, 50))
        #axes[1,2].plot([0, radialHistogramCoding[1][-1]], [0, radialHistogramCoding[0][-1]])
        #axes[1,2].plot([0, radialHistogramBlank[1][-1]], [0, radialHistogramBlank[0][-1]])
        #axes[1,2].set_ylabel('Count')
        #axes[1,2].set_xlabel('Radial distance (pixels)')

        # plt.plot([0, radialHistogram[1][-1]], [0, radialHistogram[0][-1]])
        # plt.xlabel('Radial distance (pixels)')
    # plt.ylabel('Count')

        fig.tight_layout()
        if outputfolderpath is not None:
            plt.savefig(os.path.join(outputfolderpath,expid+'_spa_.jpg'))
        plt.show()

        if path_to_codebook is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].bar(np.arange(len(bitCounts[0])), height=perBitErrors)
            axes[0].set_ylabel('Per bit error rate')
            axes[0].set_xlabel('Bit index')

            axes[1].semilogy(bcCounts[bcCounts['bc'].isin(coding_index)].sort_values('count')['count'].values, '.')
            axes[1].semilogy(np.arange(len(coding_index), len(coding_index) + len(blank_index)),
                                   bcCounts[bcCounts['bc'].isin(blank_index)].sort_values('count')['count'].values, '.')
            axes[1].legend(['Coding', 'Blank'])
            axes[1].set_ylabel('Count')
            fig.tight_layout()
            if outputfolderpath is not None:
                plt.savefig(os.path.join(outputfolderpath,expid+'_piterror.jpg'))
            plt.show()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        if adata_defined:
            sns.histplot(
                adata.obs["barcodeCount"],
                kde=False,
                label='adata',
                ax=axes[0]
            )

            sns.histplot(
                adata.obs["n_genes_by_counts"],
                kde=False,
                label='adata',
                ax=axes[1]
            )

            sns.histplot(
                adata.var["n_cells_by_counts"],
                kde=False,
                label='adata',
                ax=axes[2]
            )
        fig.tight_layout()
        if outputfolderpath is not None:
            plt.savefig(os.path.join(outputfolderpath,expid+'_bycounts.jpg'))
        plt.show()
        # genes per cell plot and Transcripts per cell plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        if adata_defined:
            sns.histplot(expressed_genes_per_cell, bins=50, kde=False, ax = axes[0])
            axes[0].set_xlabel('Number of genes per cell')
            axes[0].set_ylabel('Number of cells')
            axes[0].set_title('Genes Per Cell')
        
        if adata_defined:
            sns.histplot(adata.obs['barcodeCount'], bins=50, kde=False, ax = axes[1])
            axes[1].set_xlabel('Number of transcripts per cell')
            axes[1].set_ylabel('Number of cells')
            axes[1].set_title('Transcripts Per Cell')
        
        fig.tight_layout()
        if outputfolderpath is not None:
            plt.savefig(os.path.join(outputfolderpath,expid+'_PerCell.jpg'))
        plt.show()
        #print('adata.shape: ', adata.shape)
        try:
            if adata_defined and adata.shape[0] > 0:
                with plt.rc_context({'figure.figsize': (3, 3)}):
                    sc.pl.umap(adata, color=["leiden"], legend_loc='on data',legend_fontsize = 8, show=False)
                    if outputfolderpath is not None:
                        plt.savefig(os.path.join(outputfolderpath,expid+'_umap.jpg'), bbox_inches="tight")

                with plt.rc_context({'figure.figsize': (3, 3)}):
                    sc.pl.embedding(adata, 'spatial', color = 'leiden',size=1,cmap='plasma_r', show=False)
                    if outputfolderpath is not None:
                        plt.savefig(os.path.join(outputfolderpath,expid+'_spatial.jpg'), bbox_inches="tight")

                with plt.rc_context({'figure.figsize': (3, 3)}):
                    sc.pl.embedding(adata_all, 'spatial', color = top_other_genes,size=10, cmap="Reds", show=False)
                    if outputfolderpath is not None:
                        plt.savefig(os.path.join(outputfolderpath,expid+'_moranI_1.jpg'), bbox_inches="tight")

                with plt.rc_context({'figure.figsize': (3, 3)}):
                    sc.pl.embedding(adata_all, 'spatial', color = top_blank_genes,size=10,cmap="Reds", show=False)
                    if outputfolderpath is not None:
                        plt.savefig(os.path.join(outputfolderpath,expid+'_moranI_2.jpg'), bbox_inches="tight")
        except Exception as e:
            print("plot-umap step: ", e)

        merscope_checkerboard_analysis(data_picked_df, global_x_min, global_x_max, global_y_min, global_y_max, fov_size, qc_metrics['unique_coding_barcodes'], outputfolderpath, expid)
        
    # add corner to center ratio
    vmin_2percent_ratio, vmin_3percent_ratio, vmin_5percent_ratio = calc_corner_to_center_ratio(data_picked_df,fov_size)
    qc_metrics['corner_to_center_ratio_2per'] = float("%.2f" % vmin_2percent_ratio)
    qc_metrics['corner_to_center_ratio_3per'] = float("%.2f" % vmin_3percent_ratio)
    qc_metrics['corner_to_center_ratio_5per'] = float("%.2f" % vmin_5percent_ratio)

    if outputfolderpath is not None:
        df = pd.DataFrame.from_dict(qc_metrics, orient='index')
        #df.to_csv(os.path.join(outputfolderpath,expid+'_res.csv'))

    return qc_metrics

def calc_corner_to_center_ratio(data_df, fov_size):
    """
    From Shawn
    return the ratio of 2 percentile to mean, the ratio of 3 percentile to mean and the ratio of 5 percentile to mean
    """
    data_copy = data_df.copy()
    # M1 2048 2048/2 + 1835/2 = 1941, 2048/2 - 1835/2 = 106 [106, 1941] 
    # M1.7 2960/2 + 2642/2 = 2801 , 2960/2 - 2642/2 = 159, so the range should be [159, 2801]
    minlimit = 106
    maxlimit = 1941
    if fov_size != 0.202:
        minlimit = 159
        maxlimit = 2801
    data_copy = data_copy[(data_copy["x"] < maxlimit) & (data_copy["y"] < maxlimit) & (data_copy["x"] > minlimit) & (data_copy["y"] > minlimit)]
    h, xe, ye = np.histogram2d(data_copy["x"], data_copy["y"], bins=128)
    vmin = np.min(h)
    vmin_2percent = np.percentile(h, 2)
    vmin_3percent = np.percentile(h, 3)
    vmin_5percent = np.percentile(h, 5)
    vmean = np.mean(h)
    vmax = np.max(h)
    return(vmin_2percent/vmean, vmin_3percent/vmean, vmin_5percent/vmean)

def make_AnnData(cell_by_gene_path, meta_cell_path, min_count = 0):

    cell_by_gene = pd.read_csv(cell_by_gene_path, index_col=0)
    meta_cell = pd.read_csv(meta_cell_path, index_col=0)

    meta_cell['barcodeCount'] = cell_by_gene.sum(axis=1)

    # initialize meta_gene
    meta_gene = pd.DataFrame(index=cell_by_gene.columns.tolist())

    # Align the cell id of cell_metadata to cell_by_gene
    cell_id = cell_by_gene.index.tolist()
    meta_cell = meta_cell.loc[cell_id]

    # Check again
    if (cell_by_gene.index == meta_cell.index).sum() == len(cell_by_gene.index):
        print('The indices in cell_by_gene and cell_metadata match.')
    else:
        print('The indices in cell_by_gene and cell_metadata do not match.')

    coordinates =np.stack((meta_cell['center_x'], meta_cell['center_y']), axis=1)

    ad = sc.AnnData(X=cell_by_gene.values, obs=meta_cell, var=meta_gene, obsm={"spatial": coordinates})
    return ad

def merscope_checkerboard_values(transcripts, pixel_dimensions, cb_spacing):

    #print('pixel_dimensions ', pixel_dimensions)
    #print('transcripts: ', np.array(transcripts)[0:30])
    #print('Nan: ', [x for x in transcripts if x != x])
    transcripts = [x for x in transcripts if x == x]
    #print('len(transcripts): ', len(transcripts))
    hist, bins = np.histogram(transcripts, bins=np.arange(0,pixel_dimensions,1), density=True)
    bins_thr = bins[:-1][hist > 0.00001] # thresholded bins - extracts all bins for which hist > 0.0001
    #cb_spacing = 202 # checkerboard spacing in um
    #print('bins_thr[:10], ', bins_thr[:10])
    #print('np.amin(bins_thr), ', np.amin(bins_thr))
    #print('np.amax(bins_thr), ', np.amax(bins_thr))
    hist_thr = hist[int(np.amin(bins_thr)):int(np.amax(bins_thr))] # thresholded hist - removes all "empty" bins on the periphery
    cb_chunk = np.zeros(cb_spacing)
    fovs = int(np.floor(len(hist_thr)/cb_spacing))
    for fov in range(fovs):
        fov_chunk = hist_thr[(0+fov)*cb_spacing:(fov+1)*cb_spacing] # chunks are hist_thr values at one micron intervals within an fov
        cb_chunk = cb_chunk + fov_chunk/np.mean(fov_chunk) # determines checkerboard metric at each 1 micron within an fov
    return hist, cb_chunk

def generate_metrics(filtered_transcripts, global_x_min, global_x_max, global_y_min, global_y_max, n_genes, fov_size):
    """
    Generate various metrics from filtered transcript data.

    Parameters:
    -----------
    filtered_transcripts : numpy array or pandas DataFrame
        An array of transcript locations (x,y) excluding blanks or a DataFrame with relevant columns.

    n_genes : int
        Scalar integer representing the number of genes (usually 140 or 500).

    Returns:
    --------
    cb_mean : float
        Mean checkerboard metric value calculated across different z planes.

    cb_min : float
        Minimum checkerboard metric value calculated across different z planes.

    trans_ratio_6 : float
        Transcript count ratio between z plane 6 and z plane 0.

    trans_ratio_3 : float
        Transcript count ratio between z plane 3 and z plane 0.

    trans_per : float
        Average transcript count per gene per FOV.

    Notes:
    ------
    This function calculates various metrics from filtered transcript data.
    The metrics include checkerboard metric values, transcript count ratios,
    and average transcript count per gene per FOV.

    Example:
    --------
    filtered_transcripts = np.array([[x1, y1, z1], [x2, y2, z2], ...])
    n_genes = 140
    cb_mean, cb_min, ratio_6, ratio_3, avg_trans_per_gene = generate_metrics(filtered_transcripts, global_x_min, global_x_max, global_y_min, global_y_max, n_genes, fov_size)
    """


    # Subtract all x,y values by min x,y values to bring images in plots closer to axes
    #min_y = filtered_transcripts.global_y.min().compute()
    #min_x = filtered_transcripts.global_x.min().compute()
    #filtered_transcripts_adjusted = filtered_transcripts.copy()
    #filtered_transcripts_adjusted.loc[:,"global_y"] -= min_y
    #filtered_transcripts_adjusted.loc[:,"global_x"] -= min_x
    #max_y = int(filtered_transcripts.global_y.max().compute() - min_y)
    #max_x = int(filtered_transcripts.global_x.max().compute() - min_x)
    
    min_x, max_x, min_y, max_y = global_x_min, global_x_max, global_y_min, global_y_max
    max_y = max_y - min_y
    max_x = max_x - min_x
    filtered_transcripts_adjusted = filtered_transcripts.copy()
    filtered_transcripts_adjusted.loc[:,"global_y"] -= min_y
    filtered_transcripts_adjusted.loc[:,"global_x"] -= min_x
    # analyze by plane
    pixel_dimensions = np.round(max(max_x, max_y)) + 1000 # set axes sizes
    transcript_count = []
    cb_list = []
    #fig, ax = plt.subplots(4,4, figsize = (12, 12))

    for plane_number in range(7): # for each z plane
        plane = filtered_transcripts_adjusted[filtered_transcripts_adjusted['global_z'] == plane_number]
        #print('plane.shape, ', plane.shape)
        transcript_count.append(plane.shape[0]) # transcript count per plane
        #print(plane.loc[:10, :])
        ##print(pixel_dimensions)

        hist_x, cb_chunk_x = merscope_checkerboard_values(plane['global_x'], pixel_dimensions, int(fov_size*1000))
        hist_y, cb_chunk_y = merscope_checkerboard_values(plane['global_y'], pixel_dimensions, int(fov_size*1000))
        cb_x = np.min(cb_chunk_x)/np.max(cb_chunk_x) # min divided by max to see drop off in transcripts within an FOV
        cb_y = np.min(cb_chunk_y)/np.max(cb_chunk_y)
        cb_list.append((cb_x, cb_y))

    cb_mean = np.mean(cb_list)
    cb_min = np.min(cb_list)

    trans_ratio_6 = transcript_count[6]/transcript_count[0]
    trans_ratio_3 = transcript_count[3]/transcript_count[0]

    trans_per = filtered_transcripts_adjusted.groupby(['fov','gene'])['transcript_id'].count().reset_index()
    trans_per = trans_per['transcript_id'].mean()

    return (cb_mean, cb_min, trans_ratio_6, trans_ratio_3, trans_per)

def segmented_regression(trans_count, exp):
    """
    Perform segmented regression analysis on transcript expression data, and compared to bulk seq data.

    Parameters:
    -----------
    trans : pandas DataFrame
        Transcript data containing 'transcript_id' and 'count' columns.

    exp : pandas DataFrame
        Bulk RNA-seq expression data containing 'transcript_id' and 'count' columns.

    Returns:
    --------
    merged_df : pandas DataFrame
        A DataFrame containing merged transcript expression data with columns:
        - 'transcript_id': str
            The transcript identifier.
        - 'merfish': int
            The MERFISH transcript count.
        - 'bulk': int
            The bulk RNA-seq transcript count.

    bulk_metrics : dict
        A dictionary containing various bulk RNA-seq metrics and segmented regression results:
        - 'bulk_r2': float
            The correlation coefficient between log-transformed MERFISH and bulk RNA-seq expression.
        - 'correlation_inflection_pt': float
            The transcript abundance inflection point determined by segmented regression.
        - 'low_abundance_corr': float
            Correlation coefficient for the low-abundance transcript range.
        - 'high_abundance_corr': float
            Correlation coefficient for the high-abundance transcript range.
        - 'low_abundance_count': int
            Number of transcripts in the low-abundance range.
        - 'high_abundance_count': int
            Number of transcripts in the high-abundance range.

    plot_objects : dict
        A dictionary containing plot-related data for visualization:
        - 'x': ndarray
            Array of bulk RNA-seq expression values above a threshold.
        - 'y': ndarray
            Array of MERFISH transcript counts corresponding to 'x'.
        - 'xx_plot': ndarray
            Array of x-axis values for segmented regression plot.
        - 'yy_plot': ndarray
            Array of y-axis values calculated using segmented regression coefficients.
        - 'breakpoints': ndarray
            Array of breakpoint values determined by segmented regression.

    Notes:
    ------
    This function performs segmented regression analysis on transcript expression data.
    It calculates correlation coefficients, breakpoint values, and metrics for low
    and high abundance transcript ranges using log-transformed data.

    Example:
    --------
    trans = pd.DataFrame({'transcript_id': [...], 'count': [...]})
    exp = pd.DataFrame({'transcript_id': [...], 'count': [...]})
    merged_df, metrics, plot_data = segmented_regression(trans, exp)
    """

    #trans_count = trans['transcript_id'].value_counts().to_frame().reset_index().compute()
    #trans_count.columns = ['transcript_id','count']
    #exp = exp.reset_index()
    exp.columns = ['transcript_id','count']
    ##print(exp.loc[0:10])
    #trans_count['transcript_id'] = trans_count['transcript_id'].apply(pad_ensembl_id)
    merged_df = trans_count.merge(exp, on='transcript_id', how='inner')
    merged_df.columns = ['transcript_id','merfish','bulk']
    #print(merged_df.loc[0:10])
    merged_df = merged_df.fillna(0)
    merged_df = merged_df.loc[(merged_df['merfish'] > 0) & (merged_df['bulk'] > 0)]
    merged_df['bulk'] = [i if i > 0.0001 else 0.0001 for i in merged_df['bulk'] ] # remove extremly low numbers like 1e-150

    merfish_fpkm =np.log10(merged_df['merfish'])
    ref_fpkm = np.log10(merged_df['bulk'])
    corr = np.corrcoef(merfish_fpkm, ref_fpkm)[0,1]
    #print(corr)
    # calculate the ratio between merfish count/bulk count
    merged_df['ratio'] = merged_df['merfish'] / (merged_df['bulk'].replace(0, 1))
    mean_ratio = merged_df['ratio'].mean()
    std_ratio = merged_df['ratio'].std()

    # Calculate the upper and lower bounds for the range
    upper_bound = mean_ratio + 2 * std_ratio
    lower_bound = mean_ratio - 2 * std_ratio

    # Count the number of rows within and outside the range
    within_range_count = len(merged_df[(merged_df['ratio'] >= lower_bound) & (merged_df['ratio'] <= upper_bound)])
    outside_range_count = len(merged_df[(merged_df['ratio'] < lower_bound) | (merged_df['ratio'] > upper_bound)])

    x = merged_df[merged_df['bulk']>0.01]['bulk'].to_numpy() +0.0001
    y = merged_df[merged_df['bulk']>0.01]['merfish'].to_numpy()
    
    # Fit the piecewise regression model
    breakpoints = None
    breakpoints_1 = None
    pw_fit = piecewise_regression.Fit(np.log10(x), np.log10(y), n_breakpoints=2)
    if pw_fit.best_muggeo:
        breakpoints = sorted(pw_fit.best_muggeo.best_fit.next_breakpoints)[0]
    pw_fit_1 = piecewise_regression.Fit(np.log10(x), np.log10(y), n_breakpoints=1)
    if pw_fit_1.best_muggeo:
        breakpoints_1 = sorted(pw_fit_1.best_muggeo.best_fit.next_breakpoints)[0]

    if breakpoints and breakpoints_1:
        if breakpoints < breakpoints_1:
            pw_fit_best = pw_fit
        else:
            pw_fit_best = pw_fit_1
    elif breakpoints and not breakpoints_1:
        pw_fit_best = pw_fit
    elif not breakpoints and breakpoints_1:
        pw_fit_best = pw_fit_1
    else:
        bulk_metrics = {
            "bulk_r2": corr,
            "mean_ratio_merfish_by_bulk_count": mean_ratio,
            "stdv_ratio_merfish_by_bulk_count": std_ratio,
            "num_genes_within_mean_2stdv_ratio": within_range_count,
            "num_genes_outside_mean_2stdv_ratio": outside_range_count,
            "correlation_inflection_pt": None,
            "low_abundance_corr": None,
            "high_abundance_corr": None,
            "low_abundance_count": None,
            "high_abundance_count": None,
        }

        plot_objects = {
            "x": x,
            "y": y,
            "xx_plot": None,
            "yy_plot": None,
            "breakpoints": None,
        }
        return (merged_df, bulk_metrics, plot_objects)

    if pw_fit_best.best_muggeo:
        # Sort and extract the breakpoints
        breakpoints = sorted(pw_fit_best.best_muggeo.best_fit.next_breakpoints)

        # Check if there is at least one valid breakpoint
        if len(breakpoints) >= 1:
            final_params = pw_fit_best.best_muggeo.best_fit.raw_params

            # Extract the parameters
            intercept_hat = final_params[0]
            alpha_hat = final_params[1]
            beta_hats = final_params[2:2 + len(breakpoints)]  # Adjust to the number of breakpoints

            # Create a plot of the segmented regression
            xx_plot = np.linspace(min(pw_fit.xx), max(pw_fit.xx), 1000)
            yy_plot = intercept_hat + alpha_hat * xx_plot

            # Add segments based on the breakpoints
            for bp_count in range(len(breakpoints)):
                yy_plot += beta_hats[bp_count] * np.maximum(xx_plot - breakpoints[bp_count], 0)

            # Calculate metrics for low and high abundance transcripts based on the lower breakpoint
            low_abundance_transcripts = merged_df[(merged_df['bulk'] > 0.01) & (merged_df['bulk'] < 10**breakpoints[0])]
            high_abundance_transcripts = merged_df[(merged_df['bulk'] > 0.01) & (merged_df['bulk'] >= 10**breakpoints[0])]

            # Calculate the correlation coefficients for low and high abundance transcripts
            low_abundance_corrcoeff = np.corrcoef(np.log10(low_abundance_transcripts['bulk'] + 0.0001),
                                                  np.log10(low_abundance_transcripts['merfish']))[0, 1]

            high_abundance_corrcoeff = np.corrcoef(np.log10(high_abundance_transcripts['bulk'] + 0.0001),
                                                   np.log10(high_abundance_transcripts['merfish']))[0, 1]

            # Prepare the metrics for bulk data
            bulk_metrics = {
                "bulk_r2": corr,
                "mean_ratio_merfish_by_bulk_count": mean_ratio,
                "stdv_ratio_merfish_by_bulk_count": std_ratio,
                "num_genes_within_mean_2stdv_ratio": within_range_count,
                "num_genes_outside_mean_2stdv_ratio": outside_range_count,
                "correlation_inflection_pt": 10**breakpoints[0],  # Use the lower breakpoint
                "low_abundance_corr": low_abundance_corrcoeff,
                "high_abundance_corr": high_abundance_corrcoeff,
                "low_abundance_count": len(low_abundance_transcripts),
                "high_abundance_count": len(high_abundance_transcripts),
            }

            # Prepare the plot objects for visualization
            plot_objects = {
                "x": x,
                "y": y,
                "xx_plot": xx_plot,
                "yy_plot": yy_plot,
                "breakpoints": breakpoints
            }
    return (merged_df, bulk_metrics, plot_objects)
    
def merscope_checkerboard_analysis(filtered_transcripts, global_x_min, global_x_max, global_y_min, global_y_max, fov_size, n_genes, outputfolderpath=None, expid=None):
    """
    Perform checkerboard analysis on filtered transcript data, generate plot.

    Parameters:
    -----------
    filtered_transcripts : pandas DataFrame
        An array of transcript locations (x,y) excluding blanks or a DataFrame with relevant columns.

    n_genes : int
        Scalar integer representing the number of genes.

    outputfolderpath : str, optional
        Path to the output folder where plots will be saved. Default is None.

    expid : str, optional
        An experiment identifier for naming output files. Default is None.

    Returns:
    --------
    The main function here is used to generate plot, the return values will not be used in this case
    transcript_count : ndarray
        An array containing the transcript count per z plane.

    cb_list : ndarray
        An array containing checkerboarding information for each z plane.

    Example:
    --------
    dt = pd.read_csv(path_detected_trans, index_col = 0)
    dt['trans'] = dt['gene'].apply(lambda x: 'Blank' if x.startswith('Blank') else 'coding')
    coding_bar = dt[dt['trans']=='coding']
    n_genes = 140
    transcript_count, cb_list = merscope_checkerboard_analysis(coding_bar, n_genes)
    """

    # Subtract all x,y values by min x,y values to bring images in plots closer to axes
    min_x, max_x, min_y, max_y = global_x_min, global_x_max, global_y_min, global_y_max

    filtered_transcripts_adjusted = filtered_transcripts.copy()
    filtered_transcripts_adjusted.loc[:,"global_y"] -= min_y
    filtered_transcripts_adjusted.loc[:,"global_x"] -= min_x

    max_x = max_x - min_x
    max_y = max_y - min_y

    # analyze by plane
    pixel_dimensions = np.round(max(max_x, max_y)) + 1000 # set axes sizes
    transcript_count = []
    cb_list = []
    fig, ax = plt.subplots(4,4, figsize = (12, 12))

    for plane_number in range(7): # for each z plane
        plane = filtered_transcripts_adjusted[filtered_transcripts_adjusted['global_z'] == plane_number]
        transcript_count.append(plane.shape[0]) # transcript count per plane
        ##print(np.asarray(plane['global_x']))
        #print(pixel_dimensions)

        hist_x, cb_chunk_x = merscope_checkerboard_values(plane['global_x'], pixel_dimensions, int(fov_size*1000))
        hist_y, cb_chunk_y = merscope_checkerboard_values(plane['global_y'], pixel_dimensions, int(fov_size*1000))
        cb_x = np.min(cb_chunk_x)/np.max(cb_chunk_x) # min divided by max to see drop off in transcripts within an FOV
        cb_y = np.min(cb_chunk_y)/np.max(cb_chunk_y)
        cb_list.append((cb_x, cb_y))

        # plot image for each plane
        plane6 = filtered_transcripts_adjusted[filtered_transcripts_adjusted['global_z'] == 6]
        #if n_genes != 140:
        #    plane = plane.iloc[np.random.choice(plane.shape[0], int(np.round(0.02*len(plane6))), replace=False), :] # plot random sub-set of transcripts
        #else:
        #    plane = plane.iloc[np.random.choice(plane.shape[0], len(plane6), replace=False), :] # plot random sub-set of transcripts
        if plane_number < 4:
            ax[0][plane_number].plot(plane['global_x'], plane['global_y'], '.k', ms=0.1, alpha=0.1)
            ax[0][plane_number].set_xlim(0,pixel_dimensions)
            ax[0][plane_number].set_ylim(0,pixel_dimensions)
            ax[0][plane_number].set_xticks([])
            ax[0][plane_number].set_yticks([])
            ax[0][plane_number].title.set_text(f"Transcripts, z{plane_number}")
            ax[2][plane_number].plot(hist_x*pixel_dimensions/np.amax(hist_x), 'b', lw=0.25)
            ax[2][plane_number].plot(hist_y*pixel_dimensions/np.amax(hist_y), np.arange(1,pixel_dimensions,1), 'g', lw=0.25, alpha=0.5)
            ax[2][plane_number].set_xlim(0,pixel_dimensions)
            ax[2][plane_number].set_ylim(0,pixel_dimensions)
            ax[2][plane_number].set_xticks([])
            ax[2][plane_number].set_yticks([])
            ax[2][plane_number].title.set_text(f"Transcript counts histogram, z{plane_number}")
            ax[2][plane_number].set_xlabel("Detection in x axis")
            ax[2][plane_number].set_ylabel("Detection in y axis")
        else:
            ax[1][plane_number-4].plot(plane['global_x'], plane['global_y'], '.k', ms=0.1, alpha=0.1)
            ax[1][plane_number-4].set_xlim(0,pixel_dimensions)
            ax[1][plane_number-4].set_ylim(0,pixel_dimensions)
            ax[1][plane_number-4].set_xticks([])
            ax[1][plane_number-4].set_yticks([])
            ax[1][plane_number-4].title.set_text(f"Transcripts, z{plane_number}")
            ax[3][plane_number-4].plot(hist_x*pixel_dimensions/np.amax(hist_x), 'b', lw=0.25)
            ax[3][plane_number-4].plot(hist_y*pixel_dimensions/np.amax(hist_y), np.arange(1,pixel_dimensions,1), 'g', lw=0.25, alpha=0.5)
            ax[3][plane_number-4].set_xlim(0,pixel_dimensions)
            ax[3][plane_number-4].set_ylim(0,pixel_dimensions)
            ax[3][plane_number-4].set_xticks([])
            ax[3][plane_number-4].set_yticks([])
            ax[3][plane_number-4].title.set_text(f"Transcripts, z{plane_number}")
            ax[3][plane_number-4].set_xlabel("Detection in x axis")
            ax[3][plane_number-4].set_ylabel("Detection in y axis")

    ax[1][3].plot(transcript_count, 'ok')
    ax[1][3].set_ylim(bottom=0)
    ax[1][3].set_xticks(np.arange(0,7,1).tolist())
    ax[1][3].set_xlabel("z plane")
    ax[1][3].set_ylabel("Transcripts")
    ax[1][3].title.set_text("Transcripts per z plane")

    ax[3][3].plot(np.arange(0, len(np.asarray(cb_list)[:,0])), np.asarray(cb_list)[:,0], 'ob', alpha=0.5)
    ax[3][3].plot(np.arange(0, len(np.asarray(cb_list)[:,0])), np.asarray(cb_list)[:,1], 'og', alpha=0.5)
    ax[3][3].set_ylim(0,1)
    ax[3][3].set_xticks(np.arange(0,7,1).tolist())
    ax[3][3].set_xlabel("z plane")
    ax[3][3].set_ylabel("Checkerboarding")
    ax[3][3].title.set_text("Checkerboarding per z plane")

    fig.tight_layout()
    if outputfolderpath is not None:
        plt.savefig(os.path.join(outputfolderpath,expid+'_cb.jpg'), bbox_inches="tight")
    plt.show()

    return np.asarray(transcript_count), np.asarray(cb_list)

# metrics
#checkboard mean:                   np.mean(cb_list)
#checkboard, most severe:           np.min(cb_list)
#plan6e/plane0 transcript ratio:    transcript_count[6]/transcript_count[0]

def calc_spillover_gene(adata, cbg, verbose=False):
    """
    Calculate spillover gene activity using a given set of parameters.

    Parameters:
    -----------
    anndata : str
        Anndata that has already generate leiden clusters and spatial neighbors.

    cbg : str
        dataframe containing cell-by-gene expression data.

    verbose : bool, optional
        Whether to #print verbose output during the calculation. Default is False.

    Returns:
    --------
    spillover_sum : float
        The total spillover value calculated by summing up all spillover transcripts
        and dividing by the total number of transcripts.

    spillover_worst: float
        The gene that appear to have most spillover transcripts, the spillover of that gene

    df_spillover : DataFrame
        A DataFrame containing spillover information for each gene and leiden clusters. Columns include:
        - 'gene': str
            The gene associated with the spillover data.
        - 'positive_clusters': str
            A space-separated string of sorted positive clusters.
        - 'number_positive_clusters': int
            Number of positive clusters.
        - 'number_positive_cells': int
            Number of cells in positive clusters.
        - 'number_negative_cells': int
            Total number of negative cells (border + non-border).
        - 'number_negative_border_cells': int
            Number of negative cells at the cluster border.
        - 'number_negative_non_border_cells': int
            Number of negative cells not at the cluster border.
        - 't_stat': float
            The calculated t-statistic.
        - 'p_val': float
            The p-value associated with the test.
        - 'fold_change': float
            The calculated fold change value.
        - 'avg_expr_positive': float
            Average expression in positive clusters.
        - 'avg_expr_negative_border': float
            Average expression in negative border cells.
        - 'avg_expr_negative_non_border': float
            Average expression in negative non-border cells.
        - 'excess_transcripts_per_cell': float
            Spillover transcripts per cell.
        - 'spillover_estimate': float
            Estimated spillover value, excess_transcript_per_cell multiply by the number of negative boarder cells.
        - 'trx_count': int
            Total transcript count of the gene.

    df_spillover_gene: DataFrame
        A DataFrame containing spillover information for each gene, column include:
        - 'transcript_counts': int
            Total number of transcripts for the gene.
        - 'avg_expr_positive': float
            Average expression in positive clusters.
        - 'excess_transcripts_per_cell': float
            Spillover transcripts per cell.
        - 'spillover_proportion': float
            Proportion of spillover transcripts compared to total transcripts.
        - 'spillover_estimate': float
            Estimated spillover value.
        - 'number_positive_cells': int
            Number of cells with positive gene expression.
        - 'number_negative_cells': int
            Total number of cells with negative gene expression (border + non-border).
        - 'number_negative_border_cells': int
            Number of cells with negative gene expression at the cluster border.

    Notes:
    ------
    This function calculates spillover gene activity based on the provided inputs.
    It uses a combination of cell type information, gene expression data, and
    graph-based algorithms to perform the calculation.

    Example:
    --------
    n_neighbors = 10
    resolution = 0.1
    radius = 20

    adata = make_AnnData(path_cell_by_gene, path_cell_metadata)
    # remove the blank
    adata = adata[:, ~adata.var.index.str.contains("Blank")
    adata=adata[(adata.obs['total_counts'] >= min_counts) &
             (adata.obs['volume'] >= min_volume)]
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.pp.highly_variable_genes(adata)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=20)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution)
    sq.gr.spatial_neighbors(adata,  radius=radius, coord_type="generic", delaunay=True)

    cbg = pd.read_csv(path_cbg, index_col=0)
    cbg.index = [str(x) for x in cbg.index.tolist()]

    total_spillover, df_spillover, df_spillover_gene = calc_spillover_gene(adata, cbg)

"""
    positive_z_thresh = 1
    min_cells = 10

    # Convert the adjacency matrix to a CSR matrix for efficient row slicing
    adj_matrix = adata.obsp['spatial_connectivities']
    adj_matrix_csr = csr_matrix(adj_matrix)


    index_to_id = {index: id for index, id in enumerate(adata.obs.index)}

    list_ser = []
    for inst_leiden in adata.obs['leiden'].unique():

        inst_cells = adata.obs[adata.obs['leiden'] == inst_leiden].index.tolist()

        ser_exp = cbg.loc[inst_cells,:].mean(axis=0)
        ser_exp.name = inst_leiden

        list_ser.append(ser_exp)

    df_sig = pd.concat(list_ser, axis=1)

    df_sig_z = (df_sig - df_sig.mean(axis=1).values.reshape(-1, 1)) / df_sig.std(axis=1).values.reshape(-1, 1)


    list_ser = []
    for inst_index, inst_gene in enumerate(sorted(df_sig_z.index.tolist())):

        if inst_index % 50 == 0 and verbose == True:
            print(inst_index, inst_gene)

        inst_ser = df_sig_z.loc[inst_gene]

        positive_clusters = inst_ser[inst_ser > positive_z_thresh].index.tolist()

        # #print(inst_gene, len(positive_clusters))

        if len(positive_clusters) > 0:

            adata.obs['marker_positive'] = adata.obs['leiden'].isin(positive_clusters)

            # Get a boolean array where True means the cell is positive for the marker gene
            positive_cells = adata.obs['marker_positive'].values

            # Get a boolean array where True means the cell is negative for the marker gene
            negative_cells = ~positive_cells

            # Convert the boolean arrays to int arrays
            positive_cells_int = positive_cells.astype(int)
            negative_cells_int = negative_cells.astype(int)

            # Multiply the adjacency matrix by the positive_cells array
            # This will give a matrix where each row i contains the number of positive neighbors for cell i
            positive_neighbors = adj_matrix_csr.dot(positive_cells_int)

            # Find the indices of the cells that are negative for the marker gene and have at least one positive neighbor
            indices = np.where((negative_cells_int > 0) & (positive_neighbors > 0))[0]

            # Find the indices of the cells that are negative for the marker gene and do not have any positive neighbors
            non_border_indices = np.where((negative_cells_int > 0) & (positive_neighbors == 0))[0]

            # Create a new column in adata.obs to indicate the cell type for the current marker gene
            adata.obs['spatial_type'] = 'negative_non_border'
            adata.obs.loc[adata.obs['marker_positive'], 'spatial_type'] = 'positive'

            negative_border_indices = [index_to_id[x] for x in indices]

            adata.obs.loc[negative_border_indices, 'spatial_type'] = 'negative_border'

            # all cells
            all_cells_negative_border = adata.obs[adata.obs['spatial_type'] == 'negative_border'].index.tolist()
            all_cells_negative_non_border = adata.obs[adata.obs['spatial_type'] == 'negative_non_border'].index.tolist()
            all_cells_negative = all_cells_negative_border + all_cells_negative_non_border

            # get lists of cells
            cells_positive = adata.obs[adata.obs['spatial_type'] == 'positive'].index.tolist()

            # loop over negative Leiden clusters
            negative_leidens = adata.obs.loc[all_cells_negative, 'leiden'].unique().tolist()

            for inst_leiden in negative_leidens:

                cells_negative_border = adata.obs[
                    (adata.obs['spatial_type'] == 'negative_border') &
                    (adata.obs['leiden'] == inst_leiden)
                ].index.tolist()

                cells_negative_non_border = adata.obs[
                    (adata.obs['spatial_type'] == 'negative_non_border') &
                    (adata.obs['leiden'] == inst_leiden)
                ].index.tolist()


                if len(cells_negative_border) > min_cells and len(cells_negative_non_border) > min_cells:

                    # Calculate the average gene expression level for each group of cells
                    expr_positive = cbg.loc[cells_positive, inst_gene]
                    expr_negative_border = cbg.loc[cells_negative_border, inst_gene]
                    expr_negative_non_border = cbg.loc[cells_negative_non_border, inst_gene]

                    avg_expr_positive = expr_positive.mean()
                    avg_expr_negative_border = expr_negative_border.mean()
                    avg_expr_negative_non_border = expr_negative_non_border.mean()

                    # Perform a T-test between the negative border cells and the negative non-border cells
                    t_stat, p_value = ttest_ind(expr_negative_border, expr_negative_non_border)

                    # Calculate the fold change
                    fold_change = avg_expr_negative_border / avg_expr_negative_non_border

                    # Calculate the excess transcripts per cell
                    excess_transcripts_per_cell = avg_expr_negative_border - avg_expr_negative_non_border

                    # Calculate the spillover estimate
                    spillover_estimate = excess_transcripts_per_cell * len(cells_negative_border)

                    # total number of partitioned transcripts
                    trx_count = cbg[inst_gene].sum()

                    # Append a new row to the DataFrame
                    inst_ser = pd.Series({
                        'gene': inst_gene,
                        'positive_clusters': ' '.join(sorted(positive_clusters)),
                        'number_positive_clusters': len(positive_clusters),
                        'number_positive_cells': len(cells_positive),
                        'number_negative_cells': len(cells_negative_border) + len(cells_negative_non_border),
                        'number_negative_border_cells': len(cells_negative_border),
                        'number_negative_non_border_cells': len(cells_negative_non_border),
                        't_stat': t_stat,
                        'p_val': p_value,
                        'fold_change': fold_change,
                        'avg_expr_positive': avg_expr_positive,
                        'avg_expr_negative_border': avg_expr_negative_border,
                        'avg_expr_negative_non_border': avg_expr_negative_non_border,
                        'excess_transcripts_per_cell': excess_transcripts_per_cell,
                        'spillover_estimate': spillover_estimate,
                        'trx_count': trx_count,
                        # do not calculate proportion since not at gene level
                        # 'spillover_proportion': spillover_estimate/trx_count
                    }, name=inst_gene + '_' + inst_leiden)

                    # only keep spillover
                    if fold_change > 1.0:
                        list_ser.append(inst_ser)

    df_spillover = pd.concat(list_ser, axis=1).T


    gene_trx_partition_total = cbg.sum(axis=0)
    all_genes = sorted(gene_trx_partition_total.index.tolist())

    df_spillover_gene = pd.DataFrame()
    for inst_gene in all_genes:
        gene_spill = df_spillover[df_spillover['gene'] == inst_gene]

        spillover_estimate = gene_spill['spillover_estimate'].sum()
        number_negative_border_cells = gene_spill['number_negative_border_cells'].sum()

        # average across the merged positive cells (all the same value)
        number_positive_cells = gene_spill['number_positive_cells'].mean()
        # sum across the negative clusters
        number_negative_cells = gene_spill['number_negative_cells'].sum()

        number_negative_border_cells = gene_spill['number_negative_border_cells'].sum()

        avg_expr_positive = gene_spill['avg_expr_positive'].mean()

        gene_count = gene_trx_partition_total[inst_gene]

        if number_negative_border_cells > 0:
            excess_transcripts_per_cell = spillover_estimate/number_negative_border_cells
        else:
            excess_transcripts_per_cell = 0

        spillover_proportion = round(spillover_estimate/gene_count, 3)


        df_spillover_gene.loc[inst_gene, 'transcript_counts'] = gene_count

        df_spillover_gene.loc[inst_gene, 'avg_expr_positive'] = avg_expr_positive
        df_spillover_gene.loc[inst_gene, 'excess_transcripts_per_cell'] = excess_transcripts_per_cell
        df_spillover_gene.loc[inst_gene, 'spillover_proportion'] = spillover_proportion
        df_spillover_gene.loc[inst_gene, 'spillover_estimate'] = spillover_estimate

        df_spillover_gene.loc[inst_gene, 'number_positive_cells'] = number_positive_cells
        df_spillover_gene.loc[inst_gene, 'number_negative_cells'] = number_negative_cells
        df_spillover_gene.loc[inst_gene, 'number_negative_border_cells'] = number_negative_border_cells

    # calculate the overal spillover number, total spillover transcripts divide by total transcripts
    leiden_res = adata.obs.filter(like='leiden').columns.tolist()[0]
    cbg_filtered = cbg[cbg.index.isin(adata.obs[leiden_res].index)]
    spillover_sum = df_spillover_gene["spillover_estimate"].sum() / cbg_filtered.iloc[:, 1:].to_numpy().sum()

    cell_by_gene_sorted = cbg_filtered.sort_index(axis=1).sum()
    spillover_worst = (df_spillover_gene["spillover_estimate"] / cell_by_gene_sorted).max()

    return spillover_sum, spillover_worst, df_spillover, df_spillover_gene

def adjust_tr_remove(Zdistribution, thickness, num_fov):
    print('num_fov', num_fov)
    if num_fov == 0:
        return 0
    filtered = Zdistribution[Zdistribution['0']<=np.floor(thickness)]
    print('filtered', filtered)
    adj_trs = filtered['1'].sum()/(np.floor(thickness)+1) * 7
    return adj_trs/num_fov

def adjust_tr_7(total_tr_after_filter, thickness, num_fov):
    print('num_fov', num_fov)
    if num_fov == 0:
        return 0
    adj_trs = total_tr_after_filter/(thickness+1) * 7
    return adj_trs/num_fov

def adjust_tr_6(total_tr_after_filter, thickness, num_fov):
    print('num_fov', num_fov)
    if num_fov == 0:
        return 0
    adj_trs = total_tr_after_filter/thickness * 6
    return adj_trs/num_fov
    
def tr_z0(Zdistribution, num_fov):
    if num_fov == 0:
        return 0
    print('num_fov', num_fov)
    return Zdistribution.loc[0, '1']/num_fov
    
def tr_max(Zdistribution, num_fov):
    pass


def process_dataset(expid, qc_metrics, fov_count_dic, grid_10um, z_counts):
    
    total_trs = qc_metrics['Total counts']
    total_num_fov = len(fov_count_dic.keys())
    filtered_1 = {k:fov_count_dic[k] for k in fov_count_dic.keys() if fov_count_dic[k] > 1000}
    print('filtered_1', filtered_1)
    num_fov_filtered = len(filtered_1.keys())
    print('num_fov_filtered', num_fov_filtered)
    total_tr_after_filter = np.sum(list(filtered_1.values()))
    tr_fov = total_tr_after_filter / num_fov_filtered
    print('z_counts ', z_counts, " !!")
    z_counts_dic = {i:z_counts.loc[i].sum() if type(z_counts.loc[i].sum()) is np.int64 else z_counts.loc[i].sum().values[0] for i in np.unique(z_counts.index)}
    zDistribution = pd.DataFrame.from_dict(z_counts_dic, orient="index").reset_index()
    zDistribution.columns = ['0','1']
    thickness, profile = cal_thick(expid, zDistribution)

    #filtered = {k:filtered_1[k]/(thickness+1)*7 for k in filtered_1.keys()}
    #adjust_tr_fov = np.sum(list(filtered.values())) / len(filtered.keys())
    #zDistribution.to_csv(f"{outputfolderpath}/{expid}_zdis.csv")
    
    tr_fov_unadjusted = total_trs/total_num_fov
    
    tr_fov_z0 = tr_z0(zDistribution, num_fov_filtered)
    
    tr_fov_adjusted_thickness_removed = adjust_tr_remove(zDistribution, thickness, num_fov_filtered)
    
    tr_fov_adjusted_thickness_7 = adjust_tr_7(total_tr_after_filter, thickness, num_fov_filtered)
    
    tr_fov_adjusted_thickness_6 = adjust_tr_6(total_tr_after_filter, thickness, num_fov_filtered)

    countsPerGrid = pd.DataFrame(np.transpose(np.unique(grid_10um, return_counts=True)))
    print('countsPerGrid', countsPerGrid.loc[0:10])
    adjusted_countsPerGrid_7 = countsPerGrid/(thickness+1)*7
    adjusted_countsPerGrid_6 = countsPerGrid/(thickness)*6
    
    transcriptsPer100umMean= countsPerGrid[countsPerGrid[1] > 3][1].mean()
    transcriptsPer100umMedian = countsPerGrid[countsPerGrid[1] > 3][1].median()

    transcriptsPer100umMean_adjust = adjusted_countsPerGrid_6[countsPerGrid[1] > 3][1].mean()
    transcriptsPer100umMedian_adjust = adjusted_countsPerGrid_6[countsPerGrid[1] > 3][1].median()
    
    transcriptsPer100umMean_adjust_7 = adjusted_countsPerGrid_7[countsPerGrid[1] > 3][1].mean()
    transcriptsPer100umMedian_adjust_7 = adjusted_countsPerGrid_7[countsPerGrid[1] > 3][1].median()

    res = {
        'total_tr': total_trs,
        'tr_fov_adjust(portal)' : tr_fov,
        'thickness': thickness,
        'profile' : profile,
        'adjust_tr_fov_z0': tr_fov_z0,
        'adjust_tr_fov(thickness, remove)':  tr_fov_adjusted_thickness_removed,
        'adjust_tr_fov(thickness, 7)':  tr_fov_adjusted_thickness_7,
        'adjust_tr_fov(thickness)':  tr_fov_adjusted_thickness_6,
        'transcriptsPer100umMean': transcriptsPer100umMean,
        'transcriptsPer100umMedian': transcriptsPer100umMedian,
        'transcriptsPer100umMean_thick_adjust': transcriptsPer100umMean_adjust,
        'transcriptsPer100umMedian_thick_adjust': transcriptsPer100umMedian_adjust,
        'transcriptsPer100umMean_thick_adjust_7': transcriptsPer100umMean_adjust_7,
        'transcriptsPer100umMedian_thick_adjust_7': transcriptsPer100umMedian_adjust_7,
    }

    #print(res)
        
    return res


def cal_thick(expid, zDistribution):
    #print(zDistribution)
    zDistribution.columns = ['0','1']
    tissueThickness = None
    currentProfile = None
    
    if zDistribution.iloc[-3:]['1'].mean() > zDistribution.iloc[:3].mean()[1] - zDistribution.iloc[:3].std()[1] or zDistribution.iloc[-2:]['1'].mean() > 0.5 * zDistribution.iloc[:2].mean()['1']:
        tissueThickness = zDistribution['0'].max()
        currentProfile = 'flat'
 
    else:
        def sigmoid(x, L ,x0, k):
            y = L / (1 + np.exp(-k*(x-x0)))
            return y
 
        try:
            guess = [max(zDistribution['1']), np.median(zDistribution['0']), 1]
            params, _ = curve_fit(sigmoid, zDistribution['0'], zDistribution['1'], p0=guess, maxfev=5000)
 
            # Extract the fitted parameters
            L, x0, k = params
 
            y = 0.5*sigmoid(0, L, x0, k)
            tissueThickness = max(min(x0 + np.log((L/(y))-1)/k, zDistribution['0'].max()), 0)
            currentProfile = 'sigmoidal'
        # if sigmoidal fit doesn't converge, try fitting to exponential decay
        except RuntimeError as e:
            pass
    if tissueThickness is None or tissueThickness==0:
        def exponential_decay(x, a, b):
            return a * np.exp(b * x)
 
        initial_guess = [zDistribution['1'][0], -0.1]
 
        params_exp_decay, covariance = curve_fit(exponential_decay, zDistribution['0'], zDistribution['1'], p0=initial_guess)
 
        a_fitted, b_fitted = params_exp_decay
 
        a_fitted, b_fitted
        currentProfile = 'exponential'
        tissueThickness = np.log(0.5) / b_fitted
    plt.figure()
    plt.plot(zDistribution['1'], '.')
    #plt.title(dSetName)
    plt.show()
 
    return (tissueThickness, currentProfile)
