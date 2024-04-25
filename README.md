# generate_metrics

## usage: 
<pre>
    python  generate_metrics.py \
    --path_detected_trs s3://vz-analyzed-merfish/202403191513_4MsBrain-FF-1kDM0824-M1Ctrl-TZ_VMSC15302/PartitionTranscripts/region_3/detected_transcripts.csv  \
    --path_cell_by_gene s3://vz-analyzed-merfish/202403191513_4MsBrain-FF-1kDM0824-M1Ctrl-TZ_VMSC15302/PartitionTranscripts/region_3/cell_by_gene.csv \ 
    --path_cell_metadata s3://vz-analyzed-merfish/202403191513_4MsBrain-FF-1kDM0824-M1Ctrl-TZ_VMSC15302/DeriveEntityMetadataTask/region_3/cell_metadata.csv \
    --path_code_book s3://vz-analyzed-merfish/202403191513_4MsBrain-FF-1kDM0824-M1Ctrl-TZ_VMSC15302/codebook_0_DM0824_30bitVerification_DM0824.csv  \
    --exp_id 202403191513_4MsBrain-FF-1kDM0824-M1Ctrl-TZ_VMSC15302 \
    --output_dir test \
    --exp_data Mouse_brain_exp.csv \
    --portion 0.1 \
    --fov_size 0.202  

</pre>
