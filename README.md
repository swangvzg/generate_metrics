# generate_metrics

## usage: 
<pre>
usage: calc_metrics.py [-h] --bucket_name BUCKET_NAME --exp_name EXP_NAME [--region REGION] [--expression EXPRESSION] [--skip_spatial SKIP_SPATIAL] [--plot PLOT] [--outputfolderpath OUTPUTFOLDERPATH]

options:
  -h, --help            show this help message and exit
  --bucket_name BUCKET_NAME
                        Bucket name for S3 storage, ie. vz-analyzed-merfish
  --exp_name EXP_NAME   experiment name, ie, 202404272011_HuBrCa-FFPE-HQ1353023B-BM1506-3xV1-30-BY_VMSC14402
  --region REGION       specify the region, ie R3, if don't specify, will take all regions
  --expression EXPRESSION
                        expression tabel
  --skip_spatial SKIP_SPATIAL
                        Skip spatial analysis like umap, leiden
  --plot PLOT           Generate plots or not
  --outputfolderpath OUTPUTFOLDERPATH
                        Output folder path
</pre>

## example:
<pre>
calc_metrics.py --bucket_name vz-analyzed-merfish \
--exp_name 202410121816_241003JHMSTM000J6M-MsTMA-DM1922-V1-TZ_VMSC11802 \
--outputfolderpath output \
--skip_spatial False \
--plot True \
--expression /efs/swang/expression/Mouse_brain_exp.csv \
--region R1 \
2>&1 | tee 202410121816_241003JHMSTM000J6M-MsTMA-DM1922-V1-TZ_VMSC11802_R1.log
</pre>
