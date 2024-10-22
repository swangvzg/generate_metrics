#!/usr/bin/env python

import sys
import pandas as pd
from scipy.optimize import curve_fit
import os
import boto3
import pickle
from tqdm import tqdm
import subprocess
import re
import json
import s3fs

#sys.path.insert(0, "/efs/swang/M17_evaludation_v2/scripts")
import utils

def get_all_s3_objects(s3, **base_kwargs):
    continuation_token = None
    while True:
        list_kwargs = dict(MaxKeys=1000, **base_kwargs)
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**list_kwargs)
        yield from response.get('Contents', [])
        if not response.get('IsTruncated'):  # At the end of the list?
            break
        continuation_token = response.get('NextContinuationToken')


def find_files_in_s3(bucket_name = 'vz-analyzed-merfish', exp_name = "202404272011_HuBrCa-FFPE-HQ1353023B-BM1506-3xV1-30-BY_VMSC14402", res = {}, codebook_dic={} ):
    print(exp_name)  
    
    for file in get_all_s3_objects(boto3.client('s3'), Bucket=bucket_name, Prefix=exp_name):
        #print(file['Key'])
        # get detected_transcripts.csv, cell_by_gene.csv, cell_metadata.csv, codebook
        vals = file['Key'].split("/")
        region_index = 2 if bucket_name == 'vz-analyzed-merfish' else 1
        if len(vals) > 2 and  "REGION" in vals[region_index].upper() and len(vals[region_index]) < 20:
            #print(file['Key'])
            uid = ":".join([vals[0], vals[region_index]])
            if uid not in res:
                res[uid] = ["", "", ""]
            if vals[-1] == 'detected_transcripts.csv':
                res[uid][0] = bucket_name + "/" + file['Key'] if res[uid][0] == "" else res[uid][0]
            if vals[-1] == 'cell_by_gene.csv':
                res[uid][1] = bucket_name + "/" +file['Key'] if res[uid][1] == "" else res[uid][1]
            #if vals[-1] == 'cell_metadata.csv' and vals[-3] == 'DeriveEntityMetadataTask':
            if vals[-1] == 'cell_metadata.csv':
                if bucket_name == 'vz-analyzed-merfish':
                    if vals[-3] == 'DeriveEntityMetadataTask':
                        res[uid][2] = bucket_name + "/" +file['Key'] if res[uid][2] == "" else res[uid][2]
                else:
                    res[uid][2] = bucket_name + "/" +file['Key'] if res[uid][2] == "" else res[uid][2]
        if 'codebook' in vals[-1]:
            codebook_dic[vals[0]] = bucket_name + "/" +file['Key'] if vals[0] not in codebook_dic else codebook_dic[vals[0]]
    return res, codebook_dic

def get_required_files(bucket_name = 'vz-analyzed-merfish', exp_name = "202404272011_HuBrCa-FFPE-HQ1353023B-BM1506-3xV1-30-BY_VMSC14402"):
    res, codebook_dic = find_files_in_s3(bucket_name = bucket_name, exp_name = exp_name)
    missing = []
    for k in res:
        if any([i == '' for i in res[k]]):
            missing.append(k)
    
    if len(missing) > 0:
        print(f"Missing files: {missing}, try different bucket...")
        if bucket_name == 'vz-analyzed-merfish':
            bucket_name = 'vz-output-merfish'
        else:
            bucket_name = 'vz-analyzed-merfish'
        res, codebook_dic = find_files_in_s3(bucket_name = bucket_name, exp_name = exp_name, res = res, codebook_dic = codebook_dic)

    return res, codebook_dic
    

def load_json(path):
    if path.startswith("s3"):
        fs = s3fs.S3FileSystem()
        with fs.open(path) as f:
            return json.load(f)
    else:
        with open(path) as f:
            return json.load(f)
def get_M_version(run_name):
    json_file = f"s3://vz-analyzed-merfish/{run_name}/microscope_parameters.json"
    j = load_json(json_file)
    if 'image_dimensions' in j and  j['image_dimensions'] == [2960, 2960]:
        return 'M1.7'
    return 'M1'

# check seg version
def printver(sample, region):
    path_json = f"s3://vz-analyzed-merfish/{sample}/PrepareSegmentation/tasks/task.json"
    task = load_json(path_json)
    print(task)
    
    path_de = f"s3://vz-analyzed-merfish/{sample}/PartitionTranscripts/{region}/detected_transcripts.csv"
    try:
        de = pd.read_csv(path_de, nrows=3)
        if 'transcript_score' in de.columns:
            de_ver = "decode11"
        else:
            de_ver = "decode1"
    except:
        print(f"{sample} decoder version not found")
        de_ver = "na"
    algo = task['algorithm']['name']
    seed_channel, entity_fill_channel, nuclear_channel = 'na', 'na', 'na'
    if 'seed_channel' in task['algorithm']['arguments']:
        seed_channel = task['algorithm']['arguments']['seed_channel']
    if 'entity_fill_channel' in task['algorithm']['arguments']:
        entity_fill_channel = task['algorithm']['arguments']['entity_fill_channel']
    if 'nuclear_channel' in task['algorithm']['arguments']:
        nuclear_channel = task['algorithm']['arguments']['nuclear_channel']
    
    return (algo, seed_channel, entity_fill_channel,  nuclear_channel, de_ver)

def get_seg_version(sample, region):
    try:
        task = printver(sample, region)
    except:
        print("Except: ", sample)
        task = ("cellpose_after", 'na',  "na", "na", "na")
    return task

def get_meta_info(sample):
    json_file = f"s3://vz-analyzed-merfish/{sample}/experiment.json"
    j = load_json(json_file)
    #print(j)
    soft_ver = j['softwareVersion']
    exp_id= j['experimentId']
    exp_name = j['experimentName']
    machine_id = exp_id.split("_")[-1]
    panel_name = j['panelName']
    print({'softwareVersion': soft_ver, 'machineId': machine_id,'experimentName': exp_name, 'experimentId': exp_id, 'panelName': panel_name})
    return {'softwareVersion': soft_ver,'machineId': machine_id,  'experimentName': exp_name, 'experimentId': exp_id, 'panelName': panel_name}


def calc_metrics(res, codebook_dic, run_regions, exp_name, exp_df, outputfolderpath, skip_spatial, plot):
    print(exp_name)
    meta_info = get_meta_info(exp_name)
    regions = "-".join(run_regions[exp_name])
    output_csv = f"./{outputfolderpath}/{exp_name}_{regions}_metrics.csv"
    #if os.path.exists(output_csv):
        #print(f"Skip, {output_csv} exists!")
        #return
    output_df = []
    for region in run_regions[exp_name]:
        #if '4' not in region:
        #    continue
        k = ":".join([exp_name, region])
        data = {}
        data['experiment'] = exp_name
        data['region'] = region
        # check if it is M1 or M1.7; and segmentation
        M_ver = get_M_version(exp_name)
        data['M_version'] = M_ver
        seg_version = get_seg_version(exp_name, region)
        print('seg_version,', seg_version)
        seg_version_info = ['Seg_algo', 'seed_channel', 'entity_fill_channel', 'nuclear_channel', 'decoder_version']
        for i in range(4):
            data[seg_version_info[i]] = seg_version[i]

        # run the metrics calculation
        fov_size = 0.202 # M1
        if M_ver == 'M1.7': fov_size = 0.298

        exp = exp_df # expression dataframe

        detected_tr_csv, cell_by_gene, cell_metadata = [f"s3://{i}" if i != "" else "" for i in res[k] ]
        print("cell_metadata: ", cell_metadata)
        codebook_csv = f"s3://{codebook_dic[exp_name]}" if codebook_dic[exp_name] != "" else ""
        print("\n".join([detected_tr_csv, cell_by_gene, cell_metadata, codebook_csv]))
        if detected_tr_csv == '':
            print(f"Skip {k}: No detected_tr_csv\n")
            continue
        #try:    
        metrics = utils.calc_qc_metrics(
            detected_tr_csv, 
            cell_by_gene, 
            cell_metadata,
            fov_size = fov_size,
            exp_table = exp,
            path_to_codebook = codebook_csv,
            outputfolderpath=outputfolderpath,
            expid=k,
            portion=0.1,
            plot=plot,
            skip_spatial=skip_spatial)

        metrics = {**meta_info, **data, **metrics}
        output_df.append(pd.DataFrame.from_dict(metrics, orient="index"))
        #except Exception as e:
        #    print("calc_qc_metrics exceptions, ", e)

    if len(output_df) > 0:
        print("generate csv for ", exp_name,  output_csv)
        pd.concat(output_df, axis=1).to_csv(output_csv)


# In[ ]:

def get_run_regions(res):
    run_regions = {}
    for k in res.keys():
        run_name, region = k.split(":")
        if run_name not in run_regions:
            run_regions[run_name] = []
        run_regions[run_name].append(region)
    return run_regions

if __name__ == '__main__':
    import argparse
    import datetime
    from distutils.util import strtobool
    parser = argparse.ArgumentParser(description='Calcualte metrics for detected transcripts for all runs in the bucket')
    parser.add_argument('--bucket_name', required=True, help='Bucket name for S3 storage, ie. vz-analyzed-merfish')
    parser.add_argument('--exp_name', required=True, help='experiment name, ie, 202404272011_HuBrCa-FFPE-HQ1353023B-BM1506-3xV1-30-BY_VMSC14402')
    parser.add_argument('--region', required=False, help="specify the region, ie R3, if don't specify, will take all regions", default=None)
    parser.add_argument('--expression', required=False, help='expression tabel', default=None)
    parser.add_argument('--skip_spatial', required=False, help='Skip spatial analysis like umap, leiden', default = True, type=lambda x: bool(strtobool(x)) )
    parser.add_argument('--plot', required=False, help='Generate plots or not', default=False, type=lambda x: bool(strtobool(x)) )
    parser.add_argument('--outputfolderpath', required=False, help='Output folder path', default="output")
    
    args = parser.parse_args()
    
    bucket_name = args.bucket_name
    region = args.region
    exp_name = args.exp_name
    expression = args.expression
    skip_spatial = args.skip_spatial
    plot = args.plot
    outputfolderpath = args.outputfolderpath or './output'
    print("output folder ", outputfolderpath, args.outputfolderpath)

    exp_df = None
    if expression is not None:
        exp_df = pd.read_csv(expression)

    if not os.path.exists(outputfolderpath):
        os.makedirs(outputfolderpath)
    print("current time:- ", datetime.datetime.now())
    res, codebook_dic = get_required_files(bucket_name = bucket_name, exp_name = exp_name)
    print('res', res)
    print('codebook_dic', codebook_dic)
    run_regions = get_run_regions(res)
    if region is not None:
        picked_regions = [r for r in run_regions[exp_name] if region in r]
        run_regions[exp_name] = picked_regions
        print("Only get metrics from region: ", picked_regions)
    print('run_regions', run_regions)
    calc_metrics(res, codebook_dic, run_regions, exp_name, exp_df, outputfolderpath, skip_spatial, plot)
    print("current time:- ", datetime.datetime.now())




