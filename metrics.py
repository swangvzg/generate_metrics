import sys
import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import smart_open

def detected_trs_metrics(csv_file, portion=0.1, fov_size=0.202):
    print('portion: ', portion)
    picked = 0
    num_blanks, line_count = 0, 0
    data_picked = []
    header = []
    blank_fov, trs_fov = {}, {}
    group_by_z_blank, group_by_z_trs = {}, {}
    trs = {}
    trid_count = {}
    barcode_count = {}
    blank_in_cell, trs_in_cell = 0, 0
    blank_not_in_cell, trs_not_in_cell = 0, 0
    global_x_min, global_x_max, global_y_min, global_y_max = -9999,-9999,-9999,-9999
    fov_count_dic = {}
    gridX_10um_list = []
    gridY_10um_list = []
    grid_10um_count = {}

    cell_id_ind = ''
    for line in smart_open.open(csv_file):
        line_count += 1
        if line_count % 1000000 == 0:
            if line_count < 1e7:
                print(datetime.datetime.now(), f" - Processed {line_count} lines ... Picked {picked}")
            else:
                if line_count % 10000000 == 0:
                    print(datetime.datetime.now(), f" - Processed {line_count} lines ... Picked {picked}")
        data = line.strip().split(",")
        if(line_count == 1): # first line
            header = data
            #print(" ".join(data))
            # ,barcode_id,global_x,global_y,global_z,x,y,fov,gene,transcript_id,cell_id
            for i in range(len(data)):
                match data[i]:
                    case 'barcode_id':
                        barcode_id_ind = i
                    case 'global_x':
                        global_x_ind = i
                    case 'global_y':
                        global_y_ind = i
                    case 'global_z':
                        global_z_ind = i
                    case 'x':
                        x_ind = i
                    case 'y':
                        y_ind = i
                    case 'fov':
                        fov_ind = i
                    case 'gene':
                        gene_ind = i
                    case 'transcript_id':
                        transcript_id_ind = i
                    case 'cell_id':
                        cell_id_ind = i
            #print (" ".join([str(n) for n in [barcode_id_ind, global_x_ind, global_y_ind, global_z_ind, x_ind, y_ind, 
            #                fov_ind, gene_ind, transcript_id_ind, cell_id_ind]]))
            if 'barcode_id_ind' not in vars(): barcode_id_ind = 0
        else:
            for i in [global_x_ind, global_y_ind, global_z_ind, x_ind, y_ind]:
                data[i] = float(data[i])
            # global_x_min, global_x_max, global_y_min, global_y_max
            if global_x_min == -9999: 
                global_x_min = data[global_x_ind]
            if global_x_max == -9999: 
                global_x_max = data[global_x_ind]
            if global_y_min == -9999: 
                global_y_min = data[global_y_ind]
            if global_y_max == -9999: 
                global_y_max = data[global_y_ind]
                
            if global_x_min > data[global_x_ind]:
                global_x_min = data[global_x_ind]
            if global_x_max < data[global_x_ind]:
                global_x_max = data[global_x_ind]
            if global_y_min > data[global_y_ind]:
                global_y_min = data[global_y_ind]
            if global_y_max < data[global_y_ind]:
                global_y_max = data[global_y_ind]
                
            # fov count dic
            if data[fov_ind] not in fov_count_dic:
                fov_count_dic[data[fov_ind]] = 0
            fov_count_dic[data[fov_ind]] += 1
            
            # transcript in 10um grid
            gridX_10um = int(float(data[global_x_ind])/10)
            gridY_10um = int(float(data[global_y_ind])/10)
            gridX_10um_list.append(gridX_10um)
            gridY_10um_list.append(gridY_10um)

            # barcode count
            if data[barcode_id_ind] in barcode_count:
                barcode_count[data[barcode_id_ind]] += 1
            else:
                barcode_count[data[barcode_id_ind]] = 1

            if 'Blank' in data[gene_ind]:
                num_blanks += 1
                if cell_id_ind:
                    if data[cell_id_ind] != '-1':
                        blank_in_cell += 1
                    else:
                        blank_not_in_cell +=1;
                if data[fov_ind] in blank_fov:
                    blank_fov[data[fov_ind]] += 1
                else:
                    blank_fov[data[fov_ind]] = 1
                # group by for calculating z_error_rate 
                if data[global_z_ind] in group_by_z_blank:
                    if data[barcode_id_ind] in group_by_z_blank[data[global_z_ind]]:
                        group_by_z_blank[data[global_z_ind]][data[barcode_id_ind]] += 1
                    else:
                        group_by_z_blank[data[global_z_ind]][data[barcode_id_ind]] = 1
                else:
                    group_by_z_blank[data[global_z_ind]] = {}
            else:
                if cell_id_ind:
                    if data[cell_id_ind] != '-1':
                        trs_in_cell += 1
                    else:
                        trs_not_in_cell += 1
                if data[fov_ind] in trs_fov:
                    trs_fov[data[fov_ind]] += 1
                else:
                    trs_fov[data[fov_ind]] = 1
                if data[global_z_ind] in group_by_z_trs:
                    if data[barcode_id_ind] in group_by_z_trs[data[global_z_ind]]:
                        group_by_z_trs[data[global_z_ind]][data[barcode_id_ind]] += 1
                    else:
                        group_by_z_trs[data[global_z_ind]][data[barcode_id_ind]] = 1
                else:
                    group_by_z_trs[data[global_z_ind]] = {}

            # unique Transcripts + blank
            if data[gene_ind] in trs:
                trs[data[gene_ind]] += 1
            else:
                trs[data[gene_ind]] = 0
            if data[transcript_id_ind] in trid_count:
                trid_count[data[transcript_id_ind]] += 1
            else:
                trid_count[data[transcript_id_ind]] = 1

            # keep some data for later use
            #print('10000*portion', 10000*portion)
            if random.randint(0,1000000) <= 1000000*portion:
                #print("Picked ...\n")
                picked += 1
                data_picked.append(data[0:])
        
    print(datetime.datetime.now(), " - Done processing the file")
    
    # Calculations
    qc_metrics = {}
    qc_metrics['Num blank'] = num_blanks
    qc_metrics['Total counts'] = line_count - 1
    qc_metrics['Num transcript counts(exclude blank)'] =   qc_metrics['Total counts'] - num_blanks
    total_transcripts = qc_metrics['Total counts'] - num_blanks
    qc_metrics['blank_div_total'] = f"{(num_blanks / total_transcripts)*100:.2f}%"
    qc_metrics['fov_count'] = len( set(blank_fov.keys()).union(set(trs_fov.keys())) )
    qc_metrics['tissue_area_square_mm'] = fov_size * fov_size * qc_metrics['fov_count']
    #print('qc_metrics[tissue_area_square_mm]: ', qc_metrics['tissue_area_square_mm'])
    #print('fov_size: ', fov_size)
    #print('qc_metrics[fov_count]: ', qc_metrics['fov_count'])
    qc_metrics['transcripts_per_fov'] = total_transcripts/qc_metrics['fov_count']
    qc_metrics['transcripts_per_square_mm'] = total_transcripts/qc_metrics['tissue_area_square_mm']
    qc_metrics['blanks_per_square_mm'] = num_blanks/qc_metrics['tissue_area_square_mm']    
    qc_metrics['unique_coding_barcodes'] = len([i for i in trs.keys() if 'Blank' not in i ])
    qc_metrics['unique_blank_barcodes'] = len([i for i in trs.keys() if 'Blank'  in i ])   
    
    negative_counts_per_blank = num_blanks/ qc_metrics['unique_blank_barcodes']
    positive_counts_per_gene = total_transcripts/qc_metrics['unique_coding_barcodes']
    qc_metrics['FDR_counts'] = f"{(negative_counts_per_blank / positive_counts_per_gene)*100:.2f}%"
    
    # z error rate
    # calcuate the the number of blank transcript per blank barcode per z
    blank_per_uniq_z = np.array([sum(group_by_z_blank[i].values()) for i in group_by_z_blank.keys()]) / qc_metrics['unique_blank_barcodes']
    blank_per_uniq_z = pd.DataFrame(blank_per_uniq_z, index = [i for i in group_by_z_blank.keys()])
    ##print('blank_per_uniq_z')
    ##print(blank_per_uniq_z)
    # calcuate the the number of coding transcript per coding barcode per z
    coding_per_uniq_z = np.array([sum(group_by_z_trs[i].values()) for i in group_by_z_trs.keys()]) /  qc_metrics['unique_coding_barcodes']
    coding_per_uniq_z = pd.DataFrame(coding_per_uniq_z, index = [i for i in group_by_z_trs.keys()])
    ##print('coding_per_uniq_z')
    ##print(coding_per_uniq_z)
    
    z_error_rates = blank_per_uniq_z/coding_per_uniq_z
    z_error_rates.index = [int(float(i)) for i in z_error_rates.index]
    #print('z_error_rates')
    #print(z_error_rates)
    
    ##print(z_error_rates)
    qc_metrics['z_index_min_error'] = z_error_rates.idxmin().values[0]
    qc_metrics['z_index_max_error'] = z_error_rates.idxmax().values[0]
    qc_metrics['z_max_error'] = z_error_rates.max().values[0]
    qc_metrics['z_min_error'] = z_error_rates.min().values[0]
    qc_metrics['z_std_error'] = z_error_rates.std().values[0]

    z_counts = [sum(group_by_z_blank[i].values()) for i in group_by_z_blank.keys()] + [sum(group_by_z_trs[i].values()) for i in group_by_z_trs.keys()]
    zs = [i for i in group_by_z_blank.keys()] + [i for i in group_by_z_trs.keys()]
    #print("zs", zs)
    z_counts = pd.DataFrame(z_counts, index = [int(float(i)) for i in zs])
    #print('z_counts')
    #print(z_counts)
    
    qc_metrics['z_index_min_count'] = z_counts.idxmin().values[0]
    qc_metrics['z_index_max_count'] = z_counts.idxmax().values[0]
    qc_metrics['z_max_count_per_square_mm'] = z_counts.max().values[0]/qc_metrics['tissue_area_square_mm']
    qc_metrics['z_min_count_per_square_mm'] = z_counts.min().values[0]/qc_metrics['tissue_area_square_mm']
    qc_metrics['z_std_count_per_square_mm'] = (z_counts/qc_metrics['tissue_area_square_mm']).std().values[0]
    qc_metrics['z_mean_count_per_square_mm'] = (z_counts/qc_metrics['tissue_area_square_mm']).mean().values[0]
    
    # within cell
    if cell_id_ind:
        qc_metrics['Num coding transcripts within cell'] = trs_in_cell
        qc_metrics['Num coding transcripts not within cell'] = trs_not_in_cell
        qc_metrics['Num blanks within cell'] = blank_in_cell
        qc_metrics['Num blanks not within cell'] = blank_not_in_cell

        #qc_metrics['pct coding transcript within cell'] = f"{(qc_metrics['Num coding transcripts within cell'] / qc_metrics['Num transcript counts(exclude blank)'])*100:.2f}%"
        qc_metrics['pct coding transcript within cell'] = f"{(qc_metrics['Num coding transcripts within cell'] / (qc_metrics['Num coding transcripts within cell'] + qc_metrics['Num coding transcripts not within cell']))*100:.2f}%"

        #qc_metrics['pct blank within cell'] = f"{(qc_metrics['Num blanks within cell'] /qc_metrics['Num blank'])*100:.2f}%"
        #qc_metrics['pct blank within cell'] = f"{(qc_metrics['Num blanks within cell'] /(qc_metrics['Num blanks within cell']  + qc_metrics['Num blanks not within cell']))*100:.2f}%"

        #qc_metrics['ratio of num coding transcript over blank within cell'] = f"{(qc_metrics['Num coding transcripts within cell']/qc_metrics['Num blanks within cell'])*100:.2f}%"

    #### Radial Metrics
    data_picked_df = pd.DataFrame(data_picked)
    data_picked_df.columns = header
    for h in header[2:7]:
        data_picked_df[h] = data_picked_df[h].astype(float)
        
    if 'trans' not in data_picked_df.columns:
        data_picked_df['trans'] = data_picked_df['gene'].apply(lambda x: 0 if x.startswith('Blank') else 1)
    ##print(data_picked_df)

    coding_bar = data_picked_df[data_picked_df['trans'] == 1]
    blank_bar = data_picked_df[data_picked_df['trans'] == 0]
    
    cropWidth = 100  ## M1: 100; M1.7: 75 pixel
    fovSize = [2048, 2048] ## M1.7: [2960, 2960]
    if fov_size == 0.296: # M1.7
        cropWidth = 75
        fovSize = [2960, 2960]
    
    maxRadius = 0.5*np.min(fovSize)-cropWidth
    coding_bar['r'] = np.sqrt((coding_bar['x'] - fovSize[0]*0.5)**2 + (coding_bar['y'] - fovSize[0]*0.5)**2)
    radialHistogramCoding = plt.hist(coding_bar[coding_bar['r'] < maxRadius]['r'], bins=np.arange(0, maxRadius, 50))
    qc_metrics['radial_deviation_coding'] = ((radialHistogramCoding[1][-1] * radialHistogramCoding[0][-1])*0.5/50 - np.sum(radialHistogramCoding[0]))/np.sum(radialHistogramCoding[0])

    blank_bar['r'] = np.sqrt((blank_bar['x'] - fovSize[0]*0.5)**2 + (blank_bar['y'] - fovSize[0]*0.5)**2)
    radialHistogramBlank = plt.hist(blank_bar[blank_bar['r'] < maxRadius]['r'], bins=np.arange(0, maxRadius, 50))
    qc_metrics['radial_deviation_blank'] = ((radialHistogramBlank[1][-1] * radialHistogramBlank[0][-1])*0.5/50 - np.sum(radialHistogramBlank[0]))/np.sum(radialHistogramBlank[0])
    
    # 10um grid
    grid_10um = np.array(gridX_10um_list) * max(gridY_10um_list) + gridY_10um_list
    
    return qc_metrics, trs, trid_count, barcode_count, data_picked_df, z_error_rates, z_counts, global_x_min, global_x_max, global_y_min, global_y_max, fov_count_dic, grid_10um

if __name__ == '__main__':
    qc_ms, trs, trid_count, barcode_count, data_picked_df, z_error_rates, z_counts, global_x_min, global_x_max, global_y_min, global_y_max = detected_trs_metrics(sys.argv[1])
    for k in qc_ms.keys():
        print(k)
        print(qc_ms[k])
