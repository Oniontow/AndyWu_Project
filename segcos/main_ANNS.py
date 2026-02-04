import numpy as np    # Importing NumPy for numerical operations.
import argparse       # Importing argparse for command-line option and argument parsing.
from tqdm import tqdm # Importing tqdm for progress bars.
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
import math
plt.switch_backend('Agg')
def process_args():
    parser = argparse.ArgumentParser()  # Create a new argument parser.
    
    parser.add_argument("--dataset", default="deep1M", help="The dataset name", choices=['DEEP1B', 'Last.fm', 'COCO-I2I', 'COCO-T2I', 'NYTimes', 'glove-25', 'glove-50', 'glove-100', 'glove-200'])  # Name of the dataset
    parser.add_argument("--similarity",  help="", choices=['L1norm', 'cosine', 'segmented_cosine', 'LSH'])  # Name of the dataset.
    parser.add_argument("--query_num", type=int, default=1,  help="")  
    parser.add_argument("--total_query_num", type=int, default=0,  help="")  
    parser.add_argument('--dimension', '-d', type=int, default=512, help='dimension of extracted feature')
    parser.add_argument('--coupled_dimension', '-cd', type=int, default=1)
    parser.add_argument('--param', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta',  type=float, default=0)
    parser.add_argument("--option",  help="", choices=['CSI', 'JI', 'TLE', 'HD', 'Seg-Cos_Real_Float','Seg-Cos_Float', 'Seg-Cos_QuantAng', 'Seg-Cos_QuantAngMag', 'Seg-Cos_Fixed', 'Seg-Cos_TCAM', 'Seg-Cos_TCAM_Bit'])
    parser.add_argument("--minimum_clamp", action='store_true') 
    parser.add_argument("--bound", choices=['upper', 'lower', 'complementary'], default=None)
    parser.add_argument("--angle", choices=['ln', 'arccos'], default=None)
    parser.add_argument("--normalization", action='store_true')
    parser.add_argument("--factor", type=float)
    parser.add_argument("--complete", action='store_true', help="Complete the normalization")

    parser.add_argument("--topk", type=int, default=1000, help="The topk for distance calculation")
    
    #Hyper parameter of Seg-Cos
    parser.add_argument("--N", type=int, default=64,  help="")  
    parser.add_argument("--clip_value", type=float, default=0.3, help="The clip value for standardization")
    parser.add_argument("--tolerance_scale", type=int, default=10, help="The tolerance scale for distance calculation")
    #Main Function
    parser.add_argument("--find_best_recall", action='store_true', help="Find the best recall")

    #Differnet similarity of Seg-Cos
    parser.add_argument("--positive_similarity", action='store_true', help="Positive only")
    parser.add_argument("--positive_predict", action='store_true', help="Positive only")
    parser.add_argument("--tolerance_remap", action='store_true', help="Positive only")
    parser.add_argument("--tolerance_remap_target", default="mean", help="")
    parser.add_argument("--tolerance_function", default="linear", help="The dataset name")  # Name of the dataset.
    
    #Experiment setting
    parser.add_argument("--no_store_result", action='store_true', help="no store result")
    parser.add_argument("--batch_size", type=int, default=5000000, help="")

    #Subset
    parser.add_argument("--num_subset", type=int, default=50000, help="")
    

    return parser.parse_args()  # Parse the arguments from the command line.


class SupportSetDataset():
    def __init__(self, support_set):
        self.support_set = torch.tensor(support_set, dtype=torch.float32)

    def __len__(self):
        return len(self.support_set)

    def __getitem__(self, idx):
        return self.support_set[idx]

def do_compute_cosine(support_loader, xq, topk=1000):
    device = xq.device  
    # gamma = -0.5
    # xq_norm = torch.sqrt(torch.sum(xq**2, dim=1)) + 1e-12
    # xq = torch.sign(xq) * torch.abs(xq) ** gamma
    all_similarities = []

    for xb in tqdm(support_loader):
        xb = xb.to(device)
        # xb = torch.sign(xb) * torch.abs(xb) ** gamma
        xb_norm = torch.sqrt(torch.sum(xb**2, dim=1)) + 1e-12

        similarity = torch.matmul(xq, xb.T) / (xq_norm[:, None] * xb_norm[None, :])
        similarity = torch.clamp(similarity, min=-1.0, max=1.0)
        similarity = -1 * torch.acos(similarity) / math.pi * 180
        # similarity = -1*torch.sqrt(torch.sum((xq[:,None,:] - xb[None,:,:])**2, dim=2))
        all_similarities.append(similarity.detach().cpu())
   
    all_similarities = torch.cat(all_similarities, dim=1)

    topk_values, topk_indices = torch.topk(all_similarities, k=topk, dim=1)
    # plt.figure(figsize=(6,4))
    # plt.hist(-1*all_similarities[0].flatten().cpu().numpy(), bins=100, color='steelblue', edgecolor='black')
    # plt.title(f"Angular Difference Distribution")
    # plt.xlabel("Value (Degree)")
    # plt.ylabel("Frequency")
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.savefig(f"similarity.png")
    # print(a)
    all_similarities = all_similarities.flatten()

    return topk_indices, torch.mean(all_similarities), torch.std(all_similarities)

def do_compute_L1norm(support_loader, xq, topk=1000, clamp_ratio = 0.0):
    device = xq.device  

    all_similarities = []
    xq_norm = torch.sqrt(torch.sum(xq**2, dim=1))
    if(args.normalization):
        xq = xq / xq_norm[:, None]

    if(args.normalization):
        support_data = support_loader.dataset.support_set
        support_norm = torch.sqrt(torch.sum(support_data**2, dim=1))
        normalized_support_set = support_data / support_norm[:, None]
        max_value = torch.max(normalized_support_set).item()
        min_value = torch.min(normalized_support_set).item()
    else:
        max_value = torch.max(support_loader.dataset.support_set).item()
        min_value = torch.min(support_loader.dataset.support_set).item()
    range_value = (max_value - min_value)

    mean_value = (max_value + min_value) / 2
    max_value = mean_value + (1 - clamp_ratio) * range_value / 2
    min_value = mean_value - (1 - clamp_ratio) * range_value / 2
    range_value = (1 - clamp_ratio) * range_value
    # xq = torch.round(((xq - min_value) / range_value) * (args.N - 1))
    xq = torch.clamp(torch.floor(((xq - min_value) / range_value) * args.N), 0, args.N-1)
    # xq = torch.clamp(xq, min=min_value, max=max_value)

    for xb in tqdm(support_loader):
        xb = xb.to(device)
        if(args.normalization):
            xb_norm = torch.sqrt(torch.sum(xb**2, dim=1))
            xb = xb / xb_norm[:, None]
        # xb = torch.round(((xb - min_value) / range_value) * (args.N - 1))
        xb = torch.clamp(torch.floor(((xb - min_value) / range_value) * args.N), 0, args.N-1)
        # xb = torch.clamp(xb, min=min_value, max=max_value)
        # similarity = -1*torch.max(torch.abs(xq[:,None,:] - xb[None,:,:]), dim=2)[0]
        similarity = -1*torch.mean(torch.abs(xq[:,None,:] - xb[None,:,:]), dim=2)
        all_similarities.append(similarity.detach().cpu())
   
    all_similarities = torch.cat(all_similarities, dim=1)

    topk_values, topk_indices = torch.topk(all_similarities, k=topk, dim=1)
    all_similarities = all_similarities.flatten()

    return topk_indices, torch.mean(all_similarities), torch.mean(all_similarities), torch.std(all_similarities)

def do_compute_lsh(support_loader, xq, L, hashing_matrix, topk=1000):
    device = xq.device  
    xq_norm = torch.sqrt(torch.sum(xq**2, dim=1))

    all_differences  = []
    all_similarities = []

    if(args.normalization):
        xq = xq / xq_norm[:, None]
        # for i in range(xq.shape[1]):
        #     plt.figure(figsize=(6,4))
        #     plt.hist(xq[:,i].flatten().cpu().numpy(), bins=100, color='steelblue', edgecolor='black')
        #     plt.title(f"Distribution of Normalized xq Values - Dimension {i}")
        #     plt.xlabel("Value")
        #     plt.ylabel("Frequency")
        #     plt.grid(True, linestyle='--', alpha=0.6)
        #     plt.tight_layout()
        #     plt.savefig(f"normalized_xq_distribution_dimension_{i}.png")
        # print(a)

        # print(torch.mean(torch.abs(xq)))
        # print(torch.mean(xq[xq>0]))
        # print(torch.mean(xq[xq<0]))
        # print(a)

    for xb in tqdm(support_loader):
        xb = xb.to(device)
        xb_norm = torch.sqrt(torch.sum(xb**2, dim=1))

        similarity_truth = torch.matmul(xq, xb.T) / (xq_norm[:, None] * xb_norm[None, :])
        similarity_truth = torch.clamp(similarity_truth, -1, 1)
        angle_truth      = -1 * torch.acos(similarity_truth) / math.pi * 180
        if(args.normalization):
            xb = xb / xb_norm[:, None]
        hashed_query_vector = torch.matmul(xq, hashing_matrix)
        hashed_key_memory   = torch.matmul(xb, hashing_matrix)
        ###
        hashed_query_vector = (hashed_query_vector > 0).float()
        hashed_key_memory   = (hashed_key_memory   > 0).float()
        similarity          = torch.cos(torch.mean(torch.abs(hashed_query_vector[:, None, :] - hashed_key_memory[None, :, :]), dim=2) * math.pi)
        similarity          = torch.clamp(similarity, -1, 1)
        similarity          = -1 * torch.acos(similarity) / math.pi * 180
        
        
        # similarity = -1*torch.sqrt(torch.sum((hashed_query_vector[:,None,:] - hashed_key_memory[None,:,:])**2, dim=2))
        all_differences.append((angle_truth - similarity).detach().cpu())
        all_similarities.append(similarity.detach().cpu())
   
    all_differences = torch.cat(all_differences, dim=1)
    all_similarities = torch.cat(all_similarities, dim=1)

    topk_values, topk_indices = torch.topk(all_similarities, k=topk, dim=1)
    all_differences = all_differences.flatten()
    all_similarities = all_similarities.flatten()

    return topk_indices, torch.mean(all_similarities).item(), torch.mean(all_differences).item(), torch.std(all_differences).item()

global correlation_array
global gap_array
correlation_array = []
gap_array         = []
def do_compute_segmented_cosine(support_loader, query_vector, topk=1000, scale=None, minimum=None):
    device           = query_vector.device 
    query_vector     = query_vector[:, :args.dimension] 
    query_vector_mag = torch.sqrt(torch.sum(query_vector**2, dim=1)) + 1e-12

    query_vector_circular = torch.cat((query_vector, query_vector[:,:args.coupled_dimension-1]), dim=1)
    idx                   = torch.arange(args.coupled_dimension).unsqueeze(0) + torch.arange(args.dimension).unsqueeze(1)
    query_segment         = query_vector_circular[:,idx]
    query_segment_mag     = torch.sqrt(torch.sum(query_segment**2, dim=2)) + 1e-12
    
    if(args.option == "Seg-Cos_TCAM_Bit"):
        # if(args.N == 2):
        #     query_segment_mag_      = query_segment[:,:,0]
        # elif(args.N == 4):
        #     query_segment_mag_p1_p1 = torch.abs(query_segment[:, :, 0] - query_segment[:, :, 1]) / math.sqrt(2)
        #     query_segment_mag_p1_n1 = torch.abs(query_segment[:, :, 0] + query_segment[:, :, 1]) / math.sqrt(2)
        #     query_segment_mag_      = torch.cat((query_segment_mag_p1_n1.unsqueeze(2), query_segment_mag_p1_p1.unsqueeze(2)), dim=2)
        plane_index = torch.arange(args.N//2).to('cuda')
        plane_angle = -math.pi / args.N - plane_index * (2*math.pi / args.N)
        normal_matrix = torch.stack((torch.sin(plane_angle), -torch.cos(plane_angle)), dim=1)
        query_segment_mag_ = torch.abs(torch.matmul(query_segment, normal_matrix.T))
    all_differences  = []
    all_similarities = []
    all_angles       = []
    all_weight_mean      = []
    all_similarity_truth = []

    # data_array = []

    for support_vector in tqdm(support_loader):
        support_vector     = support_vector.to(device)
        support_vector     = support_vector[:, :args.dimension]
        support_vector_mag = torch.sqrt(torch.sum(support_vector**2, dim=1)) + 1e-12

        similarity_truth = torch.matmul(query_vector, support_vector.T) / (query_vector_mag[:, None] * support_vector_mag[None, :])
        similarity_truth = torch.clamp(similarity_truth, -1, 1)
        angle_truth      = -1 * torch.acos(similarity_truth) / math.pi * 180

        support_vector_circular = torch.cat((support_vector, support_vector[:,:args.coupled_dimension-1]), dim=1)
        support_segment         = support_vector_circular[:,idx]
        support_segment_mag     = torch.sqrt(torch.sum(support_segment**2, dim=2)) + 1e-12

        if(args.option == "Seg-Cos_TCAM_Bit"):
            # if(args.N == 2):
            #     support_segment_mag_      = support_segment[:,:,0]
            # elif(args.N == 4):
            #     support_segment_mag_p1_p1 = torch.abs(support_segment[:, :, 0] - support_segment[:, :, 1]) / math.sqrt(2) + 1e-12
            #     support_segment_mag_p1_n1 = torch.abs(support_segment[:, :, 0] + support_segment[:, :, 1]) / math.sqrt(2) + 1e-12
            #     support_segment_mag_      = torch.cat((support_segment_mag_p1_n1.unsqueeze(2), support_segment_mag_p1_p1.unsqueeze(2)), dim=2)
            plane_index = torch.arange(args.N//2).to('cuda')
            plane_angle = -math.pi / args.N - plane_index * (2*math.pi / args.N)
            normal_matrix = torch.stack((torch.sin(plane_angle), -torch.cos(plane_angle)), dim=1)
            support_segment_mag_ = torch.abs(torch.matmul(support_segment, normal_matrix.T))
        weight_support = math.sqrt(args.dimension / args.coupled_dimension) * (support_segment_mag / support_vector_mag[:,None])
        weight_query   = math.sqrt(args.dimension / args.coupled_dimension) * (  query_segment_mag /   query_vector_mag[:,None])
        # if(args.normalization):
        #     weight_support = weight_support / torch.mean(weight_support, dim=1)[:, None]
        #     weight_query   = weight_query   / torch.mean(weight_query, dim=1)[:, None]
        # weight_support = math.sqrt(args.dimension / args.coupled_dimension) * weight_support
        # weight_query   = math.sqrt(args.dimension / args.coupled_dimension) * weight_query
        # mean_weight_support = torch.mean(torch.log(weight_support), dim=1)
        # mean_weight_support_numpy = mean_weight_support.detach().cpu().numpy()

        # # Plotting the histogram
        # plt.figure(figsize=(8, 6))
        # plt.hist(mean_weight_support_numpy, bins=100, color='blue', alpha=0.7)
        # plt.title('Histogram of Mean Log-Transformed Weight Support')
        # plt.xlabel('Mean Log-Transformed Weight Support')
        # plt.ylabel('Frequency')
        # plt.grid(True)
        # plt.savefig('a.png')
        # weight = args.dimension*(support_segment_mag[None, :, :] * query_segment_mag[:, None, :]) / (args.coupled_dimension*(support_vector_mag[None,:,None] * query_vector_mag[:,None,None]))
        # data_array.append((weight * (1 - segmented_similarity))[(1-similarity_truth) < 0.25].detach().cpu())
        # data_array.append((weight * (1 - segmented_similarity))[((1-similarity_truth) > 0.25) & ((1-similarity_truth) < 0.50)].detach().cpu())
        # data_array.append((weight * (1 - segmented_similarity))[((1-similarity_truth) > 0.50) and ((1-similarity_truth) < 0.75)].detach().cpu())
        # data_array.append((weight * (1 - segmented_similarity))[((1-similarity_truth) > 0.75) and ((1-similarity_truth) < 1.00)].detach().cpu())
        # data_array.append((weight * (1 - segmented_similarity))[((1-similarity_truth) > 1.00) and ((1-similarity_truth) < 1.25)].detach().cpu())
        # data_array.append((weight * (1 - segmented_similarity))[((1-similarity_truth) > 1.25) and ((1-similarity_truth) < 1.50)].detach().cpu())
        # data_array.append((weight * (1 - segmented_similarity))[((1-similarity_truth) > 1.50) and ((1-similarity_truth) < 1.75)].detach().cpu())
        # data_array.append((weight * (1 - segmented_similarity))[((1-similarity_truth) > 1.75) and ((1-similarity_truth) < 2.00)].detach().cpu())
        # if((args.option == "CSI") or (args.option == "JI") or (args.option == "TLE") or (args.option == "HD") or (args.option == "Seg-Cos_Float")):
        segmented_similarity = torch.sum(query_segment[:, None, :, :] * support_segment[None, :, :, :], dim=3) / (query_segment_mag[:, None, :] * support_segment_mag[None, :, :])
        segmented_similarity = torch.clamp(segmented_similarity, -1+1e-7, 1-1e-7)
        # torch.set_printoptions(precision=20)
        # print(torch.min(segmented_similarity), torch.max(segmented_similarity))
        weight               = weight_support[None, :, :] * weight_query[:, None, :]
        
        # Cauchy-Schwarz Inequality ##########################################################################################################################################################################################################################################
        if(args.option == "CSI"):
            segment_lower_estimate = weight * (1 - segmented_similarity)
            segment_upper_estimate = weight * (1 + segmented_similarity)
        # Jensen's Inequality ################################################################################################################################################################################################################################################
        elif(args.option == "JI"):
            segment_lower_estimate = torch.log(weight) + torch.log(1 - segmented_similarity)
            segment_upper_estimate = torch.log(weight) + torch.log(1 + segmented_similarity)
        # Taylor Expansion  ##################################################################################################################################################################################################################################################
        elif(args.option == "TLE"):
            t_lower    = 2*math.atan(1/(scale))
            bias_lower = ((0-t_lower)*scale) + math.log(1-math.cos(t_lower))
            segment_lower_estimate = torch.log(weight) + (scale*torch.acos(segmented_similarity) + bias_lower)
            t_upper    = 2*math.atan(scale)
            bias_upper = ((t_upper-math.pi)*scale) + math.log(1+math.cos(t_upper))
            segment_upper_estimate = torch.log(weight) + (scale*(math.pi-torch.acos(segmented_similarity)) + bias_upper)
        # Hamming Distance  ##################################################################################################################################################################################################################################################
        elif(args.option == "HD"):
            if(args.angle == "ln"):
                segment_lower_estimate = torch.log(1 - segmented_similarity)
                segment_upper_estimate = torch.log(1 + segmented_similarity)
            elif(args.angle == "arccos"):
                segment_lower_estimate = torch.acos(segmented_similarity) - math.pi/2
                segment_upper_estimate = math.pi/2 - torch.acos(segmented_similarity) 
            else:
                segment_lower_estimate = -1 * segmented_similarity
                segment_upper_estimate = segmented_similarity

        elif(args.option == "Seg-Cos_Real_Float"):
            quantized_weight_support = torch.clamp(torch.log(weight_support) - (minimum / 2), max=0)
            quantized_weight_query   = torch.clamp(torch.log(weight_query)   - (minimum / 2), max=0)
            
            if(args.normalization):
                mean_weight_support = torch.mean(quantized_weight_support, dim=1)
                quantized_weight_support = -1*math.pi*args.factor * quantized_weight_support / mean_weight_support[:, None]
                quantized_weight_support = torch.where(mean_weight_support[:, None] == 0, -1*math.pi*args.factor, quantized_weight_support)
                if(args.complete):
                    mean_weight_query = torch.mean(quantized_weight_query, dim=1)
                    quantized_weight_query = -1*math.pi*args.factor * quantized_weight_query / mean_weight_query[:, None]
                    quantized_weight_query = torch.where(mean_weight_query[:, None] == 0, -1*math.pi*args.factor, quantized_weight_query)
            segment_lower_estimate = quantized_weight_support[None, :, :] + quantized_weight_query[:, None, :] + torch.log(1-segmented_similarity) - (1 / scale)
            segment_upper_estimate = quantized_weight_support[None, :, :] + quantized_weight_query[:, None, :] + torch.log(1+segmented_similarity) - (1 / scale)
        elif(args.option == "Seg-Cos_Float"):
            t    = 2*math.atan(1/(scale))
            bias = ((0-t)*scale) + math.log(1-math.cos(t))
            quantized_weight_support = (1/scale) * (torch.log(weight_support) + ((bias - minimum) / 2))
            quantized_weight_query   = (1/scale) * (torch.log(weight_query)   + ((bias - minimum) / 2))
            # if(args.normalization):
            #     a = torch.mean(torch.log(weight_support), dim=1)
            #     b = torch.log(weight_support) * (-1*0.2) / a[:, None]
            #     quantized_weight_support = (1/scale) * (torch.log(weight_support) * b + ((bias - minimum) / 2))
            # else:
            # mean_weight_support_i = torch.mean(torch.log(weight_support), dim=1)    
            # mean_weight_support   = torch.mean(quantized_weight_support, dim=1)
            # mean_weight_query     = torch.mean(quantized_weight_query,   dim=1)
            # mean_weight_all       = torch.mean(quantized_weight_support)
            # z = torch.mean(torch.log(weight_support), dim=1)
            # print(torch.sum((z<-0.39).float()) / torch.numel(z))
            # print(torch.min(torch.mean(torch.log(weight_support), dim=1)))
            # print(torch.max(torch.mean(torch.log(weight_support), dim=1)))

            quantized_weight_support = torch.clamp(quantized_weight_support, max=0)
            quantized_weight_query   = torch.clamp(quantized_weight_query,   max=0)
            
            if(args.normalization):
                # alpha_tensor = torch.ones_like(z) * 0.15 # 0.15
                # t            = 2*torch.atan(alpha_tensor)
                # bias         = ((0-t)/alpha_tensor) + torch.log(1-torch.cos(t))
                # quantized_weight_support_normalize = alpha_tensor[:, None] * (torch.log(weight_support) + ((bias[:, None] + 2) / 2))
                # mean_weight_support                = torch.mean(quantized_weight_support_normalize, dim=1)
                # diff     = torch.where(z > -0.39, 0, minimum - mean_weight_support)
                # max_diff = torch.max(torch.abs(diff))

                # while(max_diff > 0.001):
                #     alpha_tensor = torch.clamp(alpha_tensor + (torch.abs(diff) > 0.001) * -1 * diff * torch.sqrt(torch.abs(diff)), min=0)# torch.where(diff > 0, -0.0001, +0.0001)
                #     t            = 2*torch.atan(alpha_tensor)
                #     bias         = ((0-t)/alpha_tensor) + torch.log(1-torch.cos(t))
                #     quantized_weight_support_normalize = alpha_tensor[:, None] * (torch.log(weight_support) + ((bias[:, None] + 2) / 2))
                #     mean_weight_support                = torch.mean(quantized_weight_support_normalize, dim=1)
                #     diff     = torch.where(z > -0.39, 0, minimum - mean_weight_support)
                #     max_diff = torch.max(torch.abs(diff))
                # quantized_weight_support = torch.where(z[:, None] > -0.39, quantized_weight_support, quantized_weight_support_normalize)

                # (-2*math.pi/15)
                # 2:3
                # (-1*math.pi/8)
                # 3:5
                mean_weight_support = torch.mean(quantized_weight_support, dim=1)
                quantized_weight_support = quantized_weight_support + (-1*math.pi*args.factor - mean_weight_support[:, None]) * quantized_weight_support / mean_weight_support[:, None]
                if(args.complete):
                    mean_weight_query = torch.mean(quantized_weight_query, dim=1)
                    quantized_weight_query = quantized_weight_query + (-1*math.pi*args.factor - mean_weight_query[:, None]) * quantized_weight_query / mean_weight_query[:, None]
                # quantized_weight_support = quantized_weight_support + (minimum - mean_weight_support[:, None]) #* torch.log(weight_support) / mean_weight_support_i[:, None]
                # quantized_weight_query   = quantized_weight_query   + (minimum - mean_weight_query[:, None])   * quantized_weight_query   /   mean_weight_query[:, None]# torch.log(weight_query) / mean_weight_query_i[:, None]

            # mean_weight_support = torch.mean(quantized_weight_support, dim=1)
            # mean_weight_support_numpy = mean_weight_support.detach().cpu().numpy()

            # # Plotting the histogram
            # plt.figure(figsize=(8, 6))
            # plt.hist(mean_weight_support_numpy, bins=100, color='blue', alpha=0.7)
            # plt.title('Histogram of Mean Log-Transformed Weight Support')
            # plt.xlabel('Mean Log-Transformed Weight Support')
            # plt.ylabel('Frequency')
            # plt.grid(True)
            # plt.savefig('b.png')
            # quantized_weight_support = quantized_weight_support / torch.mean(quantized_weight_support, dim=1)[:, None] * torch.mean(quantized_weight_support)
            # quantized_weight_query   = quantized_weight_query   / torch.mean(quantized_weight_query, dim=1)[:, None]   * torch.mean(quantized_weight_query)
            
            maximum_distance = torch.clamp(math.pi + 2*torch.minimum(quantized_weight_support[None, :, :], quantized_weight_query[:, None, :]), min=0)
            # print(torch.mean(maximum_distance))
            # print(a)
            # maximum_distance = torch.clamp(math.pi + 2*mean_weight_all, min=0)
            segment_lower_estimate = torch.clamp(quantized_weight_support[None, :, :] + quantized_weight_query[:, None, :] + torch.acos(segmented_similarity),         max=maximum_distance)
            segment_upper_estimate = torch.clamp(quantized_weight_support[None, :, :] + quantized_weight_query[:, None, :] + math.pi-torch.acos(segmented_similarity), max=maximum_distance)

        elif(args.option == "Seg-Cos_QuantAng"):
            division = 2*math.pi/args.N
            angle_query_segment   = torch.atan2(  query_segment[:,:,1],   query_segment[:,:,0])
            angle_support_segment = torch.atan2(support_segment[:,:,1], support_segment[:,:,0])
            angle_query_segment   = torch.round(  angle_query_segment / division)
            angle_support_segment = torch.round(angle_support_segment / division)
            angle_query_segment   = torch.where(  angle_query_segment < 0, args.N +   angle_query_segment,   angle_query_segment)
            angle_support_segment = torch.where(angle_support_segment < 0, args.N + angle_support_segment, angle_support_segment)
            angle_difference = torch.abs(angle_query_segment[:, None, :] - angle_support_segment[None, :, :])
            angle_difference = torch.min(angle_difference, args.N - angle_difference) * division

            t_lower    = 2*math.atan(1/scale)
            bias_lower = ((0-t_lower)*scale) + math.log(1-math.cos(t_lower))
            segment_lower_estimate = (1/scale) * (torch.log(weight) + (scale*angle_difference + bias_lower) - minimum)
            t_upper    = 2*math.atan(scale)
            bias_upper = ((t_upper-math.pi)*scale) + math.log(1+math.cos(t_upper))
            segment_upper_estimate = (1/scale) * (torch.log(weight) + (scale*(math.pi-angle_difference) + bias_upper) - minimum)
        
        elif(args.option == "Seg-Cos_QuantAngMag"):
            division = 2*math.pi/args.N
            angle_query_segment   = torch.atan2(  query_segment[:,:,1],   query_segment[:,:,0])
            angle_support_segment = torch.atan2(support_segment[:,:,1], support_segment[:,:,0])
            angle_query_segment   = torch.round(  angle_query_segment / division)
            angle_support_segment = torch.round(angle_support_segment / division)
            angle_query_segment   = torch.where(  angle_query_segment < 0, args.N +   angle_query_segment,   angle_query_segment)
            angle_support_segment = torch.where(angle_support_segment < 0, args.N + angle_support_segment, angle_support_segment)
            angle_difference = torch.abs(angle_query_segment[:, None, :] - angle_support_segment[None, :, :])
            angle_difference = torch.min(angle_difference, args.N - angle_difference)

            t_lower    = 2*math.atan(1/scale)
            bias_lower = ((0-t_lower)*scale) + math.log(1-math.cos(t_lower))
            quantized_lower_weight_support = torch.round((1/scale) * (torch.log(weight_support) + ((bias_lower - minimum) / 2)) / division)
            quantized_lower_weight_query   = torch.round((1/scale) * (torch.log(weight_query)   + ((bias_lower - minimum) / 2)) / division)
            segment_lower_estimate = quantized_lower_weight_support[None, :, :] + quantized_lower_weight_query[:, None, :] + angle_difference
            segment_lower_estimate = segment_lower_estimate * division

            t_upper    = 2*math.atan(scale)
            bias_upper = ((t_upper-math.pi)*scale) + math.log(1+math.cos(t_upper))
            quantized_upper_weight_support = torch.round((1/scale) * (torch.log(weight_support) + ((bias_upper - minimum) / 2)) / division)
            quantized_upper_weight_query   = torch.round((1/scale) * (torch.log(weight_query)   + ((bias_upper - minimum) / 2)) / division)
            segment_upper_estimate = quantized_upper_weight_support[None, :, :] + quantized_upper_weight_query[:, None, :] + ((args.N/2)-angle_difference)
            segment_upper_estimate = segment_upper_estimate * division

        elif(args.option == "Seg-Cos_Fixed"):
            division = 2*math.pi/args.N
            angle_query_segment   = torch.atan2(  query_segment[:,:,1],   query_segment[:,:,0])
            angle_support_segment = torch.atan2(support_segment[:,:,1], support_segment[:,:,0])
            angle_query_segment   = torch.round(  angle_query_segment / division)
            angle_support_segment = torch.round(angle_support_segment / division)
            angle_query_segment   = torch.where(  angle_query_segment < 0, args.N +   angle_query_segment,   angle_query_segment)
            angle_support_segment = torch.where(angle_support_segment < 0, args.N + angle_support_segment, angle_support_segment)
            angle_difference = torch.abs(angle_query_segment[:, None, :] - angle_support_segment[None, :, :])
            angle_difference = torch.min(angle_difference, args.N - angle_difference)

            t    = 2*math.atan(1/scale)
            bias = ((0-t)*scale) + math.log(1-math.cos(t))
            quantized_weight_support = torch.round((1/scale) * (torch.log(weight_support) + ((bias - minimum) / 2)) / division)
            quantized_weight_query   = torch.round((1/scale) * (torch.log(weight_query)   + ((bias - minimum) / 2)) / division)
            quantized_weight_support = torch.clamp(quantized_weight_support, max=0)
            quantized_weight_query   = torch.clamp(quantized_weight_query,   max=0)
            segment_lower_estimate = quantized_weight_support[None, :, :] + quantized_weight_query[:, None, :] + angle_difference
            segment_upper_estimate = quantized_weight_support[None, :, :] + quantized_weight_query[:, None, :] + ((args.N/2)-angle_difference)
            segment_lower_estimate = segment_lower_estimate * division
            segment_upper_estimate = segment_upper_estimate * division

        elif(args.option == "Seg-Cos_TCAM"):
            print(query_segment[0,1:3])
            print(query_segment[53,1:3])
            print(query_segment[23,1:3])
            division = 2*math.pi/args.N
            angle_query_segment   = torch.atan2(  query_segment[:,:,1],   query_segment[:,:,0])
            angle_support_segment = torch.atan2(support_segment[:,:,1], support_segment[:,:,0])
            angle_query_segment   = torch.round(  angle_query_segment / division)
            angle_support_segment = torch.round(angle_support_segment / division)
            angle_query_segment   = torch.where(  angle_query_segment < 0, args.N +   angle_query_segment,   angle_query_segment)
            angle_support_segment = torch.where(angle_support_segment < 0, args.N + angle_support_segment, angle_support_segment)
            angle_difference = torch.abs(angle_query_segment[:, None, :] - angle_support_segment[None, :, :])
            angle_difference = torch.min(angle_difference, args.N - angle_difference)

            t    = 2*math.atan(1/scale)
            bias = ((0-t)*scale) + math.log(1-math.cos(t))
            quantized_weight_support = (1/scale) * (torch.log(weight_support) + ((bias - minimum) / 2))
            quantized_weight_query   = (1/scale) * (torch.log(weight_query)   + ((bias - minimum) / 2))
            quantized_weight_support = torch.clamp(quantized_weight_support, max=0)
            quantized_weight_query   = torch.clamp(quantized_weight_query,   max=0)
            if(args.normalization):
                num_to_set = round(args.dimension * args.factor)
                if((args.N == 2) or (args.N == 4)):
                    _, indices = torch.topk(-quantized_weight_support, num_to_set, dim=1)
                    quantized_weight_support = torch.zeros_like(quantized_weight_support)
                    quantized_weight_support.scatter_(1, indices, -math.pi)
                else:
                    mean_weight_support = torch.mean(quantized_weight_support, dim=1)
                    # quantized_weight_support = quantized_weight_support + (-1*math.pi*args.factor - mean_weight_support[:, None]) * quantized_weight_support / mean_weight_support[:, None]
                    quantized_weight_support = -1*math.pi*args.factor * quantized_weight_support / mean_weight_support[:, None]
                    quantized_weight_support = torch.where(mean_weight_support[:, None] == 0, -1*math.pi*args.factor, quantized_weight_support)
                if(args.complete):
                    if((args.N == 2) or (args.N == 4)):
                        _, indices = torch.topk(-quantized_weight_query, num_to_set, dim=1)
                        quantized_weight_query = torch.zeros_like(quantized_weight_query)
                        quantized_weight_query.scatter_(1, indices, -math.pi)
                    else:
                        mean_weight_query = torch.mean(quantized_weight_query, dim=1)
                        # quantized_weight_query = quantized_weight_query + (-1*math.pi*args.factor - mean_weight_query[:, None]) * quantized_weight_query / mean_weight_query[:, None]
                        quantized_weight_query = -1*math.pi*args.factor * quantized_weight_query / mean_weight_query[:, None]
                        quantized_weight_query = torch.where(mean_weight_query[:, None] == 0, -1*math.pi*args.factor, quantized_weight_query)
            # print(-1*math.pi*args.factor * 25)
            # print(-1*math.pi*args.factor / division * 25)
            quantized_weight_support_round = torch.round(quantized_weight_support / division)
            quantized_weight_query_round   = torch.round(quantized_weight_query   / division)

            # target_number = -1*round(args.factor * 2 * args.dimension * args.N / 2)
            # mean_weight_support = torch.mean(quantized_weight_support, dim=1)
            # sum_weight_support_round = 2*torch.sum(quantized_weight_support_round, dim=1)
            # # print(target_number)
            # diff = target_number - sum_weight_support_round
            # quantized_weight_support_fake = quantized_weight_support
            # old_min = torch.min(sum_weight_support_round)
            # old_max = torch.max(sum_weight_support_round)
            # print(old_min, old_max)
            # step  = 2e-3
            # count = 0
            # while(torch.max(torch.abs(diff)) != 0):
            #     quantized_weight_support_fake = quantized_weight_support_fake + step*torch.sign(diff[:, None]) * quantized_weight_support / mean_weight_support[:, None]
            #     quantized_weight_support_round = torch.round(quantized_weight_support_fake / division)
            #     sum_weight_support_round = 2*torch.sum(quantized_weight_support_round, dim=1)
            #     diff = target_number - sum_weight_support_round
            #     new_min = torch.min(sum_weight_support_round)
            #     new_max = torch.max(sum_weight_support_round)
            #     if((new_min <= old_min) or (new_max >= old_max)):
            #         count = count + 1
            #     else:
            #         count = 0
            #     if(count > 20):
            #         if(step > 1e-6):
            #             step = step / 2
            #         count = 0
            #         # print(step)
            #     old_min = new_min
            #     old_max = new_max
            #     if(count > 1000):
            #         break
                # print(torch.min(sum_weight_support_round), torch.max(sum_weight_support_round), torch.mean(sum_weight_support_round))
                # print(torch.max(torch.abs(diff)))


            quantized_weight_support = quantized_weight_support_round
            quantized_weight_query   = quantized_weight_query_round
                
            maximum_distance = torch.clamp((args.N/2) + 2*torch.minimum(quantized_weight_support[None, :, :], quantized_weight_query[:, None, :]), min=0)
            # print(angle_difference)
            # print(quantized_weight_support)
            # print(quantized_weight_query)
            # print(maximum_distance)
            # print(a)
            segment_lower_estimate = torch.clamp(quantized_weight_support[None, :, :] + quantized_weight_query[:, None, :] + angle_difference,            max=maximum_distance)
            segment_lower_estimate = segment_lower_estimate * division
            segment_upper_estimate = torch.clamp(quantized_weight_support[None, :, :] + quantized_weight_query[:, None, :] + (args.N/2)-angle_difference, max=maximum_distance)
            # segment_upper_estimate = torch.clamp((args.N/2)-angle_difference, max=maximum_distance)
            segment_upper_estimate = segment_upper_estimate * division

        elif(args.option == "Seg-Cos_TCAM_Bit"):
            division = 2*math.pi/args.N
            angle_query_segment   = torch.atan2(  query_segment[:,:,1],   query_segment[:,:,0])
            angle_support_segment = torch.atan2(support_segment[:,:,1], support_segment[:,:,0])
            angle_query_segment   = torch.round(  angle_query_segment / division)
            angle_support_segment = torch.round(angle_support_segment / division)
            angle_query_segment   = torch.where(  angle_query_segment < 0, args.N +   angle_query_segment,   angle_query_segment)
            angle_support_segment = torch.where(angle_support_segment < 0, args.N + angle_support_segment, angle_support_segment)
            codeword = torch.arange(args.N//2).to('cuda')
            codeword = codeword.unsqueeze(0) + torch.arange(args.N).to('cuda').unsqueeze(1)
            codeword = codeword % (args.N)
            codeword = codeword >= (args.N // 2)
            
            codeword_query   = codeword[angle_query_segment.long()]
            codeword_support = codeword[angle_support_segment.long()]

            angle_difference = codeword_query[:, None, :, :] ^ codeword_support[None, :, :, :]

            t    = 2*math.atan(1/scale)
            bias = ((0-t)*scale) + math.log(1-math.cos(t))
            if(args.normalization):
                num_to_set = math.floor(args.dimension * args.factor * (args.N // 2))
                support_segment_mag_ = support_segment_mag_.view(support_segment_mag_.shape[0], -1)
                _, indices = torch.topk(-support_segment_mag_, num_to_set, dim=1)
                quantized_weight_support = torch.ones_like(support_segment_mag_)
                quantized_weight_support.scatter_(1, indices, 0)
                quantized_weight_support = quantized_weight_support.view(support_segment_mag_.shape[0], -1, args.N // 2)
                if(args.complete):
                    query_segment_mag_ = query_segment_mag_.view(query_segment_mag_.shape[0], -1)
                    _, indices = torch.topk(-query_segment_mag_, num_to_set, dim=1)
                    quantized_weight_query = torch.ones_like(query_segment_mag_)
                    quantized_weight_query.scatter_(1, indices, 0)
                    quantized_weight_query = quantized_weight_query.view(quantized_weight_query.shape[0], -1, args.N // 2)
            segment_lower_estimate = torch.sum(angle_difference    * quantized_weight_support[None, :, :, :] * quantized_weight_query[:, None, :, :], dim=3)
            segment_upper_estimate = torch.sum((~angle_difference) * quantized_weight_support[None, :, :, :] * quantized_weight_query[:, None, :, :], dim=3)

            # segment_lower_estimate = torch.clamp(quantized_weight_support[None, :, :] + quantized_weight_query[:, None, :] + angle_difference,            max=maximum_distance)
            segment_lower_estimate = segment_lower_estimate * division
            # segment_upper_estimate = torch.clamp(quantized_weight_support[None, :, :] + quantized_weight_query[:, None, :] + (args.N/2)-angle_difference, max=maximum_distance)
            segment_upper_estimate = segment_upper_estimate * division

        if("Seg-Cos" in args.option):
            segment_lower_estimate = torch.clamp(segment_lower_estimate, min=0) # * scale + minimum
            segment_upper_estimate = torch.clamp(segment_upper_estimate, min=0) # * scale + minimum
        else:
            if(args.minimum_clamp):
                segment_lower_estimate = torch.clamp(segment_lower_estimate, min=minimum)
                segment_upper_estimate = torch.clamp(segment_upper_estimate, min=minimum)
        
        lower_estimate = -1*torch.mean(segment_lower_estimate, dim=2)
        upper_estimate =    torch.mean(segment_upper_estimate, dim=2)
        weight_mean = torch.mean(weight, dim=2)
        

        all_weight_mean.append(weight_mean)
        all_similarity_truth.append(similarity_truth)
        if(args.bound == "lower"):
            similarity = lower_estimate
        elif(args.bound == "upper"):
            similarity = upper_estimate
        elif(args.bound == "complementary"):
            similarity = (lower_estimate + upper_estimate) / 2
        # Cauchy-Schwarz Inequality ##########################################################################################################################################################################################################################################
        if(args.option == "CSI"):
            if(args.bound == "lower"):
                cosine = similarity + 1
            elif(args.bound == "upper"):
                cosine = similarity - 1
            elif(args.bound == "complementary"):
                cosine = similarity
        # Jensen's Inequality # Taylor Expansion #######################################################################################################################################################################################################################################
        elif((args.option == "JI") or (args.option == "TLE") or ((args.option == "HD") and (args.angle == "ln"))or ("Seg-Cos" in args.option)):
            if(args.bound == "lower"):
                cosine = -1*torch.exp(-1*similarity) + 1
            elif(args.bound == "upper"):
                cosine = torch.exp(similarity) - 1
            elif(args.bound == "complementary"):
                cosine = 2 / (torch.exp(-1*similarity) + 1) - 1
        # Hamming Distance  ##################################################################################################################################################################################################################################################
        elif(args.option == "HD"):
            if(args.angle == "arccos"):
                cosine = torch.cos(similarity + math.pi/2)
            else:
                cosine = similarity
        angle = -1 * torch.acos(torch.clamp(cosine, -1, 1)) / math.pi * 180
        # weighted_segmented_similarity = ( 1*torch.mean(weight * (1 + segmented_similarity), dim=2) - 1)
        # similarity = -1*torch.sqrt(torch.sum((xq[:,None,:] - xb[None,:,:])**2, dim=2))
        all_differences.append((angle_truth - angle).detach().cpu())
        all_similarities.append(similarity.detach().cpu())
        all_angles.append(angle.detach().cpu())

    all_differences  = torch.cat(all_differences, dim=1)
    all_similarities = torch.cat(all_similarities, dim=1)
    all_angles       = torch.cat(all_angles, dim=1)
    all_weight_mean      = torch.cat(all_weight_mean, dim=1)
    all_similarity_truth = torch.cat(all_similarity_truth, dim=1)

    # for i in range(all_weight_mean.shape[0]):
    #     correlation = torch.corrcoef(torch.stack([all_weight_mean[i].flatten(), all_similarity_truth[i].flatten()]))
    #     if torch.isnan(correlation[0, 1]):
    #         correlation_array.append(0.0)
    #     else:
    #         correlation_array.append(correlation[0, 1].item())

    # data_array = torch.cat([d.flatten() for d in data_array])
    # plt.figure(figsize=(8, 2.5))
    # # Distribution of Correlation
    # plt.hist(data_array, bins=50, color='blue', alpha=0.7)
    # plt.title('Cosine Similarity > 0.75')
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.xlabel('Value', fontsize=16)
    # plt.ylabel('Frequency', fontsize=16)
    # plt.grid(True)
    # plt.savefig(f'data_distribution_{args.dataset}_{args.coupled_dimension}_1.png')
    # plt.close()
    # print(a)

    # gap = 1 - all_weight_mean
    # gap_array.append(torch.mean(gap).item())

    # print("Mean Correlation Coefficient:", np.mean(correlation_array))
    # print("Mean Gap:",                     np.mean(gap_array))
    # plot_array = np.array(correlation_array)
    # plt.figure(figsize=(10, 5))
    # # Distribution of Correlation
    # plt.hist(plot_array, bins=50, color='blue', alpha=0.7)
    # plt.title('Distribution of Correlation Coefficients')
    # plt.xlabel('Correlation Coefficient')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig(f'correlation_distribution_{args.dataset}_{args.coupled_dimension}.png')
    # plt.close()

    topk_values, topk_indices = torch.topk(all_similarities, k=topk, dim=1)

    all_differences = all_differences.flatten()
    all_similarities = all_similarities.flatten()
    all_angles = all_angles.flatten()

    return topk_indices, torch.mean(all_angles).item(), torch.mean(all_differences).item(), torch.std(all_differences).item()

def main():
    if args.find_best_recall:
        find_best_recall()
    else:
        inference()

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if args.dataset == "glove-25":
        file_path = './GloVe/glove-25-angular.hdf5'
    elif args.dataset == "glove-50":
        file_path = './GloVe/glove-50-angular.hdf5'
    elif args.dataset == "glove-100":
        file_path = './GloVe/glove-100-angular.hdf5'
    elif args.dataset == "glove-200":
        file_path = './GloVe/glove-200-angular.hdf5'
    elif args.dataset == "NYTimes":
        file_path = './NYTimes/nytimes-256-angular.hdf5'
    elif args.dataset == "DEEP1B":
        file_path = './DEEP1B/deep-image-96-angular.hdf5'
    elif args.dataset == "Last.fm":
        file_path = './Last.fm/lastfm-64-dot.hdf5'
    elif args.dataset == "COCO-I2I":
        file_path = './COCO-I2I/coco-i2i-512-angular.hdf5'
    elif args.dataset == "COCO-T2I":
        file_path = './COCO-T2I/coco-t2i-512-angular.hdf5'
    else:
        file_path = f"./{args.dataset}/{args.dataset}_{args.num_subset}.hdf5"

    total_query_num = args.total_query_num
    with h5py.File(file_path, 'r') as f:
        print("Query Set shape:", f['test'].shape)
        if total_query_num == 0:
            total_query_num = f['test'].shape[0]
        support_set_dataset = SupportSetDataset(f['train'][:])
        batch_size = args.batch_size
        support_loader = DataLoader(support_set_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        print("Query Label shape:", f['neighbors'].shape)
        print("Support Set shape:", f['train'].shape)

        if(total_query_num % args.query_num != 0):
            print("Error: The total query number is not divisible by the query number.")

    counter = 0
    recall = 0
    
    if args.similarity == "LSH":
        L = args.param
        hashing_matrix = torch.randn((args.dimension, args.dimension*L), device=device)

    while counter < total_query_num:
        with h5py.File(file_path, 'r') as f:
            test_set = torch.tensor(f['test'][counter : counter+args.query_num], dtype=torch.float32).to(device)
            test_neighbors_id = f['neighbors'][counter : counter+args.query_num]

        print("Processed query from", counter, "to", counter + test_set.shape[0] - 1)

        if args.similarity == "cosine":
            result_id, average_similarity, std_similarity = do_compute_cosine(support_loader, test_set, topk=args.topk)
            average_difference = average_similarity
            std_difference = std_similarity
        elif args.similarity == "segmented_cosine":
            result_id, average_similarity, average_difference, std_difference = do_compute_segmented_cosine(support_loader, test_set, topk=args.topk, scale=(1 / args.alpha), minimum=args.beta)
        elif args.similarity == "LSH":
            result_id, average_similarity, average_difference, std_difference = do_compute_lsh(support_loader, test_set, L, hashing_matrix, topk=args.topk)
        elif args.similarity == "L1norm":
            result_id, average_similarity, average_difference, std_difference = do_compute_L1norm(support_loader, test_set, topk=args.topk)
        average_similarity_array = []
        average_difference_array = []
        std_difference_array     = []

        average_similarity_array.append(average_similarity)
        average_difference_array.append(average_difference)
        std_difference_array.append(std_difference)

        for i in range(test_set.shape[0]):
            # print("Query ID:", i, "Query Label:", test_neighbors_id[i])
            # print("Result ID:", result_id[i])
            intersection = torch.isin(result_id[i], torch.tensor(test_neighbors_id[i], device=result_id.device))
            # print("Intersection:", intersection.sum())
            recall += intersection.sum().item() / len(test_neighbors_id[i])

        counter += args.query_num

    average_similarity = torch.tensor(average_similarity_array, device=device)
    total_average_similarity = torch.mean(average_similarity)
    average_difference = torch.tensor(average_difference_array, device=device)
    total_average_difference = torch.mean(average_difference)
    std_difference = torch.tensor(std_difference_array, device=device)
    total_std_difference =  torch.sqrt(torch.mean((std_difference**2) + (average_difference - total_average_difference)**2))
    print("Difference:", total_average_difference, total_std_difference)
    print("Similarity:", total_average_similarity)

    ave_recall = recall / total_query_num
    print("Recall:", ave_recall)


def find_best_recall():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if args.dataset == "glove-25":
        file_path = './GloVe/glove-25-angular.hdf5'
    elif args.dataset == "glove-50":
        file_path = './GloVe/glove-50-angular.hdf5'
    elif args.dataset == "glove-100":
        file_path = './GloVe/glove-100-angular.hdf5'
    elif args.dataset == "glove-200":
        file_path = './GloVe/glove-200-angular.hdf5'
    elif args.dataset == "NYTimes":
        file_path = './NYTimes/nytimes-256-angular.hdf5'
    elif args.dataset == "DEEP1B":
        file_path = './DEEP1B/deep-image-96-angular.hdf5'
    elif args.dataset == "Last.fm":
        file_path = './Last.fm/lastfm-64-dot.hdf5'
    elif args.dataset == "COCO-I2I":
        file_path = './COCO-I2I/coco-i2i-512-angular.hdf5'
    elif args.dataset == "COCO-T2I":
        file_path = './COCO-T2I/coco-t2i-512-angular.hdf5'
    else:
        file_path = f"./{args.dataset}/{args.dataset}_{args.num_subset}.hdf5"

    total_query_num = args.total_query_num
    with h5py.File(file_path, 'r') as f:
        print("Query Set shape:", f['test'].shape)
        if total_query_num == 0:
            total_query_num = f['test'].shape[0]
        support_set_dataset = SupportSetDataset(f['train'][:])
        batch_size = args.batch_size
        support_loader = DataLoader(support_set_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        print("Query Label shape:", f['neighbors'].shape)
        print("Support Set shape:", f['train'].shape)

        if(total_query_num % args.query_num != 0):
            print("Error: The total query number is not divisible by the query number.")

    best_alpha  = None
    best_beta   = None
    best_recall = 0

    alpha_start = args.alpha #0.5
    alpha_end   = 2.1
    alpha_step  = 0.1
    beta_start  = args.beta #-1.5
    beta_end    = 3.0 # 1.1
    beta_step   = 0.1
    # alpha_start = 0.1#-4# 0.2
    # alpha_end   = 0.2# 5.1
    # alpha_step  = 0.1
    # beta_start  = -4
    # beta_end    = 3.14
    # beta_step   = 0.1

    for alpha in torch.arange(alpha_start, alpha_end, alpha_step):
        alpha = alpha.item()
        tmp_best_alpha  = None
        tmp_best_beta   = None
        tmp_best_recall = 0

        early_stop = 0

        for beta in torch.arange(beta_start, beta_end, beta_step):
            beta = beta.item()
            print("Alpha:", alpha, "Beta:", beta)

            counter = 0
            recall = 0

            while counter < total_query_num:
                with h5py.File(file_path, 'r') as f:
                    test_set = torch.tensor(f['test'][counter : counter+args.query_num], dtype=torch.float32).to(device)
                    test_neighbors_id = f['neighbors'][counter : counter+args.query_num]

                print("Processed query from", counter, "to", counter + test_set.shape[0] - 1)

                if args.similarity == "segmented_cosine":
                    result_id, average_similarity, average_difference, std_difference = do_compute_segmented_cosine(support_loader, test_set, topk=args.topk, scale=(1 / alpha), minimum=beta)
                elif args.similarity == "L1norm":
                    result_id, average_similarity, average_difference, std_difference = do_compute_L1norm(support_loader, test_set, topk=args.topk, clamp_ratio = beta)
                average_similarity_array = []
                average_difference_array = []
                std_difference_array     = []

                average_similarity_array.append(average_similarity)
                average_difference_array.append(average_difference)
                std_difference_array.append(std_difference)

                for i in range(test_set.shape[0]):
                    # print("Query ID:", i, "Query Label:", test_neighbors_id[i])
                    # print("Result ID:", result_id[i])
                    intersection = torch.isin(result_id[i], torch.tensor(test_neighbors_id[i], device=result_id.device))
                    # print("Intersection:", intersection.sum())
                    recall += intersection.sum().item() / len(test_neighbors_id[i])

                counter += args.query_num

            average_similarity = torch.tensor(average_similarity_array, device=device)
            total_average_similarity = torch.mean(average_similarity)
            average_difference = torch.tensor(average_difference_array, device=device)
            total_average_difference = torch.mean(average_difference)
            std_difference = torch.tensor(std_difference_array, device=device)
            total_std_difference =  torch.sqrt(torch.mean((std_difference**2) + (average_difference - total_average_difference)**2))

            ave_recall = recall / total_query_num

            print("Recall:", ave_recall, "with alpha:", alpha, "with beta:", beta)
            print("Difference:", total_average_difference, total_std_difference)
            print("Similarity:", total_average_similarity)

            if ave_recall >= tmp_best_recall:
                early_stop = 0
                tmp_best_recall = ave_recall
                tmp_best_alpha = alpha
                tmp_best_beta = beta
                tmp_average_difference = total_average_difference
                tmp_std_difference     = total_std_difference
                tmp_average_similarity = total_average_similarity
                # print("Temporal Best recall:", tmp_best_recall, "with alpha:", tmp_best_alpha, "with beta:", tmp_best_beta)
                if beta == beta_end:
                    print("##############################################################")
                    print("Temporal Best recall:", tmp_best_recall, "with alpha:", tmp_best_alpha, "with beta:", tmp_best_beta)
                    print("Difference:", tmp_average_difference, tmp_std_difference)
                    print("Similarity:", tmp_average_similarity)
                    print("##############################################################")
                
                last_time_recall = ave_recall

            else:
                if ave_recall < last_time_recall:   
                    early_stop += 1
                else:
                    early_stop = 0
                last_time_recall = ave_recall
                if early_stop >= 2:
                    print("##############################################################")
                    print("Temporal Best recall:", tmp_best_recall, "with alpha:", tmp_best_alpha, "with beta:", tmp_best_beta)
                    print("Difference:", tmp_average_difference, tmp_std_difference)
                    print("Similarity:", tmp_average_similarity)
                    print("##############################################################")
                    break
                else:
                    print("Early stop count:", early_stop, "with alpha:", alpha, "with beta:", beta)
            
        if tmp_best_recall >= best_recall:
            best_recall = tmp_best_recall
            best_alpha = tmp_best_alpha
            best_beta = tmp_best_beta
            best_average_difference = tmp_average_difference
            best_std_difference = tmp_std_difference
            best_average_similarity = tmp_average_similarity
        else:
            print("##############################################################")
            print("Best recall:", best_recall, "with alpha:", best_alpha, "with beta:", best_beta)
            print("Difference:", best_average_difference, best_std_difference)
            print("Similarity:", best_average_similarity)
            print("##############################################################")
            break
        print("##############################################################")
        print("Best recall:", best_recall, "with alpha:", best_alpha, "with beta:", best_beta)
        print("Difference:", best_average_difference, best_std_difference)
        print("Similarity:", best_average_similarity)
        print("##############################################################")

        if args.option != "TLE":
            break
    
if __name__ == '__main__':
    args = process_args()
    if args.find_best_recall:
        find_best_recall()
    else:
        main()



# def scale_matrix(matrix, q_dim):
#     """
#     Normalzaiton of range sum of each row of the matrix.
#     Args:
#         matrix (torch.Tensor): 输入矩阵，形状为 (M, N)。
#         q_dim (int): 未使用参数，保留以匹配原函数签名。
#     Returns:
#         torch.Tensor: 调整后的矩阵，形状为 (M, N)，每一行的总和接近目标值。
#     """

#     device = matrix.device  # 保留设备一致性
#     # 确保输入为 PyTorch 张量
#     if not isinstance(matrix, torch.Tensor):
#         matrix = torch.tensor(matrix, dtype=torch.float32)

#     device = matrix.device  # 保留设备一致性
#     row_sums = matrix.sum(dim=1)  # 计算每一行的总和
#     # print("Std of row_sums:", row_sums.std())
#     row_sums = row_sums.float()  # 转换为浮点型以支持均值计算
#     row_sums = row_sums.to(device)  # 保证与 `matrix` 在同一设备
#     # print("STD of row_sums:", row_sums.std())

#     # 找到非零行
#     non_zero_rows = row_sums != 0

#     # 计算目标总和
#     if args.tolerance_remap_target == "mean":
#         target_sum = torch.round(row_sums.mean())
#     elif args.tolerance_remap_target == "max":
#         target_sum = torch.round(row_sums.max())
#     elif args.tolerance_remap_target == "median":
#         target_sum = torch.round(row_sums.median())
#     elif args.tolerance_remap_target == "q1":
#         target_sum = torch.round(row_sums.kthvalue(int(row_sums.size(0) * 0.25)).values)
#     elif args.tolerance_remap_target == "q3":
#         target_sum = torch.round(row_sums.kthvalue(int(row_sums.size(0) * 0.75)).values)

#     else:
#         assert False, "Invalid target sum type!"
#     # assert not torch.isnan(row_sums.mean()), "row_sums.mean() contains NaN!"
#     # assert not torch.isinf(row_sums.mean()), "row_sums.mean() contains Inf!"
#     # 计算缩放因子
#     scale_factors = torch.ones_like(row_sums, dtype=torch.float32, device=device)
#     scale_factors[non_zero_rows] = target_sum / row_sums[non_zero_rows]

#     # 缩放矩阵
#     scaled_matrix = torch.round(matrix * scale_factors.unsqueeze(1)).int()
#     # print("matrix shape:", matrix.shape, "device:", matrix.device)
#     # print("row_sums shape:", row_sums.shape, "values:", row_sums)
#     return scaled_matrix.to(device)


# def do_compute_seg_cosine(xb_dataloader, xq, topk=1000, batch_size=4000000, disable_tqdm=False): #mean
#     """
#     支持支持集合分批處理的 GPU 加速 seg_cosine_real_mob_encode 計算。
#     Args:
#         support_loader (DataLoader): 支持集合的 DataLoader，分批加載支持集合。
#         xq (torch.Tensor): 查詢集合，形狀為 (M, D)。
#         topk (int): 要返回的前 k 個最相似向量的索引。
#     Returns:
#         ids (torch.Tensor): 每個查詢向量的前 k 個最相似向量索引，形狀為 (M, topk)。
#         similarity (torch.Tensor): 每個查詢向量與支持集合的相似度矩陣。
#     """
#     device = xq.device  # 确保查询集合在 GPU 上
#     nq, q_dim = xq.shape  # 查询集合的维度
#     nb = sum(batch.shape[0] for batch in xb_dataloader)  # 计算支持集合总大小
#     # print("nb", nb)
#     coupled_dimension = 2
#     N = args.N
#     angle_division = 360 / N
#     hyper_clip_value = args.clip_value
#     hyper_tolerance_scale = args.tolerance_scale

#     idx = torch.arange(coupled_dimension).reshape(1, -1) + torch.arange(q_dim).reshape(-1, 1)
#     query_vector_fp = torch.cat([xq, xq[:, :coupled_dimension - 1]], dim=1)
#     batch_query_vector_fp = query_vector_fp[:, idx]
#     query_mag = torch.sqrt(torch.sum(batch_query_vector_fp ** 2, dim=2))
#     angle_query_vector = torch.rad2deg(torch.atan2(batch_query_vector_fp[:, :, 1], batch_query_vector_fp[:, :, 0]))
#     group_query_vector = torch.round(angle_query_vector / angle_division).to(device)
#     group_query_vector = torch.where(group_query_vector >= 0, group_query_vector, group_query_vector + N)
    

#     mean_query_mag = query_mag.mean(dim=1, keepdim=True)  
#     std_query_mag = query_mag.std(dim=1, keepdim=True)    
#     std_query_vector = (query_mag - mean_query_mag) / std_query_mag  
#     all_query_mag = torch.sqrt(torch.sum(query_mag ** 2, dim=1))  

#     if args.tolerance_function == "linear":
#         tolerance_query_vector = torch.clamp(hyper_clip_value - std_query_vector, min=0) * N / 64 * hyper_tolerance_scale
#     elif args.tolerance_function == "log":
#         tolerance_query_vector = torch.clamp(hyper_clip_value - torch.log(query_mag / all_query_mag[:, None]), min=0) * N / 64 * hyper_tolerance_scale
#     elif args.tolerance_function == "ratio":
#         tolerance_query_vector = torch.clamp(hyper_clip_value - (query_mag / all_query_mag[:, None]), min=0) * N / 64 * hyper_tolerance_scale

    
#     tolerance_query_vector = torch.round(tolerance_query_vector)
#     tolerance_query_vector = torch.clamp(tolerance_query_vector, min=0, max=max(0, N//4 - 1)).int()

   

#     key_memory_fp_list = []
    
#     for batch_xb in xb_dataloader:
#         batch_xb = batch_xb.to(device)  

#         batch_key_memory_fp = torch.cat([batch_xb, batch_xb[:, :coupled_dimension - 1]], dim=1)
#         key_memory_fp_list.append(batch_key_memory_fp)
    
#     key_memory_fp = torch.cat(key_memory_fp_list, dim=0)
#     key_memory_fp = key_memory_fp[:, idx]  
#     key_mag = torch.sqrt(torch.sum(key_memory_fp ** 2, dim=2))  
#     all_key_mag = torch.sqrt(torch.sum(key_mag ** 2, dim=1))  

#     if args.tolerance_function == "linear":
#         mean_key_mag = key_mag.mean(dim=1, keepdim=True)  
#         std_key_mag = key_mag.std(dim=1, keepdim=True)  
#         if args.dataset == "nytimes":
#             std_key_memory = (key_mag - mean_key_mag) / (std_key_mag + 1e-8)
#         else:
#             std_key_memory = (key_mag - mean_key_mag) / std_key_mag
#         tolerance_key_memory = torch.clamp(hyper_clip_value - std_key_memory, min=0) * N / 64 * hyper_tolerance_scale
#     elif args.tolerance_function == "log":
#         if args.dataset == "nytimes":
#             key_mag_ratio = key_mag / (all_key_mag[:,None] + 1e-8)
#             key_mag_ratio = torch.where(key_mag_ratio == 0, 1e-10, key_mag_ratio)
#         else:
#             key_mag_ratio = key_mag / (all_key_mag[:,None])
#         tolerance_key_memory = torch.clamp(
#             hyper_clip_value - torch.log(key_mag_ratio), min=0
#         ) * N / 64 * hyper_tolerance_scale
#     elif args.tolerance_function == "ratio":
#         tolerance_key_memory = torch.clamp(
#             hyper_clip_value - (key_mag / all_key_mag[:, None]), min=0
#         ) * N / 64 * hyper_tolerance_scale

#     tolerance_key_memory = torch.round(tolerance_key_memory)

#     if args.tolerance_remap:
#         tolerance_key_memory = scale_matrix(tolerance_key_memory, q_dim) 
    
#     tolerance_key_memory = torch.clamp(tolerance_key_memory, min=0, max=max(0, N//4 - 1))  
#     tolerance_key_memory = tolerance_key_memory.to(device)

    

    
#     similarity = torch.zeros((nq, nb), dtype=torch.float32, device=device)  
#     offset = 0
#     support_approximate_counts = torch.zeros(nb).to(device)
#     with torch.no_grad():
#         for batch_xb in tqdm(xb_dataloader, disable=disable_tqdm):
#             batch_size = batch_xb.shape[0]
#             batch_xb = batch_xb.to(device)  
#             # Batch processing
#             key_memory_fp = torch.cat([batch_xb, batch_xb[:, :coupled_dimension - 1]], dim=1)
#             batch_key_memory_fp = key_memory_fp[:, idx]
#             batch_key_memory_fp = batch_key_memory_fp.to(device)

#             angle_key_memory = torch.rad2deg(torch.atan2(batch_key_memory_fp[:, :, 1], batch_key_memory_fp[:, :, 0]))
#             group_key_memory = torch.round(angle_key_memory / angle_division)
#             group_key_memory = torch.where(group_key_memory >= 0, group_key_memory, group_key_memory + N)

#             group_key_memory = group_key_memory.to(device)
#             batch_tolerance_key_memory = tolerance_key_memory[offset : offset + batch_size]
#             batch_tolerance_key_memory = batch_tolerance_key_memory.to(device)


#             tolerance = batch_tolerance_key_memory[None, :, :] + tolerance_query_vector[:, None, :]
#             distance_group = torch.abs(group_query_vector[:, None, :] - group_key_memory[None, :, :])
#             distance_group = torch.where(distance_group <= (N / 2), distance_group, N - distance_group)

#             #No encoding no saturation
#             # max_tolerance = torch.maximum(batch_tolerance_key_memory[None, :, :], tolerance_query_vector[:, None, :])
#             # max_distance = args.N // 2 - 2 * max_tolerance

#             distance_group_pos = torch.where(distance_group < tolerance, 0, distance_group - tolerance)
#             positive_distance = torch.sum(distance_group_pos, dim=2)

#             similarity_forward_pos_neg = -positive_distance


#             similarity[:, offset : offset + batch_size] = similarity_forward_pos_neg
#             off