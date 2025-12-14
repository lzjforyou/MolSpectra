import pickle
import torch
import torch.nn.functional as F
import numpy as np

import os
import copy
import pandas as pd
import tempfile
from pprint import pprint
from datetime import datetime

from dataset import BaseDataset, data_to_device, get_dset_types,LMDataset
from model import Predictor
from misc_utils import th_temp_seed, count_parameters, get_pbar, get_scaler
from losses import get_loss_func, get_sim_func
from spec_utils import process_spec, unprocess_spec, merge_spec
from data_utils import ELEMENT_LIST
from tqdm import tqdm
import time

def write_log(log_file, message):
    """Write to log file"""
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def find_pretrained_model_path(
    config_file_path,
    data_d,
    run_d,
    model_d,
    weight_filename="best_model_state.pth",
    debug=True,
    raise_on_fail=False
):
    """
    Search for pretrained model path under base directory,
    strictly matching directory name prefix:
        run_{config_filename}_{split_key}_{split_seed}_
    If multiple candidates exist, select the newest by weight file modification time.
    """
    def log(*a):
        if debug:
            print(*a)

    # 1. Parse fields
    config_filename = os.path.splitext(os.path.basename(config_file_path))[0]

    split_seed = run_d.get("split_seed", "NA")
    split_key = run_d.get("split_key","NA")
    embed_types = model_d.get("embed_types", ["default"])[0]
    # Keep original str format: might be "['fp']"
    et_str = str(embed_types)

    # 2. Base directory (only use this one)
    base_root = "YOUR_BASE_DIR/output"  # Replace with your actual path

    if not os.path.isdir(base_root):
        msg = f"[find_pretrained_model_path] Base directory does not exist: {base_root}"
        log(msg)
        if raise_on_fail:
            raise FileNotFoundError(msg)
        return None

    # 3. Construct strict prefix
    required_prefix = f"run_{config_filename}_{split_key}_{split_seed}_"
    log(f"Strict match prefix: {required_prefix}")
    log(f"Search directory: {base_root}")

    # 4. Traverse first level of root_dir
    candidates = []
    try:
        for name in os.listdir(base_root):
            dir_path = os.path.join(base_root, name)
            if not os.path.isdir(dir_path):
                continue
            if not name.startswith(required_prefix):
                continue
            weight_path = os.path.join(dir_path, weight_filename)
            if not os.path.isfile(weight_path):
                continue
            try:
                mtime = os.path.getmtime(weight_path)
            except Exception:
                mtime = 0.0
            candidates.append({
                "dir": dir_path,
                "weight": weight_path,
                "mtime": mtime
            })
    except PermissionError:
        log(f"[Warning] No permission to access: {base_root}")
    except FileNotFoundError:
        log(f"[Warning] Directory does not exist: {base_root}")

    if not candidates:
        msg = "✗ No pretrained weights found matching strict prefix."
        log(msg)
        if raise_on_fail:
            raise FileNotFoundError(msg + f" Search directory: {base_root}")
        return None

    # 5. Sort by modification time (newest → oldest)
    candidates.sort(key=lambda x: (x["mtime"], x["dir"]), reverse=True)

    log("=== Candidates (sorted by modification time descending, top 10) ===")
    for i, c in enumerate(candidates[:10]):
        log(f"[{i}] mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c['mtime']))}  {c['weight']}")

    selected = candidates[0]
    log(f"✓ Selected: {selected['weight']}")
    return selected["weight"]

def run_train_epoch(step, epoch, model,dl_d, data_d, run_d,optimizer,scheduler, log_file):
    
    dev = torch.device(run_d["device"])
    nb = run_d["non_blocking"] 
    loss_func = get_loss_func(run_d["loss"],data_d["mz_bin_res"],agg=run_d["batch_loss_agg"])
    b_losses = []
    scaler = get_scaler(run_d["amp"]) 

    model.train()
    for b_idx, b in get_pbar(enumerate(dl_d["train"]), run_d["log_tqdm"], desc="> train", total=len(dl_d["train"])): 
        optimizer.zero_grad()
        b = data_to_device(b, dev, nb) 
        b_output = model(data=b, amp=run_d["amp"]) 
        b_pred = b_output["pred"]
        b_targ = b["spec"]
        b_loss_agg = loss_func(b_pred, b_targ)
        scaler.scale(b_loss_agg / run_d["grad_acc_interval"]).backward() 

        if step % run_d["grad_acc_interval"] == 0:
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), run_d["clip_grad_norm"])
            scaler.step(optimizer) 
            scaler.update() 
            if run_d["scheduler"] == "polynomial" or run_d["scheduler"] == "cosine":
                scheduler.step()

        step += 1
        b_losses.append(b_loss_agg.detach().to("cpu").item())

    optimizer.zero_grad()
    train_spec_loss = np.mean(b_losses)
    # Add log recording
    log_message = f"TRAIN - Epoch: {epoch}, Loss: {train_spec_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.8f}"
    write_log(log_file, log_message)
    print(f"epoch:{epoch},train_spec_loss:",{train_spec_loss})
   
    return step, epoch, {}




def run_val(step, epoch, model, dl_d, data_d, run_d, log_file):
    if not (dl_d["primary"]["val"] is None):
        dev = torch.device(run_d["device"])
        nb = run_d["non_blocking"]
        model.eval()
        pred, targ, mol_id, group_id = [], [], [], []
        with torch.no_grad():
            for b_idx, b in get_pbar(enumerate(dl_d["primary"]["val"]), run_d["log_tqdm"], desc="> val", total=len(dl_d["primary"]["val"])):
                b = data_to_device(b, dev, nb)
                b_pred = model(data=b, amp=run_d["amp"])["pred"]
                b_targ = b["spec"]
                pred.append(b_pred.detach().to("cpu", non_blocking=nb))
                targ.append(b_targ.detach().to("cpu", non_blocking=nb))
        pred = torch.cat(pred, dim=0)
        targ = torch.cat(targ, dim=0)

        # Original sim
        sin_func = get_sim_func(run_d["sim"], data_d["mz_bin_res"])
        all_sim = sin_func(pred, targ)
        mean_sim = torch.mean(all_sim, dim=0).item()
        print(f"> val sim: {mean_sim:.4f}")
        # Add log recording
        log_message = f"VAL - Epoch: {epoch}, Sim: {mean_sim:.6f}"
        write_log(log_file, log_message)

        out_d = {run_d["stop_key"]: mean_sim}
    else:
        out_d = {run_d["stop_key"]: np.nan}
    return step, epoch, out_d


def run_track(step, epoch, model, dl_d, data_d, run_d, ds):
    """
    Track and record prediction results for train, validation, and test sets
    
    Parameters:
        step: Current training step
        epoch: Current epoch
        model: Trained model
        dl_d: DataLoader dictionary
        data_d: Data configuration dictionary
        run_d: Run configuration dictionary
        ds: Dataset object
    """
    dev = torch.device(run_d["device"])
    nb = run_d["non_blocking"]
    loss_func = get_loss_func(run_d["loss"], data_d["mz_bin_res"]) 
    sim_func = get_sim_func(run_d["sim"], data_d["mz_bin_res"])

    if run_d["save_media"] and run_d["num_track"] > 0:
        
        model.to(dev)
        model.eval()
        
        # Create save directory
        base_save_dir = "YOUR_RESULTS_DIR/track_attn"  # Replace with your actual path
        os.makedirs(base_save_dir, exist_ok=True)
        
        # Get dataset name and split method
        primary_dset = data_d["primary_dset"]
        if isinstance(primary_dset, list):
            primary_dset = primary_dset[0]
        split_key = run_d.get("split_key", "unknown")
        
        # Construct filename: dataset_splitmethod_epoch{epoch}_step{step}
        csv_filename = f"{primary_dset}_{split_key}_epoch{epoch}_step{step}.csv"
        pkl_filename = f"{primary_dset}_{split_key}_epoch{epoch}_step{step}.pkl"
        csv_filepath = os.path.join(base_save_dir, csv_filename)
        pkl_filepath = os.path.join(base_save_dir, pkl_filename)
        
        print(f"\n{'='*60}")
        print(f"Start recording tracking data (Epoch {epoch}, Step {step})")
        print(f"CSV save path: {csv_filepath}")
        print(f"PKL save path: {pkl_filepath}")
        print(f"{'='*60}\n")
        
        # Lists to store all results
        all_results = []  # For CSV
        all_data = []     # For PKL
        
        # Iterate through train, val, test sets
        split_names = ["train", "val", "test"]
        
        with torch.no_grad():
            for split_name in split_names:
                # Check if this split exists
                if dl_d["primary"][split_name] is None:
                    print(f"⚠️  {split_name.upper()} set does not exist, skipping")
                    continue
                
                # Get indices for this split
                split_idx = dl_d["primary"][split_name].dataset.indices
                print(f"\n📊 Processing {split_name.upper()} set:")
                print(f"   Number of samples: {len(split_idx)}")
                
                # Get DataLoader for this split (get all samples)
                track_dl = ds.get_track_dl(split_idx, num_rand_idx=len(split_idx))
                
                for dl_type_key, dl in track_dl.items():
                    print(f"   Processing {dl_type_key}, number of batches: {len(dl)}")
                    
                    for d_idx, d in enumerate(dl):
                        # Print progress every 100 batches
                        if d_idx % 100 == 0 and d_idx > 0:
                            print(f"      Progress: {d_idx}/{len(dl)}")
                        
                        # Move data to device
                        d = data_to_device(d, dev, nb)
                        
                        # Model prediction
                        pred = model(data=d, amp=run_d["amp"])["pred"]
                        targ = d["spec"]
                        
                        # Calculate similarity (may be batch-wise)
                        sim_values = sim_func(pred, targ)
                        
                        # Process SMILES (may be batch-wise)
                        smiles_list = d["smiles"]
                        if not isinstance(smiles_list, (list, tuple)):
                            smiles_list = [smiles_list]
                        
                        # Convert predictions and targets to numpy arrays
                        pred_numpy = pred.cpu().numpy() if hasattr(pred, 'cpu') else np.array(pred)
                        targ_numpy = targ.cpu().numpy() if hasattr(targ, 'cpu') else np.array(targ)
                        
                        # Ensure sim_values is one-dimensional
                        if hasattr(sim_values, 'shape') and len(sim_values.shape) == 0:
                            # Scalar, convert to list
                            sim_values = [sim_values.item()]
                        elif hasattr(sim_values, 'cpu'):
                            # Tensor, convert to list
                            sim_values = sim_values.cpu().numpy().flatten().tolist()
                        elif not isinstance(sim_values, (list, tuple)):
                            sim_values = [float(sim_values)]
                        
                        # Batch process each sample
                        batch_size = len(smiles_list)
                        for i in range(batch_size):
                            smiles = smiles_list[i] if i < len(smiles_list) else "unknown"
                            sim_scalar = sim_values[i] if i < len(sim_values) else sim_values[0]
                            
                            # Get prediction and target for this sample
                            pred_i = pred_numpy[i] if pred_numpy.shape[0] > 1 else pred_numpy[0]
                            targ_i = targ_numpy[i] if targ_numpy.shape[0] > 1 else targ_numpy[0]
                            
                            # Add to CSV results list
                            all_results.append({
                                "smiles": smiles,
                                "dataset_split": split_name,  # train/val/test
                                "similarity": float(sim_scalar),
                                "epoch": epoch,
                                "step": step,
                                "split_method": split_key,
                                "dataset_name": primary_dset
                            })
                            
                            # Add to PKL results list (includes predictions and targets)
                            all_data.append({
                                "smiles": smiles,
                                "dataset_split": split_name,
                                "similarity": float(sim_scalar),
                                "epoch": epoch,
                                "step": step,
                                "split_method": split_key,
                                "dataset_name": primary_dset,
                                "prediction": pred_i,
                                "target": targ_i
                            })
                    
                    print(f"   ✓ {split_name.upper()} set processing complete")
        
        # 1. Save CSV file
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values(by=["dataset_split", "similarity"], ascending=[True, False])
        df_results.to_csv(csv_filepath, index=False, encoding='utf-8')
        
        # 2. Save PKL file
        with open(pkl_filepath, 'wb') as f:
            pickle.dump(all_data, f)
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"✅ Tracking data saved successfully!")
        print(f"📁 CSV file path: {csv_filepath}")
        print(f"📁 PKL file path: {pkl_filepath}")
        print(f"📊 Statistics:")
        print(f"   Total samples: {len(df_results)}")
        
        # Statistics for each dataset separately
        for split_name in split_names:
            split_df = df_results[df_results["dataset_split"] == split_name]
            if len(split_df) > 0:
                split_mean_sim = split_df["similarity"].mean()
                split_std_sim = split_df["similarity"].std()
                split_min_sim = split_df["similarity"].min()
                split_max_sim = split_df["similarity"].max()
                print(f"\n   {split_name.upper()} set: {len(split_df)} samples")
                print(f"      Average similarity: {split_mean_sim:.4f} ± {split_std_sim:.4f}")
                print(f"      Similarity range: [{split_min_sim:.4f}, {split_max_sim:.4f}]")
        
        print(f"\n   Overall average similarity: {df_results['similarity'].mean():.4f}")
        print(f"   Overall similarity range: [{df_results['similarity'].min():.4f}, {df_results['similarity'].max():.4f}]")
        
        # Print PKL file size
        pkl_size_mb = os.path.getsize(pkl_filepath) / (1024 * 1024)
        print(f"   PKL file size: {pkl_size_mb:.2f} MB")
        print(f"{'='*60}\n")

    return step, epoch, {}




def run_test(step,epoch,model,dl_d,data_d,model_d,run_d,run_dir, log_file, test_sets=None):

    if test_sets is None:
        test_sets = ["test"]
    if run_d["do_test"]:

        dev = torch.device(run_d["device"])
        nb = run_d["non_blocking"]
        print(">> test")

        model.to(dev)
        model.eval()
        out_d = {}
        for order in ["primary"]:
            out_d[order] = {}
            for dl_key, dl in dl_d[order].items():
                if not (dl_key in test_sets) or dl is None: 
                    continue
                pred, targ = [], []
                with torch.no_grad():
                    for b_idx, b in get_pbar(enumerate(dl), run_d["log_tqdm"], desc=f"> {dl_key}", total=len(dl)):
                        b = data_to_device(b, dev, nb)
                        b_pred = model(data=b, amp=run_d["amp"])["pred"] 
                        b_targ = b["spec"]
                        pred.append(b_pred.detach().to("cpu", non_blocking=nb))
                        targ.append(b_targ.detach().to("cpu", non_blocking=nb))
                pred = torch.cat(pred, dim=0)
                targ = torch.cat(targ, dim=0)
                sin_func = get_sim_func(run_d["sim"], data_d["mz_bin_res"])
                all_sim = sin_func(pred, targ)  
                mean_sim = torch.mean(all_sim, dim=0).item()
                # Add log recording
                log_message = f"TEST - Dataset: {dl_key}, Sim: {mean_sim:.6f}"
                write_log(log_file, log_message)
                print(f"> {dl_key} sim: {mean_sim:.4f}")
    else:
        out_d = {}
    return step, epoch, out_d


def rank_metrics(rank, total):

    d = {}
    d["rank"] = float(rank)
    d["top01"] = float(rank == 1)
    d["top05"] = float(rank <= 5)
    d["top10"] = float(rank <= 10)
    norm_rank = float((rank - 1) / total)
    d["norm_rank"] = norm_rank
    d["top01%"] = float(norm_rank <= 0.01) 
    d["top05%"] = float(norm_rank <= 0.05)
    d["top10%"] = float(norm_rank <= 0.10) 
    d["total"] = total
    return d


def sims_to_rank_metrics(sim, sim2, key_prefix, cand_match_mask):
   
    rm_d = {}
    key = f"{key_prefix}"
    cand_match_idx = torch.argmax(cand_match_mask.float()) 
    rm_d[f"{key}_sim"] = sim[cand_match_idx].item() 
    noisey_sim = sim + 0.00001 * torch.rand_like(sim) 
    sim_argsorted = torch.argsort(-noisey_sim, dim=0)
    rank = torch.argmax(cand_match_mask.float()[sim_argsorted]) + 1
    _rm_d = rank_metrics(rank, cand_match_mask.shape[0]) 
    rm_d.update({f"{key}_{k}":v for k,v in _rm_d.items()})

    rm_d[f"{key}_sim2"] = sim2[cand_match_idx].item() 
    noisey_sim2 = sim2 + 0.00001 * torch.rand_like(sim2) 
    sim2_argsorted = torch.argsort(-noisey_sim2, dim=0) 

    num_20p = int(np.round(0.2*sim2_argsorted.shape[0])) 
    sim2_t20p = sim2_argsorted[:num_20p] 
    if cand_match_idx not in sim2_t20p: 
        sim2_t20p = torch.cat([cand_match_idx.reshape(1,),sim2_t20p[1:]],dim=0)
    sim_t20p_argsorted = torch.argsort(-noisey_sim[sim2_t20p], dim=0)
    rank_t20p = torch.argmax(cand_match_mask.float()[sim2_t20p][sim_t20p_argsorted]) + 1
    total_t20p = sim_t20p_argsorted.shape[0] 
    _rm_d = rank_metrics(rank_t20p, total_t20p) 
    rm_d.update({f"{key}_t20p_{k}":v for k,v in _rm_d.items()})

    sim2_b20p = sim2_argsorted[-num_20p:]
    if cand_match_idx not in sim2_b20p:
        sim2_b20p = torch.cat([cand_match_idx.reshape(1,),sim2_b20p[1:]],dim=0)
    sim_b20p_argsorted = torch.argsort(-noisey_sim[sim2_b20p], dim=0)
    rank_b20p = torch.argmax(cand_match_mask.float()[sim2_b20p][sim_b20p_argsorted]) + 1
    total_b20p = sim_b20p_argsorted.shape[0]
    _rm_d = rank_metrics(rank_b20p, total_b20p)
    rm_d.update({f"{key}_b20p_{k}":v for k,v in _rm_d.items()})
    return rm_d


def run_lm(step, epoch, model, lm_ds, lm_type, data_d, model_d, run_d, use_wandb, run_dir, mr_d, update_mr_d, log_file):
    if run_d[f"do_{lm_type}"]:
        lm_d = mr_d[f"{lm_type}_d"]
        print(f">> {lm_type}")

        dev = torch.device(run_d["device"])
        nb = run_d["non_blocking"]
        model.to(dev)
        model.eval()
        model.set_mode(lm_type)

        # Get all samples
        spec_dl = lm_ds.get_dataloader(run_d, "spec")
        spec, spec_spec_id, spec_mol_id, spec_casmi_fp = [], [], [], []
        for b_idx, b in get_pbar(enumerate(spec_dl), run_d["log_tqdm"], desc=f"> spec", total=len(spec_dl)):
            spec.append(b["spec"])
            spec_spec_id.append(b["spec_id"])
            spec_mol_id.append(b["mol_id"])
            spec_casmi_fp.append(b["lm_fp"])
        spec = torch.cat(spec, dim=0)
        spec_spec_id = torch.cat(spec_spec_id, dim=0)
        spec_mol_id = torch.cat(spec_mol_id, dim=0)
        spec_casmi_fp = torch.cat(spec_casmi_fp, dim=0)

        # Get all candidates
        group_dl = lm_ds.get_dataloader(run_d, "group")  # Actually all candidates
        cand_pred, cand_mol_id, cand_spec_id, cand_casmi_fp = [], [], [], []
        with torch.no_grad():
            for b_idx, b in get_pbar(enumerate(group_dl), run_d["log_tqdm"], desc=f"> all candidates", total=len(group_dl)):
                b = data_to_device(b, dev, nb)
                cand_pred.append(b["pred"].cpu() if "pred" in b else model(data=b, amp=run_d["amp"])["pred"].cpu())
                cand_mol_id.append(b["mol_id"].cpu())
                cand_spec_id.append(b["spec_id"].cpu())
                cand_casmi_fp.append(b["lm_fp"].cpu())
        cand_pred = torch.cat(cand_pred, dim=0)
        cand_mol_id = torch.cat(cand_mol_id, dim=0)
        cand_spec_id = torch.cat(cand_spec_id, dim=0)
        cand_casmi_fp = torch.cat(cand_casmi_fp, dim=0)

        # Check all real mol_ids are in candidates
        assert torch.isin(spec_mol_id, cand_mol_id).all()

        # Evaluate each sample
        rm_ds, sims, sims2 = lm_d["rm_ds"], lm_d["sims"], lm_d["sims2"]
        sim_func = get_sim_func(run_d["sim"], data_d["mz_bin_res"])
        fp_sim_func = get_sim_func("jacc", None)
        for i in tqdm(range(spec.shape[0]), desc="LM evaluation", total=spec.shape[0]):
            query_mol_id = int(spec_mol_id[i])
            cand_match_mask = (cand_mol_id == query_mol_id)
            assert cand_match_mask.sum() == 1, f"mol_id {query_mol_id} not unique in candidates"

            cand_spec = cand_pred
            targ_spec = spec[i].unsqueeze(0).expand(cand_spec.shape[0], -1)
            sim_obj = sim_func(cand_spec, targ_spec)

            cand_fp = cand_casmi_fp
            targ_fp = spec_casmi_fp[i].unsqueeze(0).expand(cand_fp.shape[0], -1)
            sim_fp = fp_sim_func(cand_fp, targ_fp)

            if run_d.get("casmi_save_sim", False):
                sims.append(sim_obj)
                sims2.append(sim_fp)
            rm_d = sims_to_rank_metrics(sim_obj, sim_fp, lm_type, cand_match_mask)
            rm_ds.append(rm_d)
            update_mr_d(mr_d, **{f"{lm_type}_d": lm_d})

        # Statistical metrics
        rm_d = {k: np.array([d[k] for d in rm_ds]) for k in rm_ds[0]}
        log_dict = {k: np.mean(v) for k, v in rm_d.items()}
        # New: write to log
        metrics_str = ", ".join([f"{k}={v:.6f}" for k, v in log_dict.items()])
        write_log(log_file, f"LM - Epoch: {epoch}, Type: {lm_type}, {metrics_str}")
        if run_d.get("print_stats", False):
            pprint(log_dict)
    return step, epoch


def get_ds_model(data_d, model_d, run_d):

    with th_temp_seed(model_d["model_seed"]):
        embed_types = model_d["embed_types"]
        dset_types = get_dset_types(embed_types)
        assert len(dset_types) > 0, dset_types
        # Note that the data preprocessing logic for infrared spectroscopy and EI-MS is different.
        ds = BaseDataset(*dset_types, **data_d)
        dim_d = ds.get_data_dims()
    
        model = Predictor(dim_d, **model_d)
        if run_d["do_lm"]:
            lm_ds = LMDataset(ds, *dset_types, **data_d)
        else:
            lm_ds = None
    dev = torch.device(run_d["device"])
    model.to(dev)
    return ds, model,lm_ds

def init_lm_d():
    d = {}
    d["query_group_ids"] = set()
    for k in ["rm_ds","um_rm_ds","sims","sims2","group_ids"]:
        d[k] = []
    return d


def train_and_eval(data_d, model_d, run_d, config_file_path):

    base_dir = data_d["base_dir"]
    timestamp = datetime.now().strftime("%m-%d-%H")  
    custom_name = run_d["custom_name"]
    split_key = run_d["split_key"]
    # Modified to
    split_seed = run_d.get("split_seed", 42)  # Get split_seed, default value is 42
    run_dir = os.path.join(base_dir, f"run_{custom_name}_{split_key}_{split_seed}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    # Create log file
    log_file = os.path.join(run_dir, "log.txt")
    # Record initial configuration information
    config_info = f"CONFIG - Custom name: {custom_name}, Split seed: {split_seed}, Train seed: {run_d['train_seed']}"
    write_log(log_file, config_info)
    write_log(log_file, f"CONFIG - Device: {run_d['device']}, Optimizer: {run_d['optimizer']}, Scheduler: {run_d['scheduler']}")
    # set seeds
    torch.manual_seed(run_d["train_seed"])
    np.random.seed(run_d["train_seed"] // 2)

    ds, model,lm_ds = get_ds_model(data_d, model_d, run_d)
    dl_d, split_id_d = ds.get_dataloaders(run_d) 

    if run_d["optimizer"] == "adam":
        optimizer_fn = torch.optim.Adam
    elif run_d["optimizer"] == "adamw":
        optimizer_fn = torch.optim.AdamW
    else:
        raise NotImplementedError
    optimizer = optimizer_fn(model.parameters(),lr=run_d["learning_rate"],weight_decay=run_d["weight_decay"])

    if run_d["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,run_d["scheduler_period"],gamma=run_d["scheduler_ratio"])
    elif run_d["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",patience=run_d["scheduler_period"],factor=run_d["scheduler_ratio"])
    elif run_d["scheduler"] == "cosine":
        if dl_d["primary"]["train"] is None:
            num_batches = 0
        else:
            num_batches = len(dl_d["primary"]["train"])
        tot_updates = run_d["num_epochs"] * (num_batches // run_d["grad_acc_interval"]) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=tot_updates,
            eta_min=run_d.get("scheduler_end_lr", 0)
        )
    elif run_d["scheduler"] == "none":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1,gamma=1.0)
    else:
        raise NotImplementedError

    if run_d["pretrained"] is not None:
        state_dict_path = run_d["pretrained"]
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f">>> successfully loaded model from state_dict: {state_dict_path}")

    best_val_sim_mean = -np.inf
    best_epoch = -1 
    best_state_dict = copy.deepcopy(model.state_dict())
    early_stop_count = 0 
    early_stop_thresh = run_d["early_stop_thresh"] 
    step = 0 
    epoch = -1 
    lm_d = init_lm_d()
    dev = torch.device(run_d["device"])

    mr_fp = os.path.join(run_dir, "chkpt.pkl") 
    temp_mr_fp = os.path.join(run_dir, "temp_chkpt.pkl") 
    split_id_fp = os.path.join(run_dir, "split_id.pkl")
    if os.path.isfile(mr_fp):
        print(">>> reloading model from most recent checkpoint")
        mr_d = torch.load(mr_fp,map_location="cpu")
        model.load_state_dict(mr_d["mr_model_sd"])
        best_state_dict = copy.deepcopy(model.state_dict()) 
        optimizer.load_state_dict(mr_d["optimizer_sd"]) 
        scheduler.load_state_dict(mr_d["scheduler_sd"]) 
        best_epoch = mr_d["best_epoch"]
        early_stop_count = mr_d["early_stop_count"]
        step = mr_d["step"]
        epoch = mr_d["epoch"]
        lm_d = mr_d["lm_d"]
    else:
        print(">>> no checkpoint detected")
        mr_d = { 
            "mr_model_sd": model.state_dict(),
            "best_model_sd": best_state_dict,
            "optimizer_sd": optimizer.state_dict(),
            "scheduler_sd": scheduler.state_dict(),
            "best_epoch": best_epoch,
            "early_stop_count": early_stop_count,
            "step": step,
            "epoch": epoch,
            "test": False,
            "lm": False,
            "lm_d": lm_d,
        }
        if run_d["save_split"]:
            torch.save(split_id_d, split_id_fp)

        if run_d["save_state"]:
            torch.save(mr_d,temp_mr_fp) 
            os.replace(temp_mr_fp,mr_fp)

    model.to(dev)

    epoch += 1

    while epoch < run_d["num_epochs"]:

        print(f">>> start epoch {epoch}")
        
        step, epoch, _ = run_train_epoch(step, epoch, model, dl_d, data_d, run_d, optimizer, scheduler, log_file)
        
        step, epoch, val_d = run_val(step, epoch, model, dl_d, data_d, run_d, log_file)

        if run_d["scheduler"] == "step":
            scheduler.step()
        elif run_d["scheduler"] == "plateau":
            scheduler.step(val_d[run_d["stop_key"]])

        val_sim_mean = val_d[run_d["stop_key"]] 
        if best_val_sim_mean == -np.inf: 
            print(f"> val sim delta: N/A")
        else:
            print(f"> val sim delta: {val_sim_mean-best_val_sim_mean}") 
        if run_d["use_val_info"]:
            if best_val_sim_mean > val_sim_mean: 
                early_stop_count += 1
                print( 
                    f"> val sim DID NOT decrease, early stop count at {early_stop_count}/{early_stop_thresh}")
            else:
                best_val_sim_mean = val_sim_mean

                best_epoch = epoch
                early_stop_count = 0 
             
                model.to("cpu")
                best_state_dict = copy.deepcopy(model.state_dict())
             
                save_path = os.path.join(run_dir, "best_model_state.pth")
              
                torch.save(best_state_dict, save_path)
                print(f"Model state dict saved to {save_path}")

                model.to(dev)
                print("> val loss DID decrease, early stop count reset")
            if early_stop_count == early_stop_thresh: 
                print("> early stopping NOW")
                break

        mr_d = {
            "mr_model_sd": model.state_dict(), 
            "best_model_sd": best_state_dict, 
            "optimizer_sd": optimizer.state_dict(),
            "scheduler_sd": scheduler.state_dict(),
            "best_epoch": best_epoch,
            "early_stop_count": early_stop_count,
            "step": step,
            "epoch": epoch,
            "test": False,
            "lm": False,
            "lm_d": lm_d
        }
        if run_d["save_state"]:
            torch.save(mr_d, temp_mr_fp)
            os.replace(temp_mr_fp,mr_fp)
     
        print(f"Completed epoch {epoch}")
        epoch += 1

    def update_mr_d(mr_d,**kwargs):

        for k, v in kwargs.items():
            mr_d[k] = v
        if run_d["save_state"]:
            torch.save(mr_d, temp_mr_fp)
            os.replace(temp_mr_fp, mr_fp)
           

    if not mr_d["test"]:
        model.load_state_dict(best_state_dict)
        # Final test call
        step, epoch, test_d = run_test(step, epoch, model, dl_d, data_d, model_d, run_d, run_dir, log_file, test_sets=run_d["test_sets"])
        update_mr_d(mr_d,test=True)
    if not mr_d["lm"]:
        model.load_state_dict(best_state_dict)
        step, epoch = run_lm(step, epoch, model, lm_ds, "lm", data_d, model_d, run_d, "False", run_dir, mr_d, update_mr_d, log_file)
        update_mr_d(mr_d,casmi=True,lm_d={})
        if run_d["do_track"] :
            run_track(step, epoch, model, dl_d, data_d, run_d, ds)

    mr_d = {
        "best_model_sd": best_state_dict,
        "best_epoch": best_epoch,
        "epoch": epoch,
        "step": step,
        "test": True,
    }
    if run_d["save_state"]:
        torch.save(mr_d, temp_mr_fp)
        os.replace(temp_mr_fp,mr_fp)

    if run_d["device"] != "cpu" and torch.cuda.is_available():
        cuda_max_memory = torch.cuda.max_memory_allocated(device=dev)/1e9
        print(f"> GPU memory: {cuda_max_memory:.2f} GB")

    return
