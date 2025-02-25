import torch
from tqdm import tqdm
import wandb


def eval_mat(model, valDataLoader, criterion, epoch, optimizer, args, flag):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork(flag=flag)
                preds = model(img)

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "Val Loss_"+flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "Val Acc_"+flag: accuracy})

            pbar.close()

def eval_mat_mlp_mha(model, valDataLoader, criterion, epoch, optimizer, args, flag):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork(flag=flag)
                preds = model(img)

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "mlp_Val Loss_"+flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "mlp_Val Acc_"+flag: accuracy})

            pbar.close()

def eval_mat_fined(model, valDataLoader, criterion, epoch, optimizer, args, flag):
    dict = {3:'s', 6:'m', 12:'l'}
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork_fined(flag)
                preds = model(img)

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            flag = dict[flag]
            wandb.log({"Epoch": epoch + 1, "Val Loss_"+flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "Val Acc_"+flag: accuracy})

            pbar.close()


def eval_mat_combined(model, valDataLoader, criterion, epoch, optimizer, args, flag, **kwargs):

    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork(**kwargs)
                preds = model(img)

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "Val Acc_" + flag: accuracy})

def eval_mat_combined_dis(model, valDataLoader, criterion, epoch, optimizer, args, flag, device, local_rank, **kwargs):

    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(device)
                label = label.to(device)

                model.module.configure_subnetwork(**kwargs)
                preds = model(img)

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)

            if local_rank==0:
                wandb.log({"Epoch": epoch + 1, "Val Loss_" + str(local_rank) + flag: avg_loss})
                wandb.log({"Epoch": epoch + 1, "Val Acc_" + str(local_rank) + flag: accuracy})

def eval_mat_dynamic(model, valDataLoader, criterion, epoch, optimizer, args, flag, sub_dim, depth_list, mlp_ratio,
                                           mha_head, latency):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_ce_loss = 0.0
            total_latency_loss = 0.0

            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork(sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio,
                                           mha_head=mha_head, latency=latency)

                preds, mask = model(img)
                # preds, cost = model(img)

                ce_loss = criterion(preds, label)

                latency_loss = torch.square(latency - mask)
                # latency_loss = (1-latency) * cost

                loss = ce_loss + latency_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_latency_loss += latency_loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            avg_ce_loss = total_ce_loss / len(valDataLoader)
            avg_latency_loss = total_latency_loss / len(valDataLoader)

            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "Val ce Loss_" + flag: avg_ce_loss})
            wandb.log({"Epoch": epoch + 1, "Val latency Loss_" + flag: avg_latency_loss})

            wandb.log({"Epoch": epoch + 1, "Val Acc_" + flag: accuracy})


def eval_mask(model, valDataLoader, criterion, epoch, optimizer, args, flag, sub_dim, depth_list, mlp_ratio,
                                           mha_head, latency):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_ce_loss = 0.0
            total_latency_loss = 0.0

            total_attn_mask = 0
            total_mlp_mask = 0

            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_subnetwork(sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio,
                                           mha_head=mha_head, latency=latency)

                preds, attn_mask, mlp_mask = model(img)
                # preds, cost = model(img)

                ce_loss = criterion(preds, label)

                latency_loss = torch.square(latency-attn_mask) + torch.square(latency-mlp_mask)

                # latency_loss = (1-latency) * cost

                loss = ce_loss + latency_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_latency_loss += latency_loss.item()

                total_attn_mask += attn_mask.item()
                total_mlp_mask += mlp_mask.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            avg_ce_loss = total_ce_loss / len(valDataLoader)
            avg_latency_loss = total_latency_loss / len(valDataLoader)

            val_attn_mask = total_attn_mask / len(valDataLoader)
            val_mlp_mask = total_mlp_mask / len(valDataLoader)

            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "Val ce Loss_" + flag: avg_ce_loss})
            wandb.log({"Epoch": epoch + 1, "Val latency Loss_" + flag: avg_latency_loss})

            wandb.log({"Epoch": epoch + 1, "Val Acc_" + flag: accuracy})

            wandb.log({"Epoch": epoch + 1, "Val attn mask_" + flag: val_attn_mask})
            wandb.log({"Epoch": epoch + 1, "Val mlp mask_" + flag: val_mlp_mask})
def eval_all(model, valDataLoader, criterion, epoch, optimizer, args, flag, latency):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_ce_loss = 0.0
            total_latency_loss = 0.0

            total_attn_mask = 0
            total_mlp_mask = 0
            total_embed_mask = 0

            total_depth_mlp_mask = 0
            total_depth_attn_mask = 0

            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_latency(latency=latency)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask = model(img)

                # preds, cost = model(img)

                ce_loss = criterion(preds, label)

                latency_loss = torch.square(latency - attn_mask) + torch.square(latency - mlp_mask) + torch.square(
                    latency - embed_mask) + torch.square(latency - depth_mlp_mask) + torch.square(
                    latency - depth_attn_mask)

                # latency_loss = (1-latency) * cost

                loss = ce_loss + latency_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_latency_loss += latency_loss.item()

                total_attn_mask += attn_mask.item()
                total_mlp_mask += mlp_mask.item()
                total_embed_mask += embed_mask.item()
                total_depth_mlp_mask += depth_mlp_mask.item()
                total_depth_attn_mask += depth_attn_mask.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            avg_ce_loss = total_ce_loss / len(valDataLoader)
            avg_latency_loss = total_latency_loss / len(valDataLoader)

            val_attn_mask = total_attn_mask / len(valDataLoader)
            val_mlp_mask = total_mlp_mask / len(valDataLoader)
            val_embed_mask = total_embed_mask / len(valDataLoader)
            val_depth_mlp_mask = total_depth_mlp_mask / len(valDataLoader)
            val_depth_attn_mask = total_depth_attn_mask / len(valDataLoader)

            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)
            wandb.log({"Epoch": epoch + 1, "Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "Val ce Loss_" + flag: avg_ce_loss})
            wandb.log({"Epoch": epoch + 1, "Val latency Loss_" + flag: avg_latency_loss})

            wandb.log({"Epoch": epoch + 1, "Val Acc_" + flag: accuracy})

            wandb.log({"Epoch": epoch + 1, "Val attn mask_" + flag: val_attn_mask})
            wandb.log({"Epoch": epoch + 1, "Val mlp mask_" + flag: val_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "Val embed mask_" + flag: val_embed_mask})

            wandb.log({"Epoch": epoch + 1, "Val depth mlp mask_" + flag: val_depth_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "Val depth attn mask_" + flag: val_depth_attn_mask})


def eval_macs_loss(model, valDataLoader, criterion, epoch, optimizer, args, flag, latency):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_ce_loss = 0.0
            total_latency_loss = 0.0

            total_attn_mask = 0
            total_mlp_mask = 0
            total_embed_mask = 0

            total_depth_mlp_mask = 0
            total_depth_attn_mask = 0

            total_macs_sum = 0

            correct = 0
            total = 0

            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                model.configure_latency(latency=latency)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)

                # preds, cost = model(img)

                ce_loss = criterion(preds, label)

                latency_loss = torch.square(latency-total_macs)

                loss = ce_loss + latency_loss

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_latency_loss += latency_loss.item()

                total_attn_mask += attn_mask.item()
                total_mlp_mask += mlp_mask.item()
                total_embed_mask += embed_mask.item()
                total_depth_mlp_mask += depth_mlp_mask.item()
                total_depth_attn_mask += depth_attn_mask.item()

                total_macs_sum += total_macs.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            avg_ce_loss = total_ce_loss / len(valDataLoader)
            avg_latency_loss = total_latency_loss / len(valDataLoader)

            val_attn_mask = total_attn_mask / len(valDataLoader)
            val_mlp_mask = total_mlp_mask / len(valDataLoader)
            val_embed_mask = total_embed_mask / len(valDataLoader)
            val_depth_mlp_mask = total_depth_mlp_mask / len(valDataLoader)
            val_depth_attn_mask = total_depth_attn_mask / len(valDataLoader)
            val_macs = total_macs_sum / len(valDataLoader)

            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)

            wandb.log({"Epoch": epoch + 1, "Val Loss_" + flag: avg_loss})
            wandb.log({"Epoch": epoch + 1, "Val ce Loss_" + flag: avg_ce_loss})
            wandb.log({"Epoch": epoch + 1, "Val latency Loss_" + flag: avg_latency_loss})

            wandb.log({"Epoch": epoch + 1, "Val Acc_" + flag: accuracy})

            wandb.log({"Epoch": epoch + 1, "Val attn mask_" + flag: val_attn_mask})
            wandb.log({"Epoch": epoch + 1, "Val mlp mask_" + flag: val_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "Val embed mask_" + flag: val_embed_mask})

            wandb.log({"Epoch": epoch + 1, "Val depth mlp mask_" + flag: val_depth_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "Val depth attn mask_" + flag: val_depth_attn_mask})

            wandb.log({"Epoch": epoch + 1, "Val macs_" + flag: val_macs})

            wandb.log({"Epoch": epoch + 1, "Val acc/macs rate_" + flag: accuracy/val_macs})




