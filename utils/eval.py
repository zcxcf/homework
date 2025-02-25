import torch
from models.ffn_only.mat_vit import generate_mask
import wandb


def eval(model, valDataLoader, width, num_active, epoch, model_name, args):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (img, label) in enumerate(valDataLoader):
            img = img.to(args.device)
            label = label.to(args.device)
            mask = generate_mask(width * 4, num_active)
            mask = mask.to(args.device)
            preds = model(img, mask)
            _, predicted = torch.max(preds, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        avg_loss = total_loss / len(valDataLoader)
        accuracy = 100.0 * correct / total
        print("val loss", avg_loss)
        print("val acc", accuracy)
        wandb.log({"Epoch": epoch + 1, model_name+"Val Loss": avg_loss})
        wandb.log({"Epoch": epoch + 1, model_name+"Val Acc": accuracy})

