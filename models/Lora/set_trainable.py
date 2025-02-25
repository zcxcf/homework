
def set_para(model, type):
    for name, param in model.named_parameters():
        if 'blocks' in name:
            param.requires_grad = False

        if "A" in name or 'B' in name or 'head' in name:
            param.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")