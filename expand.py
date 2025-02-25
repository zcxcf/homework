import torch
from models.history.bert2bert import generate_random_match, expand_copy, expand_fpi, expand_aki

def tiny_expand_small(check_t, check_s):
    for (key_s, value_s), (key_t, value_t) in zip(check_s.items(), check_t.items()):
        if key_s == 'patch_embed.proj.weight':
            check_t[key_s] = value_t.repeat_interleave(2, dim=0)
        elif 'blocks' in key_s and 'weight' in key_s and 'norm' not in key_s:
            check_t[key_s] = value_t.repeat_interleave(2, dim=0).repeat_interleave(2, dim=-1)
        elif 'blocks' in key_s and 'mlp.fc' in key_s and 'weight' in key_s:
            check_t[key_s] = value_t.repeat_interleave(2, dim=0).repeat_interleave(2, dim=-1)
        else:
            check_t[key_s] = value_t.repeat_interleave(2, dim=-1)

    return check_t

def tiny2small(checkpoint_tiny, checkpoint_small):
    for (key_s, value_s), (key_t, value_t) in zip(checkpoint_small.items(), checkpoint_tiny.items()):
        if key_s != key_t:
            if 'norm' in key_t:
                choose_num_dict = generate_random_match(192, 384)
                expand_t = expand_fpi(value_t, 384, 1, choose_num_dict, to_expand='row')
                checkpoint_tiny[key_t] = torch.squeeze(expand_t)
            if 'head.weight' in key_t:
                choose_num_dict = generate_random_match(192, 384)
                expand_t = expand_fpi(value_t, 100, 384, choose_num_dict, to_expand='col')
                checkpoint_tiny[key_t] = expand_t
            continue
        if 'cls_token' in key_s:
            choose_num_dict = generate_random_match(192, 384)
            expand_t = expand_fpi(value_t.squeeze(), 384, 1, choose_num_dict, to_expand='row')
            checkpoint_tiny[key_t] = torch.unsqueeze(torch.unsqueeze(expand_t, dim=0), dim=0)
        elif 'pos_embed' in key_s:
            choose_num_dict = generate_random_match(192, 384)
            expand_t = expand_fpi(value_t.squeeze(), 197, 384, choose_num_dict, to_expand='col')
            checkpoint_tiny[key_t] = torch.unsqueeze(expand_t, dim=0)

        elif 'patch_embed' in key_s:
            if 'weight' in key_s:
                choose_num_dict = generate_random_match(192, 384)
                expand_t = expand_fpi(value_t.reshape(192, 3 * 16 * 16), 384, 3 * 16 * 16, choose_num_dict,
                                      to_expand='row')
                checkpoint_tiny[key_t] = expand_t.reshape(384, 3, 16, 16)
            if 'bias' in key_s:
                row = value_t.shape[0]
                choose_num_dict = generate_random_match(row, row * 2)
                expand_t = expand_fpi(value_t, row * 2, 1, choose_num_dict, to_expand='row')
                checkpoint_tiny[key_t] = torch.squeeze(expand_t)
        elif 'norm' in key_s:
            choose_num_dict = generate_random_match(192, 384)
            expand_t = expand_fpi(value_t, 384, 1, choose_num_dict, to_expand='row')
            checkpoint_tiny[key_t] = torch.squeeze(expand_t)
        elif 'blocks' in key_s:
            if 'weight' in key_s:
                row = value_t.shape[0]
                col = value_t.shape[1]
                choose_num_dict = generate_random_match(col, col * 2)
                expand_t = expand_fpi(value_t, row, col * 2, choose_num_dict, to_expand='col')
                choose_num_dict = generate_random_match(row, row * 2)
                expand_t = expand_fpi(expand_t, row * 2, col * 2, choose_num_dict, to_expand='row')
                checkpoint_tiny[key_t] = expand_t
            elif 'bias' in key_s:
                row = value_t.shape[0]
                choose_num_dict = generate_random_match(row, row * 2)
                expand_t = expand_fpi(value_t, row * 2, 1, choose_num_dict, to_expand='row')
                checkpoint_tiny[key_t] = torch.squeeze(expand_t)
        else:
            print("finish")
    return checkpoint_tiny

def split_blocks(checkpoint):
    block_list = []
    for key, value in checkpoint.items():
        if 'blocks' in key:
            block_num = int(key.split('.')[1])
            if len(block_list) == block_num:
                block_list.append({})
            block_list[block_num][key] = value
    return block_list

def expand_embed(checkpoint_tiny, checkpoint_small, if_preserve_laynorm=False):
    temp_choose = generate_random_match(192, 384)

    if if_preserve_laynorm:
        for i in range(192):
            temp_choose[i + 192] = i

    for (key_t, value_t), (key_s, value_s) in zip(checkpoint_tiny.items(), checkpoint_small.items()):
        if 'cls_token' in key_s:
            row = value_t.shape[2]

            expand_t = expand_copy(value_t.squeeze(), row * 2, 1, temp_choose, to_expand='row')

            checkpoint_tiny[key_t] = torch.unsqueeze(torch.unsqueeze(expand_t, dim=0), dim=0)

        elif 'pos_embed' in key_s:
            expand_t = expand_copy(value_t.squeeze(), 197, 384, temp_choose, to_expand='col')
            checkpoint_tiny[key_t] = torch.unsqueeze(expand_t, dim=0)

        elif 'patch_embed' in key_s:
            if 'weight' in key_s:
                expand_t = expand_copy(value_t.reshape(192, 3 * 16 * 16), 384, 3 * 16 * 16, temp_choose,
                                       to_expand='row')
                checkpoint_tiny[key_t] = expand_t.reshape(384, 3, 16, 16)
            if 'bias' in key_s:
                row = value_t.shape[0]
                expand_t = expand_copy(value_t, row * 2, 1, temp_choose, to_expand='row')
                checkpoint_tiny[key_t] = torch.squeeze(expand_t)
    return temp_choose

def expand_fpi_block(checkpoint_tiny, checkpoint_small, block_t, block_s, initial_choose):
    temp_choose = initial_choose
    for (key_t, value_t), (key_s, value_s) in zip(block_t.items(), block_s.items()):
        if 'norm' in key_s:
            expand_t = expand_copy(value_t, 384, 1, temp_choose, to_expand='row')
            checkpoint_tiny[key_t] = torch.squeeze(expand_t)

        elif 'weight' in key_s:
            if 'qkv' in key_s:
                value_list = []
                for i in range(3):
                    temp_value = value_t[i * 192:(i + 1) * 192, :]
                    expand_t = expand_fpi(temp_value, 192, 192 * 2, temp_choose, to_expand='col')

                    temp_choose = generate_random_match(192, 384)

                    expand_t = expand_copy(expand_t, 192 * 2, 192 * 2, temp_choose, to_expand='row')
                    value_list.append(expand_t)
                expand_t = torch.Tensor()
                for v in value_list:
                    expand_t = torch.cat((expand_t, v), dim=0)
                checkpoint_tiny[key_t] = expand_t
            elif 'proj' in key_s:
                row = value_t.shape[0]
                col = value_t.shape[1]
                expand_t = expand_fpi(value_t, row, col * 2, temp_choose, to_expand='col')

                temp_choose = generate_random_match(192, 384)

                expand_t = expand_copy(expand_t, row * 2, col * 2, temp_choose, to_expand='row')
                checkpoint_tiny[key_t] = expand_t

            elif 'fc1' in key_s:
                row = value_t.shape[0]
                col = value_t.shape[1]
                expand_t = expand_fpi(value_t, row, col * 2, temp_choose, to_expand='col')

                temp_choose = generate_random_match(192 * 4, 384 * 4)

                expand_t = expand_copy(expand_t, row * 2, col * 2, temp_choose, to_expand='row')
                checkpoint_tiny[key_t] = expand_t
            elif 'fc2' in key_s:
                row = value_t.shape[0]
                col = value_t.shape[1]
                expand_t = expand_fpi(value_t, row, col * 2, temp_choose, to_expand='col')

                temp_choose = generate_random_match(192, 384)

                expand_t = expand_copy(expand_t, row * 2, col * 2, temp_choose, to_expand='row')
                checkpoint_tiny[key_t] = expand_t

        elif 'bias' in key_s:
            if 'qkv' in key_s:
                value_list = []
                for i in range(3):
                    temp_value = value_t[i * 192:(i + 1) * 192]
                    expand_t = expand_copy(temp_value, 192 * 2, 1, temp_choose, to_expand='row')
                    value_list.append(expand_t)
                expand_t = torch.Tensor()
                for v in value_list:
                    expand_t = torch.cat((expand_t, v), dim=0)
                checkpoint_tiny[key_t] = expand_t
            else:
                row = value_t.shape[0]
                expand_t = expand_copy(value_t, row * 2, 1, temp_choose, to_expand='row')
                checkpoint_tiny[key_t] = torch.squeeze(expand_t)

    return temp_choose

def expand_aki_block(checkpoint_tiny, checkpoint_small, block_t, block_s, initial_choose, block_next):
    temp_choose = initial_choose
    for (key_t, value_t), (key_s, value_s), (key_next, value_next)in zip(block_t.items(), block_s.items(), block_next.items()):
        if 'norm' in key_s:
            expand_t = expand_copy(value_t, 384, 1, temp_choose, to_expand='row')
            checkpoint_tiny[key_t] = torch.squeeze(expand_t)

        elif 'weight' in key_s:
            if 'qkv' in key_s:
                value_list = []
                for i in range(3):
                    temp_value = value_t[i * 192:(i + 1) * 192, :]
                    temp_value_next = value_next[i * 192:(i + 1) * 192, :]
                    expand_t = expand_aki(temp_value, temp_value_next, 192, 192 * 2, temp_choose, to_expand='col')
                    temp_choose = generate_random_match(192, 384)
                    expand_t = expand_copy(expand_t, 192 * 2, 192 * 2, temp_choose, to_expand='row')
                    value_list.append(expand_t)
                expand_t = torch.Tensor()
                for v in value_list:
                    expand_t = torch.cat((expand_t, v), dim=0)
                checkpoint_tiny[key_t] = expand_t
            elif 'proj' in key_s:
                row = value_t.shape[0]
                col = value_t.shape[1]
                expand_t = expand_aki(value_t, value_next, row, col * 2, temp_choose, to_expand='col')
                temp_choose = generate_random_match(192, 384)
                expand_t = expand_copy(expand_t, row * 2, col * 2, temp_choose, to_expand='row')
                checkpoint_tiny[key_t] = expand_t

            elif 'fc1' in key_s:
                row = value_t.shape[0]
                col = value_t.shape[1]
                expand_t = expand_aki(value_t, value_next, row, col * 2, temp_choose, to_expand='col')

                temp_choose = generate_random_match(192 * 4, 384 * 4)

                expand_t = expand_copy(expand_t, row * 2, col * 2, temp_choose, to_expand='row')
                checkpoint_tiny[key_t] = expand_t
            elif 'fc2' in key_s:
                row = value_t.shape[0]
                col = value_t.shape[1]
                expand_t = expand_aki(value_t, value_next, row, col * 2, temp_choose, to_expand='col')

                temp_choose = generate_random_match(192, 384)

                expand_t = expand_copy(expand_t, row * 2, col * 2, temp_choose, to_expand='row')
                checkpoint_tiny[key_t] = expand_t

        elif 'bias' in key_s:
            if 'qkv' in key_s:
                value_list = []
                for i in range(3):
                    temp_value = value_t[i * 192:(i + 1) * 192]
                    expand_t = expand_copy(temp_value, 192 * 2, 1, temp_choose, to_expand='row')
                    value_list.append(expand_t)
                expand_t = torch.Tensor()
                for v in value_list:
                    expand_t = torch.cat((expand_t, v), dim=0)
                checkpoint_tiny[key_t] = expand_t
            else:
                row = value_t.shape[0]
                expand_t = expand_copy(value_t, row * 2, 1, temp_choose, to_expand='row')
                checkpoint_tiny[key_t] = torch.squeeze(expand_t)

    return temp_choose

def expand_head(checkpoint_tiny, checkpoint_small, choose):
    for (key_t, value_t), (key_s, value_s) in zip(checkpoint_tiny.items(), checkpoint_small.items()):
        if 'fc_norm' in key_s:
            choose_num_dict = choose
            expand_t = expand_copy(value_t, 384, 1, choose_num_dict, to_expand='row')
            checkpoint_tiny[key_t] = torch.squeeze(expand_t)
        elif 'head.weight' in key_s:
            choose_num_dict = choose
            expand_t = expand_fpi(value_t, 100, 384, choose_num_dict, to_expand='col')
            checkpoint_tiny[key_t] = expand_t


def tiny2small_strict_fpi(checkpoint_tiny, checkpoint_small):
    choose = {}
    choose['embed'] = generate_random_match(192, 384)
    choose['fc1'] = generate_random_match(192 * 4, 384 * 4)
    for i in range(192):
        choose['embed'][i + 192] = i

    for (key_t, value_t), (key_s, value_s) in zip(checkpoint_tiny.items(), checkpoint_small.items()):
        if 'cls_token' in key_s:
            row = value_t.shape[2]
            choose_num_dict = choose['embed']
            expand_t = expand_copy(value_t.squeeze(), row * 2, 1, choose_num_dict, to_expand='row')
            checkpoint_tiny[key_t] = torch.unsqueeze(torch.unsqueeze(expand_t, dim=0), dim=0)

        elif 'pos_embed' in key_s:
            choose_num_dict = choose['embed']
            expand_t = expand_copy(value_t.squeeze(), 197, 384, choose_num_dict, to_expand='col')
            checkpoint_tiny[key_t] = torch.unsqueeze(expand_t, dim=0)

        elif 'patch_embed' in key_s:
            if 'weight' in key_s:
                choose_num_dict = choose['embed']
                expand_t = expand_copy(value_t.reshape(192, 3 * 16 * 16), 384, 3 * 16 * 16, choose_num_dict,
                                       to_expand='row')
                checkpoint_tiny[key_t] = expand_t.reshape(384, 3, 16, 16)
            if 'bias' in key_s:
                row = value_t.shape[0]
                choose_num_dict = choose['embed']
                expand_t = expand_copy(value_t, row * 2, 1, choose_num_dict, to_expand='row')
                checkpoint_tiny[key_t] = torch.squeeze(expand_t)

        elif 'norm' in key_s:
            choose_num_dict = choose['embed']
            expand_t = expand_copy(value_t, 384, 1, choose_num_dict, to_expand='row')
            checkpoint_tiny[key_t] = torch.squeeze(expand_t)

        elif 'blocks' in key_s:
            if 'weight' in key_s:
                if 'qkv' in key_s:
                    value_list = []
                    for i in range(3):
                        temp_value = value_t[i * 192:(i + 1) * 192, :]
                        choose_num_dict = choose['embed']
                        expand_t = expand_fpi(temp_value, 192, 192 * 2, choose_num_dict, to_expand='col')

                        expand_t = expand_copy(expand_t, 192 * 2, 192 * 2, choose_num_dict, to_expand='row')
                        value_list.append(expand_t)
                    expand_t = torch.Tensor()
                    for v in value_list:
                        expand_t = torch.cat((expand_t, v), dim=0)
                    checkpoint_tiny[key_t] = expand_t
                elif 'fc1' in key_s:
                    row = value_t.shape[0]
                    col = value_t.shape[1]
                    choose_num_dict = choose['embed']
                    expand_t = expand_fpi(value_t, row, col * 2, choose_num_dict, to_expand='col')
                    choose_num_dict = choose['fc1']
                    expand_t = expand_copy(expand_t, row * 2, col * 2, choose_num_dict, to_expand='row')
                    checkpoint_tiny[key_t] = expand_t
                elif 'fc2' in key_s:
                    row = value_t.shape[0]
                    col = value_t.shape[1]
                    choose_num_dict = choose['fc1']
                    expand_t = expand_fpi(value_t, row, col * 2, choose_num_dict, to_expand='col')
                    choose_num_dict = choose['embed']
                    expand_t = expand_copy(expand_t, row * 2, col * 2, choose_num_dict, to_expand='row')
                    checkpoint_tiny[key_t] = expand_t
                else:
                    row = value_t.shape[0]
                    col = value_t.shape[1]
                    choose_num_dict = choose['embed']
                    expand_t = expand_fpi(value_t, row, col * 2, choose_num_dict, to_expand='col')
                    expand_t = expand_copy(expand_t, row * 2, col * 2, choose_num_dict, to_expand='row')
                    checkpoint_tiny[key_t] = expand_t

            elif 'bias' in key_s:
                if 'qkv' in key_s:
                    value_list = []
                    for i in range(3):
                        temp_value = value_t[i * 192:(i + 1) * 192]
                        choose_num_dict = choose['embed']
                        expand_t = expand_copy(temp_value, 192 * 2, 1, choose_num_dict, to_expand='row')
                        value_list.append(expand_t)
                    expand_t = torch.Tensor()
                    for v in value_list:
                        expand_t = torch.cat((expand_t, v), dim=0)
                    checkpoint_tiny[key_t] = expand_t
                elif 'fc1' in key_s:
                    row = value_t.shape[0]
                    choose_num_dict = choose['fc1']
                    expand_t = expand_copy(value_t, row * 2, 1, choose_num_dict, to_expand='row')
                    checkpoint_tiny[key_t] = torch.squeeze(expand_t)
                else:
                    row = value_t.shape[0]
                    choose_num_dict = choose['embed']
                    expand_t = expand_copy(value_t, row * 2, 1, choose_num_dict, to_expand='row')
                    checkpoint_tiny[key_t] = torch.squeeze(expand_t)

        elif 'head.weight' in key_s:
            choose_num_dict = choose['embed']
            expand_t = expand_fpi(value_t, 100, 384, choose_num_dict, to_expand='col')
            checkpoint_tiny[key_t] = expand_t

    return checkpoint_tiny


def tiny2small_fpi(checkpoint_tiny, checkpoint_small):
    choose = expand_embed(checkpoint_tiny, checkpoint_small)
    blocks_tiny = split_blocks(checkpoint_tiny)
    blocks_small = split_blocks(checkpoint_small)
    for index, (block_t, block_s) in enumerate(zip(blocks_tiny, blocks_small)):
        choose = expand_fpi_block(checkpoint_tiny, checkpoint_small, block_t, block_s, choose)
    expand_head(checkpoint_tiny, checkpoint_small, choose)
    return checkpoint_tiny


def tiny2small_aki(checkpoint_tiny, checkpoint_small):
    choose = expand_embed(checkpoint_tiny, checkpoint_small)
    blocks_tiny = split_blocks(checkpoint_tiny)
    blocks_small = split_blocks(checkpoint_small)
    for index, (block_t, block_s) in enumerate(zip(blocks_tiny, blocks_small)):
        if index != (len(blocks_tiny)-1):
            choose = expand_aki_block(checkpoint_tiny, checkpoint_small, block_t, block_s, choose, blocks_tiny[index+1])
        else:
            choose = expand_fpi_block(checkpoint_tiny, checkpoint_small, block_t, block_s, choose)
    expand_head(checkpoint_tiny, checkpoint_small, choose)
    return checkpoint_tiny





