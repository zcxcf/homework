import random

# Original list
depth_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Randomly decide how many elements to remove (1, 2, or 3)
num_to_remove = random.choice([0, 1, 2, 3])

# Randomly select indices to remove
indices_to_remove = random.sample(range(len(depth_list)), num_to_remove)

# Remove the elements by indices
new_depth_list = [depth_list[i] for i in range(len(depth_list)) if i not in indices_to_remove]

print(f"Original list: {depth_list}")
print(f"New list after removing {num_to_remove} elements: {new_depth_list}")
