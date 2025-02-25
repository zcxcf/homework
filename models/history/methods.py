from expand_methods.fpi_strict import expand_vit as method_1
from expand_methods.aki import expand_vit as method_2

def fpi_strict(checkpoint_initial, initial_width, target_width, initial_depth, target_depth, insert_positions):
    return method_1(checkpoint_initial, initial_width, target_width, initial_depth, target_depth, insert_positions)

def aki(checkpoint_initial, initial_width, target_width, initial_depth, target_depth, insert_positions):
    return method_2(checkpoint_initial, initial_width, target_width, initial_depth, target_depth, insert_positions)
