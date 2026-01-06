function = flame['functions'][0]

rotation = rot_mat(2 * math.pi * t)

pre_trans = function['pre_trans'][:4].reshape((2, 2))
pre_trans[:] = pre_trans @ rotation