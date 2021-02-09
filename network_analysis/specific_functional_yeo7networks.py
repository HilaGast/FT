
def network_id_list(ntype,side):

    if ntype == 'viz':
        il = list(range(0,14))
        ir = list(range(114,99,-1))

    if ntype == 'sommot':
        il = list(range(14,30))
        ir = list(range(133,114,-1))


    if side == "both":
        return il+ir

    elif side == "l":
        return il

    elif side == "r":
        return ir




