import random
def random_state():
    rs_list = [1,2,5,7,42,21,11,17,25,51,None] #xgboost
    #rs_list = [42,1,11,14,26,27,99,37,1,11,14,26,27,None, None]
    rs = random.choice(rs_list)
    return rs