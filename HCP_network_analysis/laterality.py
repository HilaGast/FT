def comp_li(cr, nc=[], atlas='bna', ignore_zeros=True):
    li=0

    if atlas=='bna':
        if len(nc) > 0 and ignore_zeros:
            cr[nc] = 0
            for i in range(1,246,2):
                if cr[i] != cr[i+1]:
                    li+=1
        else:
            for i in range(1,246,2):
                if cr[i] != cr[i+1]:
                    li+=1
        li = li/123 * 100

    elif atlas == 'yeo7_200':
        if len(nc) > 0 and ignore_zeros:
            cr[nc] = 0
            for i in range(1,100):
                if cr[i] != cr[i+100]:
                    li+=1
        else:
            for i in range(1,100):
                if cr[i] != cr[i+100]:
                    li+=1
    return li