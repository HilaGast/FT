
import scipy.io as sio
# load:
mat_file_name = r''
var_name = ''
mat=sio.loadmat(mat_file_name)
var = mat[var_name]

#save:
''' saves "var" variable into a variable named "var_name" in "mat_file_name" .mat file'''
sio.savemat(mat_file_name,{'var_name':var})


n=1
for r in nwei:
    nwei_h=nwei_h+list(r[0:n])
    n+=1