# modules
import h5py
import numpy as np

sf = h5py.File('shuffle.h5', 'r')
nf = h5py.File('noshuffle.h5', 'r')

for key in sf.keys():
    sd = sf[key]
    nd = nf[key]
    # print(sd)
    # # print(sd[0:2].shape)
    # # print(sd[0:2])
    # print(sd[:].shape)
    # arr = np.array(sd[:])
    # print(arr.shape)
    # print(arr)
    
    print(sd)
    if np.all(sd[:] == nd[:]): print("match")
    # if sd[0] == nd[0]: print("yee")
    # for i in range(0,sd.shape[0]):
    #     for j in range(0, sd.shape[1]):
    #         # print(key)
    #         # if key == 'BES_vars': print(sd[0][0], nd[0][0])
    #         if key == 'BES_vars':
    #             # if sd[0][0] == nd[0][0]: print(key + "match")
    #             if sd[i][j] != nd[i][j]: quit()
    #         else:
    #             # if sd[0][0][0] == nd[0][0][0]: print(key + " match")
    #             for k in range(0,sd.shape[2]):
    #                 if sd[i][j][k] != nd[i][j][k]: quit()
    print('')
x = np.array([[0,1],[2,3]])
y = np.array([[3,1],[0,4]])
print(x==y)

# x = [0,1,2,3,4]
# y = [2,1,4,3,0]
print(np.any(x==y))
print(np.all(x==y))

# if 1.2834 == 1.2834: print('test yes')
print("script finished")
