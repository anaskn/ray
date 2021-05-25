import argparse
import pickle 
import numpy as np

parser = argparse.ArgumentParser(description='...')
#parser.add_argument('-l','--layer', type=int, nargs='+', required=True, action='append', help='layer list')
parser.add_argument("--activation", nargs="+", default= ["relu"])

args = parser.parse_args()

#print(args.layer)
print(args.activation)
cpt = 1

string1 =  'data4/listfile_40_'+str(cpt)+'.data' #_evol'+ , _pos'+
with open(string1, 'rb') as filehandle:
# read the data as binary data stream
    lst = pickle.load(filehandle)


string2 = 'data4/nei_tab_pos_40_'+str(cpt)+'.data'
with open(string2, 'rb') as filehandle:
    # read the data as binary data stream
    nei_tab = pickle.load(filehandle)

print(np.shape(nei_tab)) # good (40,20)
print(np.shape(lst)) # good (40,20)


print(lst[19][19])
