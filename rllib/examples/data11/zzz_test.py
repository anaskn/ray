import pickle


if __name__ == '__main__':

   
    string2 = 'nei_tab_pos_dist10_'+str(10)+'.data'   #'data4/nei_tab_pos_40_'+str(cpt)+'.data'
    with open(string2, 'rb') as filehandle:
        # read the data as binary data stream
        nei_tab = pickle.load(filehandle)


    string1 =  'listfile_dist10_'+str(10)+'.data' #_evol'+ , _pos'+   #'data4/listfile_40_'+str(cpt)+'.data'
    with open(string1, 'rb') as filehandle:
    # read the data as binary data stream
        lst = pickle.load(filehandle)



    #print(lst)
    print("++++++")
    print(lst[0])
    print("++++++")
    print(lst[1])
    print("++++++")
    print(lst[2])



    # so data8 for rsu (with high demand) 