
# to plot on the same figure ppo, ddpg and td3 for unused and unsatisfied 
    # 1 simple 
    # 2 multiple agent
    # plot reward of simple / multi-agent / rsu 

"""

# 1***Simple agent ***** 

import matplotlib.pyplot as plt
import pickle
var = "rc"

pdf_plot = var # R_c, C_o, C_u, k
lstt = [var]#, "C_o", "C_u", "k"]


for pdf_plot in lstt:

    with open('z_20ep_resources_'+var+'_ppo.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'
        # read the data as binary data stream
        ppo = pickle.load(filehandle)
        zipped_lists = zip(ppo[0], ppo[1])  # zip of unused shared and own resources
        ppo_unused = [x + y for (x, y) in zipped_lists] # sum list
        zipped_lists = zip(ppo[2], ppo[3])
        ppo_unsatisfied = [x + y for (x, y) in zipped_lists]

    with open('z_20ep_resources_'+var+'_ddpg.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
        # read the data as binary data stream
        ddpg =pickle.load(filehandle)
        zipped_lists = zip(ddpg[0], ddpg[1])  # zip of unused shared and own resources
        ddpg_unused = [x + y for (x, y) in zipped_lists] # sum list
        zipped_lists = zip(ddpg[2], ddpg[3])  # zip of unused shared and own resources
        ddpg_unsatisfied = [x + y for (x, y) in zipped_lists] # sum list
   
    with open('z_20ep_resources_'+var+'_td3.data', 'rb') as filehandle: # 1_ddpg4442C_o
        # read the data as binary data stream
        td3 = pickle.load(filehandle)
        zipped_lists = zip(td3[0], td3[1])  # zip of unused shared and own resources
        td3_unused = [x + y for (x, y) in zipped_lists] # sum list
        zipped_lists = zip(td3[2], td3[3])  # zip of unused shared and own resources
        td3_unsatisfied = [x + y for (x, y) in zipped_lists] # sum list


    times = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]



    plt.plot(times , ppo_unused, color='orange', linestyle='dotted', marker='x' ,label='PPO_unused') #  unused shared  'ppo_$Unused$'
    plt.plot(times , ddpg_unused, color='red', linestyle='dashed', marker='D' ,label='DDPG_unused') #  unused shared
    plt.plot(times , td3_unused, color='blue', linestyle='--', marker='2' ,label='TD3_unused') #  unused shared  

    plt.plot(times , ppo_unsatisfied, color='green', linestyle='dotted', marker='s' ,label='PPO_unsatisfied') #  unused shared  'ppo_$Unused$'
    plt.plot(times , ddpg_unsatisfied, color='pink', linestyle='solid', marker='<' ,label='DDPG_unsatisfied') #  unused shared
    plt.plot(times , td3_unsatisfied, color='brown', linestyle='--', marker='2' ,label='TD3_unsatisfied') #  unused shared  

    


    plt.ylabel('Caching Resources', size= 8 ) #resource
    plt.xlabel('$'+var+'$', size= 10) #'$'+pdf_plot[para]+'$ '   $'+var+'$    Communication range

    plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),size = 7)
    plt.yticks(size = 7)
    plt.grid()


    plt.legend()#ncol=1, bbox_to_anchor=(1, 0.5))#c_u
    plt.grid()

    plt.savefig('zz_caching_'+var+'_g+o_z.pdf') #abbbs_    b_test_five_'+var+'_plot.pdf




    #plt.show()
    print("EEND")
    print("End")

"""

"""
# 2***multi agent ***** 

import matplotlib.pyplot as plt
import pickle
var = "rc"

pdf_plot = var # R_c, C_o, C_u, k
lstt = [var]#, "C_o", "C_u", "k"]


for pdf_plot in lstt:

    with open('z_20ep_multi_agent_'+var+'_ppo.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'
        # read the data as binary data stream
        ppo = pickle.load(filehandle)
        zipped_lists = zip(ppo[0], ppo[1])  # zip of unused shared and own resources
        ppo_unused = [x + y for (x, y) in zipped_lists] # sum list
        zipped_lists = zip(ppo[2], ppo[3])
        ppo_unsatisfied = [x + y for (x, y) in zipped_lists]

    with open('z_20ep_multi_agent_'+var+'_ddpg.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
        # read the data as binary data stream
        ddpg =pickle.load(filehandle)
        zipped_lists = zip(ddpg[0], ddpg[1])  # zip of unused shared and own resources
        ddpg_unused = [x + y for (x, y) in zipped_lists] # sum list
        zipped_lists = zip(ddpg[2], ddpg[3])  # zip of unused shared and own resources
        ddpg_unsatisfied = [x + y for (x, y) in zipped_lists] # sum list
   
    with open('z_20ep_multi_agent_'+var+'_td3.data', 'rb') as filehandle: # 1_ddpg4442C_o
        # read the data as binary data stream
        td3 = pickle.load(filehandle)
        zipped_lists = zip(td3[0], td3[1])  # zip of unused shared and own resources
        td3_unused = [x + y for (x, y) in zipped_lists] # sum list
        zipped_lists = zip(td3[2], td3[3])  # zip of unused shared and own resources
        td3_unsatisfied = [x + y for (x, y) in zipped_lists] # sum list


    times = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]



    plt.plot(times , ppo_unused, color='orange', linestyle='dotted', marker='x' ,label='PPO_unused') #  unused shared  'ppo_$Unused$'
    plt.plot(times , ddpg_unused, color='red', linestyle='dashed', marker='D' ,label='DDPG_unused') #  unused shared
    plt.plot(times , td3_unused, color='blue', linestyle='--', marker='2' ,label='TD3_unused') #  unused shared  

    plt.plot(times , ppo_unsatisfied, color='green', linestyle='dotted', marker='s' ,label='PPO_unsatisfied') #  unused shared  'ppo_$Unused$'
    plt.plot(times , ddpg_unsatisfied, color='pink', linestyle='solid', marker='<' ,label='DDPG_unsatisfied') #  unused shared
    plt.plot(times , td3_unsatisfied, color='brown', linestyle='--', marker='2' ,label='TD3_unsatisfied') #  unused shared  

    


    plt.ylabel('Caching Resources', size= 8 ) #resource
    plt.xlabel('$'+var+'$', size= 10) #'$'+pdf_plot[para]+'$ '   $'+var+'$    Communication range

    plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),size = 7)
    plt.yticks(size = 7)
    plt.grid()


    plt.legend()#ncol=1, bbox_to_anchor=(1, 0.5))#c_u
    plt.grid()

    plt.savefig('zz_multi_caching_'+var+'_g+o_z.pdf') #abbbs_    b_test_five_'+var+'_plot.pdf




    #plt.show()

    print("End")
"""

"""

# 3***reward simple / multi-agent / rsu ***** 

import matplotlib.pyplot as plt
import pickle
import numpy as np
var = "k"

pdf_plot = var # R_c, C_o, C_u, k
lstt = [var]#, "C_o", "C_u", "k"]

print("okokokok")
for pdf_plot in lstt:

    with open('Reward_ppo.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'
        # read the data as binary data stream
        single = pickle.load(filehandle)
    single = single[0][:219999]

    window_width= 100
    cumsum_vec = np.cumsum(np.insert(single, 0, 0))
    single = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        

 
    with open('Reward_multi_agentppo.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
        # read the data as binary data stream
        multi =pickle.load(filehandle)
    multi = multi[0][:219999]
    window_width= 100
    cumsum_vec = np.cumsum(np.insert(multi, 0, 0))
    multi = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
       

    with open('Reward_multi_agent_RSUppo.data', 'rb') as filehandle: # 1_ddpg4442C_o
        # read the data as binary data stream
        rsu = pickle.load(filehandle)
    rsu = rsu[0][:219999]
    window_width= 100
    cumsum_vec = np.cumsum(np.insert(multi, 0, 0))
    multi = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        


    plt.plot(single, color='orange', linestyle='dotted', marker='x' ,label='Single_agent') #  unused shared  'ppo_$Unused$'
    plt.plot(multi, color='red', linestyle='dashed', marker='D' ,label='Multi_agent') #  unused shared
    plt.plot(rsu, color='blue', linestyle='--', marker='2' ,label='Multi_agent_RSU') #  unused shared  


    plt.ylabel('Reward', size= 8 ) #resource
    plt.xlabel('Epochs', size= 10) #'$'+pdf_plot[para]+'$ '   $'+var+'$    Communication range

    #plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),size = 7)
    plt.yticks(size = 7)
    plt.grid()


    plt.legend()#ncol=1, bbox_to_anchor=(1, 0.5))#c_u
    plt.grid()

    plt.savefig('zz_reward_all.pdf') #abbbs_    b_test_five_'+var+'_plot.pdf


    #plt.show()

    print("End")

"""


import matplotlib.pyplot as plt
import numpy as np 
import pickle

with open('Reward_ppo.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'
        # read the data as binary data stream
        single = pickle.load(filehandle)
        print("LEN SINGLE = ", len(single[0]))

single = single[0][:20000]
single = [ single[xx] for xx in range(len(single)) if xx%20==0 ]
window_width= 100
cumsum_vec = np.cumsum(np.insert(single, 0, 0))
single = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width


with open('Reward_multi_agentppo.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'
        # read the data as binary data stream
        multi = pickle.load(filehandle)
        print("LEN multi = ", len(multi[0]))

multi = multi[0][:20000]
multi = [ multi[xx] for xx in range(len(multi)) if xx%20==0 ]
window_width= 100
cumsum_vec = np.cumsum(np.insert(multi, 0, 0))
multi = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width


with open('Reward_multi_agent_RSUppo.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'
        # read the data as binary data stream
        rsu = pickle.load(filehandle)
        print("LEN rsu = ", len(rsu[0]))

rsu = rsu[0][:20000]
rsu = [ rsu[xx] for xx in range(len(rsu)) if xx%20==0 ]
window_width= 100
cumsum_vec = np.cumsum(np.insert(rsu, 0, 0))
rsu = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width



print("LEN SINGLE = ", len(single))
print("LEN multi = ", len(multi))
print("LEN rsu = ", len(rsu))







x = np.arange(len(single))
times = range(len(single))

# plot our data along a line
fig,ax = plt.subplots()
ax.plot(times, single, '-', color='tab:blue', linestyle='dotted', marker='x' ,label='Single_agent')
ax.plot(times, multi, '-', color='tab:orange', linestyle='dashed', marker='D' ,label='Multi_agent')
ax.plot(times, rsu, '-', color='tab:red', linestyle='--', marker='2' ,label='Multi_agent_RSU')

ax.set_title('')
plt.xticks(np.arange(min(x), max(x)+1, 200))  # [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
plt.xlabel('Epochs', size= 10)

ax.set_ylabel('Reward')

# create a confidence band of +/- 10% error
y_lower = [i - 0.1 * i for i in single]
y_upper = [i + 0.1 * i for i in single]

y_lower_multi = [i - 0.1 * i for i in multi]
y_upper_multi = [i + 0.1 * i for i in multi]

y_lower_rsu = [i - 0.1 * i for i in rsu]
y_upper_rsu= [i + 0.1 * i for i in rsu]



# plot our confidence band
ax.fill_between(times, y_lower, y_upper, alpha=0.2, color='tab:blue')
ax.fill_between(times, y_lower_multi, y_upper_multi, alpha=0.2, color='tab:orange')
ax.fill_between(times, y_lower_rsu, y_upper_rsu, alpha=0.2, color='tab:red')




print("min = ", min(x))
print("max = ", max(x))
print("len x = ", len(x))

plt.legend()
plt.grid()
plt.savefig('zz_reward_all.pdf')
plt.show()


#"""
