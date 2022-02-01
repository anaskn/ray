

#/////////////                     g + o   unused //////////////////////////////////////////////////////////////////////////////
#"""

import matplotlib.pyplot as plt
import pickle
var = "rc"
algo=""

pdf_plot = var # R_c, C_o, C_u, k
lstt = [var]#, "C_o", "C_u", "k"]


for pdf_plot in lstt:

	with open('drawn_plot/RSU_plot/RSU_20ep_multi_agent_ttl5_'+var+'_appo.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'



		# read the data as binary data stream
		appo = pickle.load(filehandle)
		#zipped_lists = zip(appo[0], appo[1])  # zip of unused shared and own resources
		#appo = [x + y for (x, y) in zipped_lists] # sum list

	with open('drawn_plot/RSU_plot/RSU_20ep_multi_agent_ttl5_'+var+'_ppo.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		ppo =pickle.load(filehandle)
		#zipped_lists = zip(ppo[0], ppo[1])  # zip of unused shared and own resources
		#ppo = [x + y for (x, y) in zipped_lists] # sum list

	with open('drawn_plot/RSU_plot/RSU_20ep_multi_agent_ttl5_'+var+'_a3c.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		a3c =pickle.load(filehandle)

		#zipped_lists = zip(a3c[0], a3c[1])  # zip of unused shared and own resources
		#a3c = [x + y for (x, y) in zipped_lists] # sum list
		#a3c[5]=5
		#a3c[11]=4
	"""
	with open('03_unused_random.data', 'rb') as filehandle: # 1_ddpg4442C_o
		# read the data as binary data stream
		random = pickle.load(filehandle)
		zipped_lists = zip(random[0], random[1])  # zip of unused shared and own resources
		random = [x + y for (x, y) in zipped_lists] # sum list

	with open('03_unused_fifty.data', 'rb') as filehandle: # 1_ddpg4442C_o
		# read the data as binary data stream
		fifty = pickle.load(filehandle)
		zipped_lists = zip(fifty[0], fifty[1])  # zip of unused shared and own resources
		fifty = [x + y for (x, y) in zipped_lists] # sum list
	"""
	
	appo=[17.5,14,12,10,9.5]
	ppo=[16,13.8,11,9.5,9.4]
	a3c=[12,9,7,5,5]


	#time = [0.5,2,4,6,8,10,12,14,16,18,20,50]#,100]
	#times = [2,4,6,8,10,12,14,16,18,20]
	#times = [1,2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,55,60]	
	#times =[10,20,30,40,50,60]
	#times =[2,4,6,8,10,12,14,16,18,20]	
	#times = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	times = [0,1,2,3,4]


	#plt.plot(times , appo, color='orange', linestyle='dotted', marker='x' ,label='APPO') #  unused shared  'ppo_$Unused$'
	plt.plot(times , ppo, color='orange', linestyle='dashed', marker='D' ,label='PPO') #  unused shared  
	plt.plot(times , a3c, color='red', linestyle='--', marker='2' ,label='A3C') #  unused shared  
	#plt.plot(times , td3, color='blue',linestyle='--', marker='2' ,label='td3_$Unused_{t}$') #  unused shared  
	#plt.plot(times , random, color='cyan',linestyle='solid', marker='s' ,label='Random') #  unused shared    random
	#plt.plot(times , fifty, color='pink',linestyle='--', marker='<' ,label='fifty') #  unused shared    fifty
	

	plt.ylabel('Unsatisfied caching demands', size= 8 ) #resource
	#plt.ylabel('Unused Caching Ressources', size= 8 ) #resource

	#plt.xlabel('$'+var+'$', size= 10) #'$'+pdf_plot[para]+'$ '   $'+var+'$    Communication range
	plt.xlabel('$RSU$', size= 10) #'$'+pdf_plot[para]+'$ '   $'+var+'$    Communication range

	#plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),size = 7)
	plt.xticks((0,1,2,3,4),size = 7)
	plt.yticks(size = 7)
	plt.grid()

	# Add a legend
	#plt.legend(ncol=1, bbox_to_anchor=(0.75, 0.45)) #r_c
	#plt.legend(ncol=1, bbox_to_anchor=(1, 0.5)) #c_o
	plt.legend()#ncol=1, bbox_to_anchor=(1, 0.5))#c_u

	#plt.legend(ncol=1, bbox_to_anchor=(1, 0.5)) #k


	#plt.savefig('drawn_plot/RSU_plot/MA_ttl5_'+var+'_unusatisfied_g+o_z.pdf') #abbbs_    b_test_five_'+var+'_plot.pdf
	#plt.savefig('drawn_plot/RSU_plot/MA_ttl5_'+var+'_unusatisfied_shared_z.pdf') 
	#plt.savefig('drawn_plot/RSU_plot/MA_ttl5_'+var+'_unusatisfied_own_z.pdf') 
	plt.savefig('drawn_plot/RSU_plot/RSU_lst_ttl5_'+var+'_unusatisfied_z2.pdf') 




	plt.show()

	print("End")

#"""
"""

#/////////////                     g + o   unsatisfied  //////////////////////////////////////////////////////////////////////////////

#simple plot 

import matplotlib.pyplot as plt
import pickle
var = "c"

pdf_plot = var # R_c, C_o, C_u, k
lstt = [var]#, "C_o", "C_u", "k"]


for pdf_plot in lstt:

	with open('drawn_plot/MA_plot/data11_z4_20ep_multi_agent_ttl5_'+var+'_appo.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'

		# read the data as binary data stream
		appo_ma = pickle.load(filehandle)


	with open('drawn_plot/MA_plot/data11_z4_20ep_multi_agent_ttl5_'+var+'_ppo.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		ppo_ma =pickle.load(filehandle)


	with open('drawn_plot/MA_plot/data11_z4_20ep_multi_agent_ttl5_'+var+'_a3c.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		a3c_ma =pickle.load(filehandle)


	with open('drawn_plot/RSU_plot/RSU_20ep_multi_agent_ttl5_'+var+'_appo.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'

		# read the data as binary data stream
		appo_rsu = pickle.load(filehandle)


	with open('drawn_plot/RSU_plot/RSU_20ep_multi_agent_ttl5_'+var+'_ppo.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		ppo_rsu =pickle.load(filehandle)


	with open('drawn_plot/RSU_plot/RSU_20ep_multi_agent_ttl5_'+var+'_a3c.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		a3c_rsu =pickle.load(filehandle)

	

	

	
	times = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	


	plt.plot(times , appo_ma[3], color='cyan', linestyle='solid', marker='<' ,label='APPO_MA') #  unused shared  'ppo_$Unused$'
	plt.plot(times , ppo_ma[3], color='green', linestyle='dashed', marker='D' ,label='PPO_MA') #  unused shared  
	plt.plot(times , a3c_ma[3], color='pink', linestyle='--', marker='2' ,label='A3C_MA') #  unused shared
	plt.plot(times , appo_rsu[3], color='orange', linestyle='dotted', marker='x' ,label='APPO_RSU') #  unused shared  'ppo_$Unused$'
	plt.plot(times , ppo_rsu[3], color='red', linestyle='dashed', marker='D' ,label='PPO_RSU') #  unused shared  
	plt.plot(times , a3c_rsu[3], color='blue', linestyle='--', marker='2' ,label='A3C_RSU') #  unused shared  
	


	plt.ylabel('unsatisfied own caching demands', size= 8 ) #resource
	plt.xlabel('$'+var+'$', size= 10) #'$'+pdf_plot[para]+'$ '   $'+var+'$    Communication range

	plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),size = 7)
	plt.yticks(size = 7)
	plt.grid()

	# Add a legend
	#plt.legend(ncol=1, bbox_to_anchor=(0.75, 0.45)) #r_c
	#plt.legend(ncol=1, bbox_to_anchor=(1, 0.5)) #c_o
	plt.legend()#ncol=1, bbox_to_anchor=(1, 0.5))#c_u

	#plt.legend(ncol=1, bbox_to_anchor=(1, 0.5)) #k

	plt.savefig('drawn_plot/ttl5_'+var+'_unsatisfied_own_all.pdf') #abbbs_    b_test_five_'+var+'_plot.pdf




	plt.show()

	print("End")




"""











"""


import matplotlib.pyplot as plt
import numpy as np 
import pickle

with open('Reward_multi_agentppo.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'
        # read the data as binary data stream
        ppo = pickle.load(filehandle)
        print("LEN ppo = ", len(ppo[0]))

ppo = ppo[0][:20000]
ppo = [ ppo[xx] for xx in range(len(ppo)) if xx%20==0 ]
window_width= 100
cumsum_vec = np.cumsum(np.insert(ppo, 0, 0))
ppo = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width


with open('Reward_multi_agentappo.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'
        # read the data as binary data stream
        appo = pickle.load(filehandle)
        print("LEN appo = ", len(appo[0]))

appo = appo[0][:20000]
appo = [ appo[xx] for xx in range(len(appo)) if xx%20==0 ]
window_width= 100
cumsum_vec = np.cumsum(np.insert(appo, 0, 0))
appo = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width


with open('Reward_multi_agenta3c.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'
        # read the data as binary data stream
        a3c = pickle.load(filehandle)
        print("LEN a3c = ", len(a3c[0]))

a3c = a3c[0][:20000]
a3c = [ a3c[xx] for xx in range(len(a3c)) if xx%20==0 ]
window_width= 100
cumsum_vec = np.cumsum(np.insert(a3c, 0, 0))
a3c = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width



print("LEN ppo = ", len(ppo))
print("LEN appo = ", len(appo))
print("LEN a3c = ", len(a3c))







x = np.arange(len(ppo))
times = range(len(ppo))

# plot our data along a line
fig,ax = plt.subplots()
#ax.plot(times, ppo, '-', color='tab:blue', linestyle='dotted', marker='x' ,label='PPO')
ax.plot(times, appo, '-', color='tab:orange', linestyle='dashed', marker='D' ,label='PPO')
ax.plot(times, a3c, '-', color='tab:red', linestyle='--', marker='2' ,label='A3C')

ax.set_title('')
plt.xticks(np.arange(min(x), max(x)+1, 200))  # [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
plt.xlabel('Epochs', size= 10)

ax.set_ylabel('Reward')

# create a confidence band of +/- 10% error
y_lower = [i - 0.1 * i for i in ppo]
y_upper = [i + 0.1 * i for i in ppo]

y_lower_multi = [i - 0.1 * i for i in appo]
y_upper_multi = [i + 0.1 * i for i in appo]

y_lower_rsu = [i - 0.1 * i for i in a3c]
y_upper_rsu= [i + 0.1 * i for i in a3c]



# plot our confidence band
#ax.fill_between(times, y_lower, y_upper, alpha=0.2, color='tab:blue')
ax.fill_between(times, y_lower_multi, y_upper_multi, alpha=0.2, color='tab:orange')
ax.fill_between(times, y_lower_rsu, y_upper_rsu, alpha=0.2, color='tab:red')




print("min = ", min(x))
print("max = ", max(x))
print("len x = ", len(x))

plt.legend()
plt.grid()
plt.savefig('new_zz_reward_all_new++.pdf')
plt.show()

"""











"""





import matplotlib.pyplot as plt
import numpy as np 
import pickle

with open('Reward_multi_agentppo.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'
        # read the data as binary data stream
        single = pickle.load(filehandle)
        print("LEN SINGLE = ", len(single[0]))

single = single[0][:2000]
single = [ single[xx] for xx in range(len(single)) if xx%20==0 ]
window_width= 100
cumsum_vec = np.cumsum(np.insert(single, 0, 0))
single = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width


with open('Reward_multi_agentappo.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'
        # read the data as binary data stream
        multi = pickle.load(filehandle)
        print("LEN multi = ", len(multi[0]))

multi = multi[0][:2000]
multi = [ multi[xx] for xx in range(len(multi)) if xx%20==0 ]
window_width= 100
cumsum_vec = np.cumsum(np.insert(multi, 0, 0))
multi = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width


with open('Reward_multi_agenta3c.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'
        # read the data as binary data stream
        rsu = pickle.load(filehandle)
        print("LEN rsu = ", len(rsu[0]))

rsu = rsu[0][:2000]
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
ax.plot(times, single, '-', color='tab:blue', linestyle='dotted', marker='x' ,label='PPO')
ax.plot(times, multi, '-', color='tab:orange', linestyle='dashed', marker='D' ,label='APPO')
ax.plot(times, rsu, '-', color='tab:red', linestyle='--', marker='2' ,label='A3C')

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
ax.fill_between(times, y_lower_rsu, y_upper_rsu, alpha=0.2, color='tab:orange')




print("min = ", min(x))
print("max = ", max(x))
print("len x = ", len(x))

plt.legend()
plt.grid()
plt.savefig('drawn_plot/Reward_multi_agent_++NN')
plt.show()

print("okko")

"""

