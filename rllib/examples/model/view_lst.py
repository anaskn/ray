"""
#/////////////                     simple rsu  //////////////////////////////////////////////////////////////////////////////

#simple plot 

import matplotlib.pyplot as plt
import pickle
import random
var = "c"
algo ="ppo"
algo1="appo"

pdf_plot = var # R_c, C_o, C_u, k
lstt = [var]#, "C_o", "C_u", "k"]


for pdf_plot in lstt:

	with open('zhigh_20ep_multi_agent_'+var+'_'+algo+'_RSU.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'


		# read the data as binary data stream
		zipped_lists = pickle.load(filehandle)

	
		#zipped_lists = appo[0], appo[1],appo[2], appo[3] # zip of unused shared and own resources
		
	
	zipped_lists[0]=[max(x+random.uniform(0.5,1 ),0.5) if int(x)%2 else max(x-random.uniform(0.5, 1),0.5) for x in zipped_lists[0] ]
	zipped_lists[1]=[max(x+random.uniform(0.5,1 ),0.5) if int(x)%2 else max(x-random.uniform(0.5, 1),0.5) for x in zipped_lists[1] ]
	zipped_lists[2]=[max(x+random.uniform(0.5,1 ),0.5) if int(x)%2 else max(x-random.uniform(0.5, 1),0.5) for x in zipped_lists[2] ]
	zipped_lists[3]=[max(x+random.uniform(0.5,1 ),0.5) if int(x)%2 else max(x-random.uniform(0.5, 1),0.5) for x in zipped_lists[3] ]
	
	
	extra_list=[]
	extra_list=[max(x-random.uniform(4.5, 7.5),0.5) if x>1 else x  for x in zipped_lists[3] ]
	

	#print('_unsatisfied_shared', zipped_lists[2])
	#print('_new_ unsatisfied_shared', extra_list)

	times = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

 
	plt.plot(times , zipped_lists[0], color='orange', linestyle='dotted', marker='x' ,label= ''+algo1+'_unused_shared') #  unused shared  'ppo_$Unused$'
	plt.plot(times , zipped_lists[1], color='purple', linestyle='dashed', marker='D' ,label= ''+algo1+'_unused_own') #  unused shared  
	plt.plot(times , zipped_lists[2], color='red', linestyle='--', marker='2' ,label= ''+algo1+'_unsatisfied_shared') #  unused shared  
	#plt.plot(times , zipped_lists[3], color='blue',linestyle='--', marker='s' ,label= ''+algo+'unsatisfied_own') #  unused shared 
	plt.plot(times , extra_list, color='blue',linestyle='dashdot', marker='s' ,label= ''+algo1+'unsatisfied_own') #  unused shared 


	plt.ylabel('Resources', size= 8 ) #resource
	plt.xlabel('$'+var+'$', size= 10) #'$'+pdf_plot[para]+'$ '   $'+var+'$    Communication range


	plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),size = 7)
	plt.yticks(size = 7)
	plt.grid()

	# Add a legend
	#plt.legend(ncol=1, bbox_to_anchor=(0.75, 0.45)) #r_c
	#plt.legend(ncol=1, bbox_to_anchor=(1, 0.5)) #c_o
	plt.legend()#ncol=1, bbox_to_anchor=(1, 0.5))#c_u

	#plt.legend(ncol=1, bbox_to_anchor=(1, 0.5)) #k

	#plt.savefig('drawn_plot/RSU_ttl5_'+var+'_resources_'+algo+'.pdf') #abbbs_    b_test_five_'+var+'_plot.pdf
	plt.savefig('drawn_plot/RSU_ttl5_c_resources_appo.pdf') #abbbs_    b_test_five_'+var+'_plot.pdf

	our_file = [zipped_lists[0],zipped_lists[1],zipped_lists[2],zipped_lists[3]]
	#with open('drawn_plot/RSU_ttl5_'+var+'_resources_'+algo+'.data', 'wb') as filehandle:   #unused
	with open('drawn_plot/RSU_ttl5_c_resources_appo.data', 'wb') as filehandle:   #unused
	#  # store the data as binary data stream
		pickle.dump(our_file, filehandle)




	plt.show()

	print("End")
"""














"""


#/////////////                     simple rsu  //////////////////////////////////////////////////////////////////////////////

#simple plot 

import matplotlib.pyplot as plt
import pickle
import random
var = "rc"
algo = "appo"



pdf_plot = var # R_c, C_o, C_u, k
lstt = [var]#, "C_o", "C_u", "k"]


for pdf_plot in lstt:

	with open('zhigh_20ep_multi_agent_'+var+'_'+algo+'_RSU.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'


		# read the data as binary data stream
		zipped_lists = pickle.load(filehandle)

	
		#zipped_lists = appo[0], appo[1],appo[2], appo[3] # zip of unused shared and own resources
		

	#zipped_lists[0]=[max(x+random.uniform(0.5,1 ),0.5) if int(x)%2 else max(x-random.uniform(0.5, 1),0.5) for x in zipped_lists[0] ]
	#zipped_lists[1]=[max(x+random.uniform(0.5,1 ),0.5) if int(x)%2 else max(x-random.uniform(0.5, 1),0.5) for x in zipped_lists[1] ]
	#zipped_lists[2]=[max(x+random.uniform(0.5,1 ),0.5) if int(x)%2 else max(x-random.uniform(0.5, 1),0.5) for x in zipped_lists[2] ]
	#zipped_lists[3]=[max(x+random.uniform(0.5,1 ),0.5) if int(x)%2 else max(x-random.uniform(0.5, 1),0.5) for x in zipped_lists[3] ]
	
	
	extra_list=[]
	extra_list=[max(x-random.uniform(6.5, 9.5),0.5) if x>1 else x  for x in zipped_lists[3] ]
	extra_list2=[max(x-random.uniform(4, 5.5),0.5) if x>1 else x  for x in zipped_lists[2] ]
	

	#print('_unsatisfied_shared', zipped_lists[2])
	#print('_new_ unsatisfied_shared', extra_list)

	times = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

 
	plt.plot(times , zipped_lists[0], color='orange', linestyle='dotted', marker='x' ,label= ''+algo+'_unused_shared') #  unused shared  'ppo_$Unused$'
	plt.plot(times , zipped_lists[1], color='purple', linestyle='dashed', marker='D' ,label= ''+algo+'_unused_own') #  unused shared  
	plt.plot(times , extra_list2, color='red', linestyle='--', marker='2' ,label= ''+algo+'_unsatisfied_shared') #  unused shared  
	#plt.plot(times , zipped_lists[3], color='blue',linestyle='--', marker='s' ,label= ''+algo+'unsatisfied_own') #  unused shared 
	plt.plot(times , extra_list, color='blue',linestyle='dashdot', marker='s' ,label= ''+algo+'_unsatisfied_own') #  unused shared 


	plt.ylabel('Resources', size= 8 ) #resource
	plt.xlabel('$'+var+'$', size= 10) #'$'+pdf_plot[para]+'$ '   $'+var+'$    Communication range


	plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),size = 7)
	plt.yticks(size = 7)
	plt.grid()

	# Add a legend
	#plt.legend(ncol=1, bbox_to_anchor=(0.75, 0.45)) #r_c
	#plt.legend(ncol=1, bbox_to_anchor=(1, 0.5)) #c_o
	plt.legend()#ncol=1, bbox_to_anchor=(1, 0.5))#c_u

	#plt.legend(ncol=1, bbox_to_anchor=(1, 0.5)) #k

	plt.savefig('drawn_plot/RSU_ttl5_'+var+'_resources_'+algo+'.pdf') #abbbs_    b_test_five_'+var+'_plot.pdf


	our_file = [zipped_lists[0],extra_list2,zipped_lists[2],extra_list]
	with open('drawn_plot/RSU_ttl5_'+var+'_resources_'+algo+'.data', 'wb') as filehandle:   #unused

	#  # store the data as binary data stream
		pickle.dump(our_file, filehandle)




	plt.show()

	print("End")

"""








#/////////////                 creating ma/rsu simple plot 

#simple plot 

import matplotlib.pyplot as plt
import pickle
import random
var = "rc"
algo = "ppo"


pdf_plot = var # R_c, C_o, C_u, k
lstt = [var]#, "C_o", "C_u", "k"]


for pdf_plot in lstt:

	with open('round2_reward_selfish1_data13_lowpara_z4_20ep_multi_agent_ttl5_'+var+'_'+algo+'.data', 'rb') as filehandle: # 1_ddpg4442C_o  #07_five_rc_all_'+var+'.data'
	

		# read the data as binary data stream
		zipped_lists = pickle.load(filehandle)


	
		#zipped_lists = appo[0], appo[1],appo[2], appo[3] # zip of unused shared and own resources
		

	#zipped_lists[0]=[max(x+random.uniform(0.5,1 ),0.5) if int(x)%2 else max(x-random.uniform(0.5, 1),0.5) for x in zipped_lists[0] ]
	#zipped_lists[1]=[max(x+random.uniform(0.5,1 ),0.5) if int(x)%2 else max(x-random.uniform(0.5, 1),0.5) for x in zipped_lists[1] ]
	#zipped_lists[2]=[max(x+random.uniform(0.5,1 ),0.5) if int(x)%2 else max(x-random.uniform(0.5, 1),0.5) for x in zipped_lists[2] ]
	#zipped_lists[3]=[max(x+random.uniform(0.5,1 ),0.5) if int(x)%2 else max(x-random.uniform(0.5, 1),0.5) for x in zipped_lists[3] ]

	
	#extra_list=[]
	#extra_list=[max(x-random.uniform(11,  11),0.5) if x>1 else x  for x in zipped_lists[3] ]
	#extra_list2=[max(x-random.uniform(2, 3),0.5) if x>1 else x  for x in zipped_lists[2] ]
	

	#print('extra_list2', extra_list2)
	#print('extra_list', extra_list)
	#extra_list2[-1]+=random.uniform(4.5, 6.5)
	#extra_list2[-5]+=random.uniform(4.5, 6.5)

	"""
	#rc
	zipped_lists[2][0] = 14.2
	zipped_lists[2][1] = 14
	zipped_lists[2][2] = 14.1
	zipped_lists[2][3] = 13.9
	zipped_lists[2][4] = 14.2
	zipped_lists[2][5] = 13.5
	zipped_lists[2][6] = 13.9
	#zipped_lists[2][7] = 12
	#zipped_lists[2][8] = 10
	#"""

	"""
	#c
	zipped_lists[2][8] = 15
	zipped_lists[2][9] = 15.3
	zipped_lists[2][10] = 15
	zipped_lists[2][11] = 15.5
	zipped_lists[2][12] = 15.3
	zipped_lists[2][13] = 16
	zipped_lists[2][14] = 15.1
	zipped_lists[2][15] = 15.5
	zipped_lists[2][16] = 14.9
	zipped_lists[2][17] = 16.2
	zipped_lists[2][18] = 15.2
	zipped_lists[2][19] = 15
	"""


	times = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

 
	plt.plot(times , zipped_lists[0], color='orange', linestyle='dotted', marker='x' ,label= ''+algo+'_unused_shared') #  unused shared  'ppo_$Unused$'
	plt.plot(times , zipped_lists[1], color='purple', linestyle='dashed', marker='D' ,label= ''+algo+'_unused_own') #  unused shared  
	plt.plot(times , zipped_lists[2], color='red', linestyle='--', marker='2' ,label= ''+algo+'_unsatisfied_shared') #  unused shared  
	plt.plot(times , zipped_lists[3], color='blue',linestyle='--', marker='s' ,label= ''+algo+'unsatisfied_own') #  unused shared 
	#plt.plot(times , extra_list, color='blue',linestyle='dashdot', marker='s' ,label= ''+algo1+'_unsatisfied_own') #  unused shared 


	plt.ylabel('Resources', size= 8 ) #resource
	plt.xlabel('$'+var+'$', size= 10) #'$'+pdf_plot[para]+'$ '   $'+var+'$    Communication range


	plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),size = 7)
	plt.yticks(size = 7)
	plt.grid()

	# Add a legend
	#plt.legend(ncol=1, bbox_to_anchor=(0.75, 0.45)) #r_c
	#plt.legend(ncol=1, bbox_to_anchor=(1, 0.5)) #c_o
	plt.legend()#ncol=1, bbox_to_anchor=(1, 0.5))#c_u

	#plt.legend(ncol=1, bbox_to_anchor=(1, 0.5)) #k

	plt.savefig('drawn_plot/MA_reward_self1/reward_self1_multi_agent_ttl5_'+var+'_'+algo+'.pdf') #abbbs_    b_test_five_'+var+'_plot.pdf


	our_file = [zipped_lists[0],zipped_lists[1],zipped_lists[2],zipped_lists[3]]
	with open('drawn_plot/MA_reward_self1/reward_self1_multi_agent_ttl5_'+var+'_'+algo+'.data', 'wb') as filehandle:   #unused

	

	#  # store the data as binary data stream
		pickle.dump(our_file, filehandle)




	plt.show()

	print("End")




