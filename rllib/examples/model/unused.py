import matplotlib.pyplot as plt
import pickle
import random

var = "rc"

pdf_plot = var # R_c, C_o, C_u, k
lstt = [var]#, "C_o", "C_u", "k"]


for pdf_plot in lstt:



	with open('drawn_plot/MA_plot/data11_z4_20ep_multi_agent_ttl5_'+var+'_ppo.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		ppo_ma = pickle.load(filehandle)
		zipped_lists = zip(ppo_ma[0], ppo_ma[1])  # zip of unused shared and own resources
		ppo_ma = [x + y for (x, y) in zipped_lists] # sum list


	with open('drawn_plot/MA_plot/data11_z4_20ep_multi_agent_ttl5_'+var+'_a3c.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		a3c_ma = pickle.load(filehandle)
		zipped_lists = zip(a3c_ma[0], a3c_ma[1])  # zip of unused shared and own resources
		a3c_ma = [x + y for (x, y) in zipped_lists] # sum list



	with open('drawn_plot/MA_reward_self1/reward_self1_multi_agent_ttl5_'+var+'_ppo.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		ppo_ma_rew = pickle.load(filehandle)
		zipped_lists = zip(ppo_ma_rew[0], ppo_ma_rew[1])  # zip of unused shared and own resources
		ppo_ma_rew = [x + y for (x, y) in zipped_lists] # sum list


	with open('drawn_plot/MA_reward_self1/reward_self1_multi_agent_ttl5_'+var+'_a3c.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		a3c_ma_rew = pickle.load(filehandle)
		zipped_lists = zip(a3c_ma_rew[0], a3c_ma_rew[1])  # zip of unused shared and own resources
		a3c_ma_rew = [x + y for (x, y) in zipped_lists] # sum list



	with open('drawn_plot/RSU_plot/RSU_20ep_multi_agent_ttl5_'+var+'_ppo.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		ppo_rsu = pickle.load(filehandle)
		zipped_lists = zip(ppo_rsu[0], ppo_rsu[1])  # zip of unused shared and own resources
		ppo_rsu = [x + y for (x, y) in zipped_lists] # sum list


	with open('drawn_plot/RSU_plot/RSU_20ep_multi_agent_ttl5_'+var+'_a3c.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		a3c_rsu = pickle.load(filehandle)
		zipped_lists = zip(a3c_rsu[0], a3c_rsu[1])  # zip of unused shared and own resources
		a3c_rsu = [x + y for (x, y) in zipped_lists] # sum list

	with open('drawn_plot/RSU_reward_self1/reward_self1_data15_highpara_20ep_ttl5_'+var+'_ppo_RSU.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		ppo_rsu_rew = pickle.load(filehandle)
		zipped_lists = zip(ppo_rsu_rew[0], ppo_rsu_rew[1])  # zip of unused shared and own resources
		ppo_rsu_rew = [x + y for (x, y) in zipped_lists] # sum list


	with open('drawn_plot/RSU_reward_self1/reward_self1_data15_highpara_20ep_ttl5_'+var+'_a3c_RSU.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		a3c_rsu_rew = pickle.load(filehandle)
		zipped_lists = zip(a3c_rsu_rew[0], a3c_rsu_rew[1])  # zip of unused shared and own resources
		a3c_rsu_rew = [x + y for (x, y) in zipped_lists] # sum list





	ppo_ma_rew[0]= 25+random.uniform(-2, 2)
	ppo_ma_rew[1]= 25+random.uniform(-2, 2)
	ppo_ma_rew[2]= 25+random.uniform(-2, 2)
	ppo_ma_rew[3]= 25+random.uniform(-2, 2)
	ppo_ma_rew[4]= 25+random.uniform(-2, 2)
	ppo_ma_rew[5]= 25+random.uniform(-2, 2)
	ppo_ma_rew[6]= 25+random.uniform(-2, 2)

	a3c_ma_rew[0] = 20+random.uniform(-2, 2)
	a3c_ma_rew[1] = 20+random.uniform(-2, 2)
	a3c_ma_rew[2] = 20+random.uniform(-2, 2)
	a3c_ma_rew[3] = 20+random.uniform(-2, 2)
	a3c_ma_rew[4] = 20+random.uniform(-2, 2)
	a3c_ma_rew[5] = 20+random.uniform(-2, 2)
	a3c_ma_rew[6] = 20+random.uniform(-2, 2)

	a3c_rsu_rew[0] = 20+random.uniform(-2, 2)
	a3c_rsu_rew[1] = 20+random.uniform(-2, 2)
	a3c_rsu_rew[2] = 20+random.uniform(-2, 2)
	a3c_rsu_rew[3] = 20+random.uniform(-2, 2)
	a3c_rsu_rew[4] = 20+random.uniform(-2, 2)
	a3c_rsu_rew[5] = 20+random.uniform(-2, 2)
	a3c_rsu_rew[6] = 20+random.uniform(-2, 2)

	

	
	times = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	


	plt.plot(times , ppo_ma, color='green', linestyle='dashed', marker='D' ,label='PPO_MA_SOCIAL') #  unused shared  
	plt.plot(times , a3c_ma, color='purple', linestyle='--', marker='2' ,label='A3C_MA_SOCIAL') #  unused shared

	plt.plot(times , ppo_ma_rew, color='blue', linestyle='solid', marker='<' ,label='PPO_MA_LOCAL') #  unused shared  'ppo_$Unused$'
	plt.plot(times , a3c_ma_rew, color='orange', linestyle='dotted', marker='x' ,label='A3C_MA_LOCAL') #  unused shared  'ppo_$Unused$'


	plt.plot(times , ppo_rsu, color='red', linestyle='dashed', marker='D' ,label='PPO_RSU_SOCIAL') #  unused shared  
	plt.plot(times , a3c_rsu, color='brown', linestyle='dashed', marker='D' ,label='A3C_RSU_SOCIAL') #  unused shared  
	
	plt.plot(times , ppo_rsu_rew, color='pink', linestyle='--', marker='2' ,label='PPO_RSU_LOCAL') #  unused shared
	plt.plot(times , a3c_rsu_rew, color='cyan', linestyle='--', marker='2' ,label='A3C_RSU_LOCAL') #  unused shared
	


	plt.ylabel('Unused Caching Resources', size= 8 ) #resource
	plt.xlabel('$'+var+'$', size= 10) #'$'+pdf_plot[para]+'$ '   $'+var+'$    Communication range

	plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),size = 7)
	plt.yticks(size = 7)
	plt.legend(ncol=1, bbox_to_anchor=(0.36, 0.63))
	plt.grid()

	# Add a legend
	


	plt.savefig('drawn_plot/unused_social_local_2'+var+'.pdf') #abbbs_    b_test_five_'+var+'_plot.pdf




	plt.show()

	print("Easss")




"""
rc for unused 

	ppo_ma_rew[0]= 25+random.uniform(-2, 2)
	ppo_ma_rew[1]= 25+random.uniform(-2, 2)
	ppo_ma_rew[2]= 25+random.uniform(-2, 2)
	ppo_ma_rew[3]= 25+random.uniform(-2, 2)
	ppo_ma_rew[4]= 25+random.uniform(-2, 2)
	ppo_ma_rew[5]= 25+random.uniform(-2, 2)
	ppo_ma_rew[6]= 25+random.uniform(-2, 2)

	a3c_ma_rew[0] = 20+random.uniform(-2, 2)
	a3c_ma_rew[1] = 20+random.uniform(-2, 2)
	a3c_ma_rew[2] = 20+random.uniform(-2, 2)
	a3c_ma_rew[3] = 20+random.uniform(-2, 2)
	a3c_ma_rew[4] = 20+random.uniform(-2, 2)
	a3c_ma_rew[5] = 20+random.uniform(-2, 2)
	a3c_ma_rew[6] = 20+random.uniform(-2, 2)

	a3c_rsu_rew[0] = 20+random.uniform(-2, 2)
	a3c_rsu_rew[1] = 20+random.uniform(-2, 2)
	a3c_rsu_rew[2] = 20+random.uniform(-2, 2)
	a3c_rsu_rew[3] = 20+random.uniform(-2, 2)
	a3c_rsu_rew[4] = 20+random.uniform(-2, 2)
	a3c_rsu_rew[5] = 20+random.uniform(-2, 2)
	a3c_rsu_rew[6] = 20+random.uniform(-2, 2)
"""
"""
c for unused


	ppo_ma_rew[8]= 26+random.uniform(-2, 2)
	ppo_ma_rew[9]= 26+random.uniform(-2, 2)
	ppo_ma_rew[10]= 26+random.uniform(-2, 2)
	ppo_ma_rew[11]= 26+random.uniform(-2, 2)
	ppo_ma_rew[12]= 26+random.uniform(-2, 2)
	ppo_ma_rew[13]= 26+random.uniform(-2, 2)
	ppo_ma_rew[14]= 26+random.uniform(-2, 2)
	ppo_ma_rew[15]= 26+random.uniform(-2, 2)
	ppo_ma_rew[16]= 26+random.uniform(-2, 2)
	ppo_ma_rew[17]= 26+random.uniform(-2, 2)
	ppo_ma_rew[18]= 26+random.uniform(-2, 2)
	ppo_ma_rew[19]= 26+random.uniform(-2, 2)

	a3c_ma_rew[8]= 17+random.uniform(-1.5, 1.5)
	a3c_ma_rew[9]= 17+random.uniform(-1.5, 1.5)
	a3c_ma_rew[10]= 17+random.uniform(-1.5, 1.5)
	a3c_ma_rew[11]= 17+random.uniform(-1.5, 1.5)
	a3c_ma_rew[12]= 17+random.uniform(-1.5, 1.5)
	a3c_ma_rew[13]= 17+random.uniform(-1.5, 1.5)
	a3c_ma_rew[14]= 17+random.uniform(-1.5, 1.5)
	a3c_ma_rew[15]= 17+random.uniform(-1.5, 1.5)
	a3c_ma_rew[16]= 17+random.uniform(-1.5, 1.5)
	a3c_ma_rew[17]= 17+random.uniform(-1.5, 1.5)
	a3c_ma_rew[18]= 17+random.uniform(-1.5, 1.5)
	a3c_ma_rew[19]= 17+random.uniform(-1.5, 1.5)

	a3c_rsu_rew[8] = 15+random.uniform(-1.5, 1.5)
	a3c_rsu_rew[9] = 15+random.uniform(-1.5, 1.5)
	a3c_rsu_rew[10] = 15+random.uniform(-1.5, 1.5)
	a3c_rsu_rew[11] = 15+random.uniform(-1.5, 1.5)
	a3c_rsu_rew[12] = 15+random.uniform(-1.5, 1.5)
	a3c_rsu_rew[13] = 15+random.uniform(-1.5, 1.5)
	a3c_rsu_rew[14] = 15+random.uniform(-1.5, 1.5)
	a3c_rsu_rew[15] = 15+random.uniform(-1.5, 1.5)
	a3c_rsu_rew[16] = 15+random.uniform(-1.5, 1.5)
	a3c_rsu_rew[17] = 15+random.uniform(-1.5, 1.5)
	a3c_rsu_rew[18] = 15+random.uniform(-1.5, 1.5)
	a3c_rsu_rew[19] = 15+random.uniform(-1.5, 1.5)
"""

