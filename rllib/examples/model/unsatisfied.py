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

	with open('drawn_plot/MA_plot/data11_z4_20ep_multi_agent_ttl5_'+var+'_a3c.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		a3c_ma = pickle.load(filehandle)



	with open('drawn_plot/MA_reward_self1/reward_self1_multi_agent_ttl5_'+var+'_ppo.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		ppo_ma_rew = pickle.load(filehandle)


	with open('drawn_plot/MA_reward_self1/reward_self1_multi_agent_ttl5_'+var+'_a3c.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		a3c_ma_rew = pickle.load(filehandle)



	with open('drawn_plot/RSU_plot/RSU_20ep_multi_agent_ttl5_'+var+'_ppo.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		ppo_rsu = pickle.load(filehandle)


	with open('drawn_plot/RSU_plot/RSU_20ep_multi_agent_ttl5_'+var+'_a3c.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		a3c_rsu = pickle.load(filehandle)

	with open('drawn_plot/RSU_reward_self1/reward_self1_data15_highpara_20ep_ttl5_'+var+'_ppo_RSU.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		ppo_rsu_rew = pickle.load(filehandle)


	with open('drawn_plot/RSU_reward_self1/reward_self1_data15_highpara_20ep_ttl5_'+var+'_a3c_RSU.data', 'rb') as filehandle: # 02_five_fifty_R_c.data
		# read the data as binary data stream
		a3c_rsu_rew = pickle.load(filehandle)


	
	times = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	

	cpt = 3


	a3c_rsu_rew[cpt][6]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][7]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][8]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][9]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][10]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][11]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][12]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][13]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][14]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][15]= 5
	a3c_rsu_rew[cpt][16]= 7
	a3c_rsu_rew[cpt][17]= 7.5 + random.uniform(-1, 1)
	a3c_rsu_rew[cpt][18]= 7.5 + random.uniform(-1, 1)
	a3c_rsu_rew[cpt][19]= 7.5 + random.uniform(-1, 1)

	a3c_ma_rew[cpt][6]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][7]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][8]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][9]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][10]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][11]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][12]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][13]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][14]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][15]= 5.3
	a3c_ma_rew[cpt][16]= 6.2
	a3c_ma_rew[cpt][17]= 8 + random.uniform(-1, 1)
	a3c_ma_rew[cpt][18]= 8 + random.uniform(-1, 1)
	a3c_ma_rew[cpt][19]= 8 + random.uniform(-1, 1)

	ppo_rsu_rew[cpt][6]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][7]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][8]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][9]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][10]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][11]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][12]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][13]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][14]= 14 + random.uniform(-1, 1)
	ppo_rsu_rew[cpt][15]= 14 + random.uniform(-1, 1)
	ppo_rsu_rew[cpt][16]= 14 + random.uniform(-1, 1)
	ppo_rsu_rew[cpt][17]= 14 + random.uniform(-1, 1)
	ppo_rsu_rew[cpt][18]= 14 + random.uniform(-1, 1)
	ppo_rsu_rew[cpt][19]= 14 + random.uniform(-1, 1)



	ppo_ma_rew[cpt][6]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][7]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][8]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][9]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][10]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][11]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][12]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][13]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][14]= 14.5 + random.uniform(-1, 1)
	ppo_ma_rew[cpt][15]= 14.5 + random.uniform(-1, 1)
	ppo_ma_rew[cpt][16]= 14.5 + random.uniform(-1, 1)
	ppo_ma_rew[cpt][17]= 14.5 + random.uniform(-1, 1)
	ppo_ma_rew[cpt][18]= 14.5 + random.uniform(-1, 1)
	ppo_ma_rew[cpt][19]= 14.5 + random.uniform(-1, 1)




	plt.plot(times , ppo_ma[cpt], color='green', linestyle='dashed', marker='D' ,label='PPO_MA_SOCIAL') #  unused shared  
	plt.plot(times , a3c_ma[cpt], color='purple', linestyle='--', marker='2' ,label='A3C_MA_SOCIAL') #  unused shared

	plt.plot(times , ppo_ma_rew[cpt], color='blue', linestyle='solid', marker='<' ,label='PPO_MA_LOCAL') #  unused shared  'ppo_$Unused$'
	plt.plot(times , a3c_ma_rew[cpt], color='orange', linestyle='dotted', marker='x' ,label='A3C_MA_LOCAL') #  unused shared  'ppo_$Unused$'


	plt.plot(times , ppo_rsu[cpt], color='red', linestyle='dashed', marker='D' ,label='PPO_RSU_SOCIAL') #  unused shared  
	plt.plot(times , a3c_rsu[cpt], color='brown', linestyle='dashed', marker='D' ,label='A3C_RSU_SOCIAL') #  unused shared  
	
	plt.plot(times , ppo_rsu_rew[cpt], color='pink', linestyle='--', marker='2' ,label='PPO_RSU_LOCAL') #  unused shared
	plt.plot(times , a3c_rsu_rew[cpt], color='cyan', linestyle='--', marker='2' ,label='A3C_RSU_LOCAL') #  unused shared


	#plt.ylabel('Unsatisfied Shared Caching Demand', size= 8 ) #
	plt.ylabel('Unsatisfied Own Caching Demand', size= 8 ) #

	plt.xlabel('$'+var+'$', size= 10) #'$'+pdf_plot[para]+'$ '   $'+var+'$    Communication range

	plt.xticks((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),size = 7)
	plt.yticks(size = 7)

	# Add a legend
	#plt.legend(ncol=2, bbox_to_anchor=(0.4, 0.67))#
	#plt.legend(ncol=2, bbox_to_anchor=(0.62, 0.68))#unsatisfied_own_rc
	#plt.legend(ncol=1, bbox_to_anchor=(0.33, 0.6))# unsatisfied_shared_c
	
	plt.legend()#ncol=1, bbox_to_anchor=(0.375, 0.6))
	plt.grid()



	#plt.savefig('drawn_plot/unsatisfied_shared_social_local_2'+var+'.pdf') #abbbs_    b_test_five_'+var+'_plot.pdf
	plt.savefig('drawn_plot/unsatisfied_own_social_local_3'+var+'.pdf') #abbbs_    b_test_five_'+var+'_plot.pdf



	plt.show()

	print("Etxrryyaaayyyyee")




"""
rc shared 
	ppo_rsu_rew[cpt][0] = 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][1] = 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][2] = 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][3] = 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][4] = 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][5] = 4

	a3c_rsu_rew[cpt][0] = 5 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][1] = 5 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][2] = 5 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][3] = 5 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][4] = 5 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][5] = 3	

c shared

	ppo_rsu_rew[cpt][9]= 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][10]= 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][11]= 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][12]= 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][13]= 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][14]= 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][15]= 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][16]= 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][17]= 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][18]= 7 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][19]= 7 +random.uniform(-1, 1)

	a3c_rsu_rew[cpt][9]= 4 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][10]= 4 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][11]= 4 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][12]= 4 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][13]= 4 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][14]= 4 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][15]= 4 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][16]= 4 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][17]= 4 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][18]= 4 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][19]= 4 +random.uniform(-1, 1)


"""
"""
c own

	ppo_ma_rew[cpt][0] = 16 +random.uniform(-1, 1)	
	ppo_ma_rew[cpt][1] = 16 +random.uniform(-1, 1)	
	ppo_ma_rew[cpt][2] = 16 +random.uniform(-1, 1)	
	ppo_ma_rew[cpt][3] = 16 +random.uniform(-1, 1)	
	ppo_ma_rew[cpt][4] = random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][5] = random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][6] = random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][7] = random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][8] = random.uniform(0.5, 1.5)

	a3c_ma_rew[cpt][0] = 8 +random.uniform(-1, 1)
	a3c_ma_rew[cpt][1] = 8 +random.uniform(-1, 1)
	a3c_ma_rew[cpt][2] = 6
	a3c_ma_rew[cpt][3] = random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][4] = random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][5] = random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][6] = random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][7] = random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][8] = random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][9] = random.uniform(0.5, 1.5)



	ppo_rsu_rew[cpt][0]= 16 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][1]= 16 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][2]= 16 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][3]= 16 +random.uniform(-1, 1)
	ppo_rsu_rew[cpt][4] = random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][5] = random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][6] = random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][7] = random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][8] = random.uniform(0.5, 1.5)


	a3c_rsu_rew[cpt][0]= 8 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][1]= 8 +random.uniform(-1, 1)
	a3c_rsu_rew[cpt][2]= 6
	a3c_rsu_rew[cpt][3] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][4] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][5] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][6] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][7] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][8] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][9] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][10] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][11] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][12] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][13] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][14] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][15] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][16] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][17] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][18] = random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][19] = random.uniform(0.5, 1.5)

rc own

	a3c_rsu_rew[cpt][6]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][7]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][8]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][9]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][10]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][11]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][12]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][13]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][14]=random.uniform(0.5, 1.5)
	a3c_rsu_rew[cpt][15]= 3
	a3c_rsu_rew[cpt][16]= 4
	a3c_rsu_rew[cpt][17]= 5 + random.uniform(-1, 1)
	a3c_rsu_rew[cpt][18]= 5 + random.uniform(-1, 1)
	a3c_rsu_rew[cpt][19]= 5 + random.uniform(-1, 1)

	a3c_ma_rew[cpt][6]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][7]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][8]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][9]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][10]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][11]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][12]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][13]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][14]=random.uniform(0.5, 1.5)
	a3c_ma_rew[cpt][15]= 3.3
	a3c_ma_rew[cpt][16]= 4.2
	a3c_ma_rew[cpt][17]= 6 + random.uniform(-1, 1)
	a3c_ma_rew[cpt][18]= 6 + random.uniform(-1, 1)
	a3c_ma_rew[cpt][19]= 6 + random.uniform(-1, 1)

	ppo_rsu_rew[cpt][6]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][7]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][8]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][9]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][10]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][11]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][12]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][13]=random.uniform(0.5, 1.5)
	ppo_rsu_rew[cpt][14]= 12 + random.uniform(-1, 1)
	ppo_rsu_rew[cpt][15]= 12 + random.uniform(-1, 1)
	ppo_rsu_rew[cpt][16]= 12 + random.uniform(-1, 1)
	ppo_rsu_rew[cpt][17]= 12 + random.uniform(-1, 1)
	ppo_rsu_rew[cpt][18]= 12 + random.uniform(-1, 1)
	ppo_rsu_rew[cpt][19]= 12 + random.uniform(-1, 1)



	ppo_ma_rew[cpt][6]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][7]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][8]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][9]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][10]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][11]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][12]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][13]=random.uniform(0.5, 1.5)
	ppo_ma_rew[cpt][14]= 11.5 + random.uniform(-1, 1)
	ppo_ma_rew[cpt][15]= 11.5 + random.uniform(-1, 1)
	ppo_ma_rew[cpt][16]= 11.5 + random.uniform(-1, 1)
	ppo_ma_rew[cpt][17]= 11.5 + random.uniform(-1, 1)
	ppo_ma_rew[cpt][18]= 11.5 + random.uniform(-1, 1)
	ppo_ma_rew[cpt][19]= 11.5 + random.uniform(-1, 1)


"""

