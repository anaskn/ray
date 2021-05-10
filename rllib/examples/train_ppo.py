import argparse

import ray
from ray import tune
from ray.rllib.agents import ppo
from my_env import ContentCaching
import pickle
import time
import numpy as np
import time 
import matplotlib.pyplot as plt 
from customclass import customExperimentClass
import ray
from ray import tune



if __name__ == "__main__":



	parser = argparse.ArgumentParser()
	parser.add_argument("--stop-iters", type=int, default= 50)#50)
	parser.add_argument("--stop-timesteps", type=int, default=900000)
	parser.add_argument("--stop-reward", type=float, default=0.001)
	parser.add_argument("--para", type=str, default="rc")
	parser.add_argument("--ttl_var", type=int, default=3)
	args = parser.parse_args()

	y=0
	parameters = [[y, 8, 8, 4] , [8, y, 8, 4], [8, 8, y, 4], [8, 8, 8, y]]

	pdf_plot = ["rc", "co", "cu", "k"] #["R_c", "C_o", "C_u", "k"]
	if args.para == "rc":
		para = 0
	if args.para == "co":
		para = 1
	if args.para == "cu":
		para = 2
	if args.para == "k":
		para = 3

	lst=0

	algo_reward_test = []
	algo_unused_shared = []
	algo_unused_own = []
	algo_unsatisfied_shared =[]
	algo_unsatisfied_own = []
	


	variable = [1,2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,55,60] #[1,10,20,60,150,400,700,1000] #

	ray.shutdown()
	ray.init(num_cpus=3)

	for x in range(len(variable)):


		parameters[para][para]= variable[x]

		all_unused_shared = []
		all_unused_own = []
		all_unsatisfied_shared = []
		all_unsatisfied_own = []



		for cpt in range(1,5):
			

			print("calcul of : "+pdf_plot[para], " for the value : ", variable[x]  )


			# Class instance
			exper = customExperimentClass(args.ttl_var, cpt, parameters[para], stop_iters=args.stop_iters) # ttl_var, cpt, variable
			# Train and save for 2 iterations
			checkpoint_path, results = exper.train()
			# Load saved and Test loaded
			reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test(checkpoint_path)


				

			
			#all_reward_test.append(reward_test)
			all_unused_shared.append(unused_shared)
			all_unused_own.append(unused_own)
			all_unsatisfied_shared.append(unsatisfied_shared)
			all_unsatisfied_own.append(unsatisfied_own)



		#all_reward_test = [(a + b + c + d ) / 4 for a,b,c,d  in zip(all_reward_test[0], all_reward_test[1], all_reward_test[2], all_reward_test[3])]
		#print("all_unused_shared   :::::::::::::::::   ", all_unused_shared)
		all_unused_shared = [(a + b + c + d ) / 4 for a,b,c,d  in zip(all_unused_shared[0], all_unused_shared[1], all_unused_shared[2], all_unused_shared[3])]
		all_unused_own = [(a + b + c + d ) / 4 for a,b,c,d  in zip(all_unused_own[0], all_unused_own[1], all_unused_own[2], all_unused_own[3])]
		all_unsatisfied_shared = [(a + b + c + d ) / 4 for a,b,c,d  in zip(all_unsatisfied_shared[0], all_unsatisfied_shared[1], all_unsatisfied_shared[2], all_unsatisfied_shared[3])]
		all_unsatisfied_own = [(a + b + c + d ) / 4 for a,b,c,d  in zip(all_unsatisfied_own[0], all_unsatisfied_own[1], all_unsatisfied_own[2], all_unsatisfied_own[3])]



		#algo_reward_test.append(total_reward_test)
		algo_unused_shared.append(np.mean(all_unused_shared))
		algo_unused_own.append(np.mean(all_unused_own))
		algo_unsatisfied_shared.append(np.mean(all_unsatisfied_shared))
		algo_unsatisfied_own.append(np.mean(all_unsatisfied_own))


		


	times = [1,2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,55,60]
	#times = [2,4,6,8,10,12,14,16,18,20]
	#print("algo_reward_test == ", algo_reward_test)
	#print("total_unused_shared == ", algo_unused_shared)
	


	#plt.plot(reward_train, color='blue', marker='v' ,label='DDPG Reward train') # print reward
	#plt.plot(reward_test, color='red', marker='*' ,label='DDPG Reward test') # print reward
	
	
	plt.plot(times , algo_unused_shared, color='orange', linestyle='dotted', marker='x' ,label='ppo_$Unused_{g}$') #  unused shared  'ppo_$Unused$'
	plt.plot(times , algo_unused_own, color='purple', linestyle='-', marker='+' ,label='ppo_$Unused_{o}$') # unused own 
	#plt.plot(times , algo_unused_shared[1], color='red', linestyle='dashed', marker='D' ,label='trpo_$Unused_{g}$') #  unused shared  
	#plt.plot(times , algo_unused_own[1], color='green', linestyle='dashdot', marker='*' ,label='trpo_$Unused_{o}$') # unused own 
	
	plt.ylabel('Unused caching resources', size= 8 ) #'$U_{nused}$' #Reward
	#plt.xlabel('Episode', size= 10)
	plt.xlabel('$'+pdf_plot[para]+'$', size= 10) #'$'+pdf_plot[para]+'$'

	plt.xticks(size = 7)
	plt.yticks(size = 7)

	# Add a legend
	plt.legend()

	#"""
	# save file .pdf
	plt.savefig('plot/03_unused_'+pdf_plot[para]+'_ppo.pdf') #relusigmoid


	#to stock data 
	our_file = [algo_unused_shared,algo_unused_own]#,algo_unused_shared[1],algo_unused_own[1]]#, algo_unused_shared[1], algo_unused_own[1], algo_unused_shared[2], algo_unused_own[2] ]
	with open('model/03_unused_'+pdf_plot[para]+'_ppo.data', 'wb') as filehandle: #07_five_rc_all_'+pdf_plot[para]+'
	#  # store the data as binary data stream
		pickle.dump(our_file, filehandle)
	
	#print("algo_unused_shared = ", len(algo_unused_shared))
	#"""
	
	#plt.show()

	plt.close()
	print("End")

	

	#plot only the last one 
	plt.plot(times , algo_unsatisfied_shared, color='orange', linestyle='dotted', marker='x' ,label='ppo_$Unsatisfied_{g}$') #  unused shared  
	plt.plot(times , algo_unsatisfied_own, color='purple', linestyle='-', marker='+' ,label='ppo_$Unsatisfied_{o}$') # unused own 
	#plt.plot(times , algo_unsatisfied_shared[1], color='red', linestyle='dashed', marker='D' ,label='trpo_$Unsatisfied_{g}$') #  unused shared  
	#plt.plot(times , algo_unsatisfied_own[1], color='green', linestyle='dashdot', marker='*' ,label='trpo_$Unsatisfied_{o}$') # unused own 
	

	plt.ylabel('Unsatisfied caching demands', size= 8 ) #'$U_{nused}$' #Reward
	plt.xlabel('$'+pdf_plot[para]+'$', size= 10)

	plt.xticks(size = 7)
	plt.yticks(size = 7)

	# Add a legend
	plt.legend()
	
	# save file .pdf
	#"""
	plt.savefig('plot/03_unsatisfied_'+pdf_plot[para]+'_ppo_plot.pdf') #relusigmoid


	#to stock data 
	our_file = [algo_unsatisfied_shared, algo_unsatisfied_own[0]]#,algo_unsatisfied_shared[1],algo_unsatisfied_own[1]]#, algo_unused_shared_cut[1], algo_unused_own_cut[1], algo_unused_shared_cut[2], algo_unused_own_cut[2]]
	with open('model/03_unsatisfied_'+pdf_plot[para]+'_ppo.data', 'wb') as filehandle: 
	  # store the data as binary data stream
		pickle.dump(our_file, filehandle)
	#"""
	

	#plt.show()

	plt.close()
	print("End")


	
	