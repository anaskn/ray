import argparse
import ray
from ray import tune
#from my_env import ContentCaching
import pickle
import time
import numpy as np
import time 
import matplotlib.pyplot as plt 
from caching20 import customExperimentClass

#import torch
#torch.set_deterministic(True)

if __name__ == "__main__":



	parser = argparse.ArgumentParser()
	parser.add_argument("--run", type=str, default="ppo")	
	parser.add_argument("--epochs", type=int, default= 50)#50)
	parser.add_argument("--stop-timesteps", type=int, default=99999900)
	parser.add_argument("--stop-reward", type=float, default=0.000001)
	parser.add_argument("--para", type=str, default="rc")
	parser.add_argument("--ttl_var", type=int, default=3)
	parser.add_argument("--cpu", type=int, default= 8)
	parser.add_argument("--gpu", type=float, default= 0)
	parser.add_argument("--lr", type=float, nargs="+", default=[1e-2])
	parser.add_argument("--activation", nargs="+", default= ["relu"])
	parser.add_argument('-l','--layer', type=int, nargs='+', required=True, action='append', help='layer list')
	parser.add_argument("--num_gpus_per_worker", type=float, default= 0)
	parser.add_argument("--num_workers", type=int, default= 0)

	args = parser.parse_args()

	ray.shutdown()
	ray.init()
              

	y=0
	parameters = [[y, 8, 4] , [8, y, 4], [8, 8, y]]

	pdf_plot = ["rc", "c", "k"] #["R_c", "C_o", "C_u", "k"]
	if args.para == "rc":
		para = 0
	if args.para == "c":
		para = 1
	if args.para == "k":
		para = 2

	lst=0


	algo_reward_test = []
	algo_unused_shared = []
	algo_unused_own = []
	algo_unsatisfied_shared =[]
	algo_unsatisfied_own = []

	max_algo_unused_shared = []
	max_algo_unused_own = []
	max_algo_unsatisfied_shared =[]
	max_algo_unsatisfied_own = []

	max_algo_unused_all = []
	max_algo_unsatisfied_all = []
	algo_unused_all =[]
	algo_unsatisfied_all = []

	algo_unused_all
	
	#variable =  [1,2,3,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20]# [2,4,6,8,10,12,14,16,18,20] #[1,10,20,60,150,400,700,1000] #
	variable =  [11,12,13,14,15,16,17,18,19,20]

	for x in range(len(variable)):

		parameters[para][para]= variable[x]

		all_unused_shared = []
		all_unused_own = []
		all_unsatisfied_shared = []
		all_unsatisfied_own = []

		for cpt in range(1,11):#11 ##3
			

			print("calcul of : "+pdf_plot[para], " for the value : ", variable[x]  )
 
			if args.run == "ppo" or args.run == "ddpg" or args.run == "appo" or args.run == "td3" or args.run == "a3c" or args.run == "impala":
				# Class instance
				exper = customExperimentClass(args.run, args.ttl_var, cpt, parameters[para], \
											fcnet_hidd_lst = args.layer, fcnet_act_lst = args.activation, lr_lst = args.lr, stop_iters=args.epochs, num_gpus=args.gpu, num_gpus_per_worker=args.num_gpus_per_worker, num_workers=args.num_workers) 									
				#checkpoint_path, results, lr, fc_hid, fc_act = exper.train(args.run)
				# Load saved and Test loaded
				#reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test(args.run, checkpoint_path, lr, fc_hid, fc_act)	
				all_in = exper.train(args.run)
				checkpoint_path=all_in[0]
				results= all_in[1]
				lr= all_in[2]
				fc_hid= all_in[3]
				frame_work = all_in[4]
				fc_act= all_in[5]


				if args.run== 'ppo' or args.run== 'appo':
					vf_loss=all_in[6]
					kl_target= all_in[7]
					clip=all_in[8]

				if args.run== 'ddpg' or args.run== 'td3':
					target_noise=all_in[6]
					lea_starts=all_in[7]
				if args.run== 'a2c' or args.run== 'a3c':
					vf_loss=all_in[6]
					grad=all_in[7]


				if args.run== 'ppo'  or args.run== 'appo':
					reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test(args.run ,checkpoint_path, lr, fc_hid, frame_work, fc_act, vf_loss, kl_target, clip, -1, -1, -1)
				if args.run== 'ddpg' or args.run== 'td3':
					reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test(args.run ,checkpoint_path, lr, fc_hid, frame_work, fc_act, -1,         -1,      -1,   target_noise, lea_starts, -1)
				if args.run== 'a2c' or args.run== 'a3c':
					reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test(args.run ,checkpoint_path, lr, fc_hid, frame_work, fc_act, vf_loss,    -1,      -1,   -1, -1,  grad)

	
			if args.run == "random" :
				# Class instance
				exper = customExperimentClass(args.run, args.ttl_var, cpt, parameters[para], \
											fcnet_hidd_lst =fc_hid, fcnet_act_lst =fc_act, lr_lst =args.lr, stop_iters=args.epochs, num_gpus=args.gpu, num_gpus_per_worker=args.num_gpus_per_worker, num_workers=args.num_workers) 									
				# Load saved and Test loaded
				reward, unused_shared ,unused_own, unsatisfied_shared, unsatisfied_own  = exper.test_random(args.run)		

			#all_reward_test.append(reward_test)
			all_unused_shared.append(unused_shared)
			all_unused_own.append(unused_own)
			all_unsatisfied_shared.append(unsatisfied_shared)
			all_unsatisfied_own.append(unsatisfied_own)

	
		mean_all_unused_shared = [(a + b + c + d + e + f + g + h + i + j) / 10 for a,b,c,d,e,f,g,h,i,j  in zip(all_unused_shared[0], all_unused_shared[1], \
			all_unused_shared[2], all_unused_shared[3], all_unused_shared[4],all_unused_shared[5], all_unused_shared[6], all_unused_shared[7], all_unused_shared[8], all_unused_shared[9])]
		
		mean_all_unused_own = [(a + b + c + d + e + f + g + h + i + j) / 10 for a,b,c,d,e,f,g,h,i,j  in zip(all_unused_own[0], all_unused_own[1], all_unused_own[2],\
			all_unused_own[3],all_unused_own[4], all_unused_own[5], all_unused_own[6], all_unused_own[7],all_unused_own[8], all_unused_own[9])]

		mean_all_unsatisfied_shared = [(a + b + c + d + e + f + g + h + i + j) / 10 for a,b,c,d,e,f,g,h,i,j  in zip(all_unsatisfied_shared[0], \
			all_unsatisfied_shared[1], all_unsatisfied_shared[2], all_unsatisfied_shared[3],all_unsatisfied_shared[4], all_unsatisfied_shared[5], all_unsatisfied_shared[6],all_unsatisfied_shared[7], all_unsatisfied_shared[8], all_unsatisfied_shared[9])]
		
		mean_all_unsatisfied_own = [(a + b + c + d + e + f + g + h + i + j) / 10 for a,b,c,d,e,f,g,h,i,j  in zip(all_unsatisfied_own[0], all_unsatisfied_own[1],\
			all_unsatisfied_own[2], all_unsatisfied_own[3], all_unsatisfied_own[4], all_unsatisfied_own[5], all_unsatisfied_own[6], all_unsatisfied_own[7], all_unsatisfied_own[8], all_unsatisfied_own[9])]
		
		"""
		mean_all_unused_shared = [(a + b  ) / 2 for a,b  in zip(all_unused_shared[0], all_unused_shared[1] )]
		
		mean_all_unused_own = [(a + b ) / 2 for a,b  in zip(all_unused_own[0], all_unused_own[1] )]

		mean_all_unsatisfied_shared = [(a + b ) / 2 for a,b  in zip(all_unsatisfied_shared[0], all_unsatisfied_shared[1])]
		
		mean_all_unsatisfied_own = [(a + b  ) / 2 for a,b  in zip(all_unsatisfied_own[0], all_unsatisfied_own[1] )]
		"""

		algo_unused_shared.append(np.mean(mean_all_unused_shared))
		algo_unused_own.append(np.mean(mean_all_unused_own))
		algo_unsatisfied_shared.append(np.mean(mean_all_unsatisfied_shared))
		algo_unsatisfied_own.append(np.mean(mean_all_unsatisfied_own))


		algo_unused_all.append(np.mean(mean_all_unused_shared)+np.mean(mean_all_unused_own))
		algo_unsatisfied_all.append(np.mean(mean_all_unsatisfied_shared)+np.mean(mean_all_unsatisfied_own))


		#print("mean_all_unused_shared[0] ", mean_all_unused_shared[0])

		"""		all_unused_shared = [max(a,b,c,d,e,f,g,h,i,j)  for a,b,c,d,e,f,g,h,i,j  in zip(all_unused_shared[0], all_unused_shared[1], \
			all_unused_shared[2], all_unused_shared[3], all_unused_shared[4],all_unused_shared[5], all_unused_shared[6], all_unused_shared[7], all_unused_shared[8], all_unused_shared[9])]
		
		all_unused_own = [max(a,b,c,d,e,f,g,h,i,j) for a,b,c,d,e,f,g,h,i,j  in zip(all_unused_own[0], all_unused_own[1], all_unused_own[2],\
			all_unused_own[3],all_unused_own[4], all_unused_own[5], all_unused_own[6], all_unused_own[7],all_unused_own[8], all_unused_own[9])]

		all_unsatisfied_shared = [max(a,b,c,d,e,f,g,h,i,j) for a,b,c,d,e,f,g,h,i,j  in zip(all_unsatisfied_shared[0], \
			all_unsatisfied_shared[1], all_unsatisfied_shared[2], all_unsatisfied_shared[3],all_unsatisfied_shared[4], all_unsatisfied_shared[5], all_unsatisfied_shared[6],all_unsatisfied_shared[7], all_unsatisfied_shared[8], all_unsatisfied_shared[9])]
		
		all_unsatisfied_own = [max(a,b,c,d,e,f,g,h,i,j) for a,b,c,d,e,f,g,h,i,j  in zip(all_unsatisfied_own[0], all_unsatisfied_own[1],\
			all_unsatisfied_own[2], all_unsatisfied_own[3], all_unsatisfied_own[4], all_unsatisfied_own[5], all_unsatisfied_own[6], all_unsatisfied_own[7], all_unsatisfied_own[8], all_unsatisfied_own[9])]
		
		all_unused_shared = [max(a,b)  for a,b  in zip(all_unused_shared[0], all_unused_shared[1] )]
		
		all_unused_own = [max(a,b) for a,b  in zip(all_unused_own[0], all_unused_own[1] )]

		all_unsatisfied_shared = [max(a,b) for a,b  in zip(all_unsatisfied_shared[0], all_unsatisfied_shared[1] )]
		
		all_unsatisfied_own = [max(a,b) for a,b in zip(all_unsatisfied_own[0], all_unsatisfied_own[1] )]
		


		max_algo_unused_shared.append(np.mean(all_unused_shared))
		max_algo_unused_own.append(np.mean(all_unused_own))
		max_algo_unsatisfied_shared.append(np.mean(all_unsatisfied_shared))
		max_algo_unsatisfied_own.append(np.mean(all_unsatisfied_own))

		max_algo_unused_all.append(np.mean(all_unused_shared)+np.mean(all_unused_own))
		max_algo_unsatisfied_all.append(np.mean(all_unsatisfied_shared)+np.mean(all_unsatisfied_own))
		"""



	#times = [1,2,3,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20]# [2,4,6,8,10,12,14,16,18,20]#
	times =  [11,12,13,14,15,16,17,18,19,20]
	
	#plt.plot(times , algo_unused_shared, color='orange', linestyle='dotted', marker='x' ,label=args.run+'_$Unused_{g}$') #  unused shared  'ppo_$Unused$'
	#plt.plot(times , algo_unused_own, color='purple', linestyle='-', marker='+' ,label=args.run+'_$Unused_{o}$') # unused own 
	#plt.plot(times , max_algo_unused_shared, color='red', linestyle='dashed', marker='v' ,label='max_'+args.run+'_$Unused_{g}$') #  unused shared  'ppo_$Unused$'
	#plt.plot(times , max_algo_unused_own, color='blue', linestyle='--', marker='D' ,label='max_'+args.run+'_$Unused_{o}$') # unused own 


	#plot avg of unused own + shared // unsatisfied own + shared
	plt.plot(times , algo_unused_shared, color='orange', linestyle='dotted', marker='x' ,label=args.run+'_$Unused_shared') #  unused shared  
	plt.plot(times , algo_unused_own, color='purple', linestyle='-', marker='+' ,label=args.run+'_$Unused_own') # unused own 
	plt.plot(times , algo_unsatisfied_shared, color='red', linestyle='dashed', marker='v' ,label=args.run+'_$Unsatisfied_shared') #  unused shared  'ppo_$Unused$'
	plt.plot(times , algo_unsatisfied_own, color='blue', linestyle='--', marker='D' ,label=args.run+'_$Unsatisfied_own') # unused own 


	#plt.plot(times , algo_unused_all, color='orange', linestyle='dotted', marker='x' ,label=args.run+'_$Unused$') #  unused shared  'ppo_$Unused$'
	#plt.plot(times , algo_unsatisfied_all, color='purple', linestyle='-', marker='+' ,label=args.run+'_$unsatisfied$') # unused own 

	plt.ylabel('Resources', size= 8 ) #'$U_{nused}$' #Reward
	plt.xlabel('$'+pdf_plot[para]+'$', size= 10) #'$'+pdf_plot[para]+'$'

	plt.xticks(size = 7)
	plt.yticks(size = 7)

	# Add a legend
	plt.legend()
	plt.grid()

	# save file .pdf
	plt.savefig('plot/z3_20ep_resources_part2_'+pdf_plot[para]+'_'+args.run+'.pdf')  #unused

	#to stock data 
	#our_file = [algo_unused_shared,algo_unused_own,max_algo_unused_shared,max_algo_unused_own]
	our_file = [algo_unused_shared, algo_unused_own, algo_unsatisfied_shared, algo_unsatisfied_own]
	with open('model/z3_20ep_resources_part2_'+pdf_plot[para]+'_'+args.run+'.data', 'wb') as filehandle:   #unused
	#  # store the data as binary data stream
		pickle.dump(our_file, filehandle)
	
	#plt.show()
	plt.close()
	print("End")
	"""
	#plot only the last one 
	#plt.plot(times , algo_unsatisfied_shared, color='orange', linestyle='dotted', marker='x' ,label=args.run+'_$Unsatisfied_{g}$') #  unused shared  
	#plt.plot(times , algo_unsatisfied_own, color='purple', linestyle='-', marker='+' ,label=args.run+'_$Unsatisfied_{o}$') # unused own 
	#plt.plot(times , max_algo_unsatisfied_shared, color='red', linestyle='dashed', marker='v' ,label='max_'+args.run+'_$Unused_{g}$') #  unused shared  'ppo_$Unused$'
	#plt.plot(times , max_algo_unsatisfied_own, color='blue', linestyle='--', marker='D' ,label='max_'+args.run+'_$Unused_{o}$') # unused own 


	#plot avg of unused own + shared // unsatisfied own + shared
	plt.plot(times , max_algo_unused_shared, color='orange', linestyle='dotted', marker='x' ,label='max_'+args.run+'_$Unused_shared') #  unused shared  
	plt.plot(times , max_algo_unused_own, color='purple', linestyle='-', marker='+' ,label='max_'+args.run+'_$Unused_own') # unused own 
	plt.plot(times , max_algo_unsatisfied_shared, color='red', linestyle='dashed', marker='v' ,label='max_'+args.run+'_$Unsatisfied_shared') #  unused shared  'ppo_$Unused$'
	plt.plot(times , max_algo_unsatisfied_own, color='blue', linestyle='--', marker='D' ,label='max_'+args.run+'_$Unsatisfied_own') # unused own
	

	#plt.plot(times , max_algo_unused_all, color='orange', linestyle='dotted', marker='x' ,label=args.run+'_$Unused$') #  unused shared  
	#plt.plot(times , max_algo_unsatisfied_all, color='purple', linestyle='-', marker='+' ,label=args.run+'_$Unsatisfied$') # unused own 
	
	plt.ylabel('Resources', size= 8 ) 
	plt.xlabel('$'+pdf_plot[para]+'$', size= 10)

	plt.xticks(size = 7)
	plt.yticks(size = 7)

	# Add a legend
	plt.legend()
	
	# save file .pdf
	
	plt.savefig('plot/z_20ep_max_resources_'+pdf_plot[para]+'_'+args.run+'.pdf')  #unsatisfied
	#to stock data 
	#our_file = [algo_unsatisfied_shared, algo_unsatisfied_own,max_algo_unsatisfied_shared, max_algo_unsatisfied_own]
	our_file = [max_algo_unused_shared, max_algo_unused_own, max_algo_unsatisfied_shared, max_algo_unsatisfied_own]

	with open('model/z_20ep_max_resources_'+pdf_plot[para]+'_'+args.run+'.data', 'wb') as filehandle:   #unsatisfied 
	  # store the data as binary data stream
		pickle.dump(our_file, filehandle)
	
	#plt.show()
	plt.close()
	print("End")
	"""

	


	
	
