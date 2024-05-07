from __future__ import print_function
def main():
	opt = args.args()

	print("GPU information")
	print(torch.cuda.is_available())
	print(torch.cuda.current_device())
	print(torch.cuda.device_count())
	print(torch.cuda.get_device_name(0))
	# print(torch.cuda.get_device_name(1))
	deviceee = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('Using device:', deviceee)
	print()

	if opt.load_dir: 
		assert os.path.isdir(opt.load_dir)
		opt.save_dir = opt.load_dir
	else: 			  		   		
		opt.save_dir = '{}/{}'.format(opt.save_dir, "NLNL5")
	try:
		os.makedirs(opt.save_dir)
	except OSError:
		pass
	cudnn.benchmark = True

	print(opt.save_dir)

	logger = logging.getLogger("ydk_logger")
	fileHandler = logging.FileHandler(opt.save_dir+'/train.log')
	streamHandler = logging.StreamHandler()

	logger.addHandler(fileHandler)
	logger.addHandler(streamHandler)

	logger.setLevel(logging.INFO)
	logger.info(opt)

	print("cpu num")
	logger.info(os.cpu_count())

	##
	# Computing mean
	# Changed
	# trainset = dset.ImageFolder(root='{}/{}/train'.format(opt.dataroot, opt.dataset), 
	# 	transform=transforms.ToTensor())

	# Test: 10000
	# Train: 45000
	# Validation: 5000

	# get data
	data = pd.read_csv('./real_data/Data_Entry_2017.csv') #.head(1000) # here
	data_image_paths = {os.path.basename(x): x for x in glob(os.path.join('real_data', 'images*', '*', '*.png'))}
	print('Scans found:', len(data_image_paths), ', Total Headers', data.shape[0])

	data['path'] = data['Image Index'].map(data_image_paths.get)

	# TODO: what to do with no label
	# data['Finding Labels'] = data['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
	# print(data['Finding Labels'].str.contains("\|"))
	# print(data['Finding Labels'])

	# ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
	# [ 1.5399,         5.9636,          6.1887,       13.1200,  1.7587,       9.7910,     8.7467,   82.0000,      0.6521,     2.8398,    0.0971,     2.0061,         5.9099,          23.4286,       3.9281]
	# 		+														+												+			  						+

	# Added: drop no finding data['Finding Labels'] and ['Hernia']
	# data = data.drop(data[data['Finding Labels'] == "No Finding"].index)

	data = data.drop(data[data['Finding Labels'].str.contains("\|") == True].index) # drop multi label

	data = data.drop(data[data['Finding Labels'].str.contains("Cardiomegaly") == True].index)
	data = data.drop(data[data['Finding Labels'].str.contains("Consolidation") == True].index)
	data = data.drop(data[data['Finding Labels'].str.contains("Edema") == True].index)
	data = data.drop(data[data['Finding Labels'].str.contains("Emphysema") == True].index)
	
	data = data.drop(data[data['Finding Labels'].str.contains("Fibrosis") == True].index)
	data = data.drop(data[data['Finding Labels'].str.contains("Hernia") == True].index)
	
	# data = data.drop(data[data['Finding Labels'].str.contains("Mass") == True].index)
	data = data.drop(data[data['Finding Labels'].str.contains("Pleural_Thickening") == True].index)
	data = data.drop(data[data['Finding Labels'].str.contains("Pneumonia") == True].index)
	data = data.drop(data[data['Finding Labels'].str.contains("Pneumothorax") == True].index)
	data = data.drop(data[data['Finding Labels'].str.contains("No Finding") == True].index)

	print("now dataset length")
	print(len(data))

	all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
	all_labels = [x for x in all_labels if len(x)>0]
	print(all_labels)
	num_classes = len(all_labels)
	print("num class")
	print(num_classes)
	in_channels = 3
	for c_label in all_labels:
		if len(c_label)>1: # leave out empty labels
			data[c_label] = data['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

	# since the dataset is very unbiased, we can resample it to be a more reasonable collection
	# weight is 0.04 + number of findings
	sample_weights = data['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
	sample_weights /= sample_weights.sum()
	print(len(data))
	# TODO: remember change back
	# data = data.sample(5000, weights=sample_weights) # 20000
	# data = data.sample(1000, weights=sample_weights)
	# print(data.apply(lambda x:print(x)))
	data['disease_vec'] = data.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0]).map(lambda x: np.array([x.argmax()]))

	with open("./real_data/train_val_list.txt") as f:
		train_data_filenames = f.read().splitlines() 
	with open("./real_data/test_list.txt") as f:
		test_data_filenames = f.read().splitlines()

	all_sample_size = 19000
	# all_sample_size = 1000
	num_class_sel = num_classes
	all_sample_weight = 0.2
	test_sample_size = round(all_sample_size * all_sample_weight)

	train_each_sample = round((all_sample_size - test_sample_size) / num_class_sel)
	test_each_sample = round(test_sample_size / num_class_sel)

	# ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
	# [ 1.5399,         5.9636,          6.1887,       13.1200,  1.7587,       9.7910,     8.7467,   82.0000,      0.6521,     2.8398,    0.0971,     2.0061,         5.9099,          23.4286,       3.9281]
	# 		+														+												+			  +			+			+
	# want 5 classes, each with 4000 labels
	data_no_temp = data[data['Finding Labels'] == "Effusion"]
 	
	data_no_temp_test = data_no_temp[data_no_temp['Image Index'].isin(test_data_filenames)]
	data_no_temp = data_no_temp[data_no_temp['Image Index'].isin(train_data_filenames)]
	train_df = data_no_temp
	test_df = data_no_temp_test

	data_no_temp = data[data['Finding Labels'] == "Infiltration"]
	data_no_temp_test = data_no_temp[data_no_temp['Image Index'].isin(test_data_filenames)]
	data_no_temp = data_no_temp[data_no_temp['Image Index'].isin(train_data_filenames)]
	train_df = pd.concat([train_df, data_no_temp.sample(2800)], ignore_index=True, sort=False)
	test_df = pd.concat([test_df, data_no_temp_test], ignore_index=True, sort=False)


	data_no_temp = data[data['Finding Labels'] == "Atelectasis"]
	data_no_temp_test = data_no_temp[data_no_temp['Image Index'].isin(test_data_filenames)]
	data_no_temp = data_no_temp[data_no_temp['Image Index'].isin(train_data_filenames)]
	train_df = pd.concat([train_df, data_no_temp], ignore_index=True, sort=False)
	test_df = pd.concat([test_df, data_no_temp_test], ignore_index=True, sort=False)

	# These two less than the sample size
	data_no_temp = data[data['Finding Labels'] == "Nodule"]
	data_no_temp_test = data_no_temp[data_no_temp['Image Index'].isin(test_data_filenames)]
	data_no_temp = data_no_temp[data_no_temp['Image Index'].isin(train_data_filenames)]
	train_df = pd.concat([train_df, data_no_temp], ignore_index=True, sort=False)
	test_df = pd.concat([test_df, data_no_temp_test], ignore_index=True, sort=False)

	data_no_temp = data[data['Finding Labels'] == "Mass"]
	data_no_temp_test = data_no_temp[data_no_temp['Image Index'].isin(test_data_filenames)]
	data_no_temp = data_no_temp[data_no_temp['Image Index'].isin(train_data_filenames)]
	train_df = pd.concat([train_df, data_no_temp], ignore_index=True, sort=False)
	test_df = pd.concat([test_df, data_no_temp_test], ignore_index=True, sort=False)

	data_no_temp = None
	data_no_temp_test = None
	data = None

	print(train_df)
	print(test_df)

	# data = data.drop(data[data['Finding Labels'].str.contains("Hernia") == True].index)
	# # data = data.drop(data[data['Finding Labels'].str.contains("Edema") == True].index)
	# # data = data.drop(data[data['Finding Labels'].str.contains("Fibrosis") == True].index)
	# data = data.drop(data[data['Finding Labels'].str.contains("Atelectasis") == True].index)
	# # data = data.drop(data[data['Finding Labels'].str.contains("Effusion") == True].index)
	# data = data.drop(data[data['Finding Labels'].str.contains("Infiltration") == True].index)
	# data = data.drop(data[data['Finding Labels'].str.contains("Pneumonia") == True].index)
	# data = data.drop(data[data['Finding Labels'].str.contains("Pneumothorax") == True].index)

	# train_df, test_df = train_test_split(data, 
	#                                    test_size = 0.18, # 0.18182 
	#                                    random_state = 2018,
	#                                    stratify = data['Finding Labels'].map(lambda x: x[:4]))

	# train_df = data[data['Image Index'].isin(train_data_filenames)]
	# sample_weights = train_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
	# sample_weights /= sample_weights.sum()
	# train_df = train_df.sample(all_sample_size - test_sample_size, weights=sample_weights)

	# test_df = data[data['Image Index'].isin(test_data_filenames)]
	# sample_weights = test_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
	# sample_weights /= sample_weights.sum()
	# test_df = test_df.sample(test_sample_size, weights=sample_weights)

	print('train', train_df.shape[0], 'test', test_df.shape[0])
	# save to csv -> not working properly
	# train_df.to_csv('train_df.csv', sep='\t', encoding='utf-8', header='true', columns=['Image Index', 'disease_vec', 'path'])
	# test_df.to_csv('test_df.csv', sep='\t', encoding='utf-8', header='true', columns=['Image Index', 'disease_vec', 'path'])
	train_df.to_csv('train_df.csv', columns=['Image Index', 'disease_vec', 'path'])
	test_df.to_csv('test_df.csv', columns=['Image Index', 'disease_vec', 'path'])

	pd.set_option("display.max_columns", None)
	print(train_df['disease_vec'])
	df_values = train_df[['disease_vec', 'path']].values
	labels = [x_value[0] for x_value in df_values]
	paths = [x_value[1] for x_value in df_values]

	print(labels[0:4])

	trainset_array = [(paths[i], pd.to_numeric(labels[i], downcast='float')) for i in range(len(labels))]

	transform_train = transforms.Compose(
		[
		# transforms.Resize(128),
		transforms.Resize(224),
		transforms.ToTensor(),
		])

	trainset = ImageDataset(trainset_array, transform=transform_train)

	# trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
	#                                           shuffle=False, num_workers=opt.workers)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
											shuffle=False, num_workers=opt.workers)

	mean = 0
	for i, data in enumerate(trainloader, 0):
		imgs, labels, index = data

		mean += torch.from_numpy(np.mean(np.asarray(imgs), axis=(2,3))).sum(0)
	mean = mean / len(trainset)
	#

	transform_train = transforms.Compose(
		[
		transforms.Resize(opt.imageSize),
		transforms.RandomCrop(opt.imageSize, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((mean[0], mean[1], mean[2]), (1.0, 1.0, 1.0))
			# transforms.Normalize((mean[0], mean[1], mean[2]), (0.5, 0.5, 0.5))
		])

	transform_test = transforms.Compose(
		[
		transforms.Resize(opt.imageSize),
		transforms.ToTensor(),
		transforms.Normalize((mean[0], mean[1], mean[2]), (1.0, 1.0, 1.0))
			# transforms.Normalize((mean[0], mean[1], mean[2]), (0.5, 0.5, 0.5))
		])

	logger.info(transform_train)
	logger.info(transform_test)

	trainset = ImageDataset(trainset_array, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
											shuffle=True, num_workers=opt.workers)

	print("Train Loader Finished")

	df_values = test_df[['disease_vec', 'path']].values
	labels = [x_value[0] for x_value in df_values]
	paths = [x_value[1] for x_value in df_values]

	# trainset = torch.tensor([[read_image(paths[i]), labels[i], i] for i in range(len(labels))])
	testset_array = [(paths[i], pd.to_numeric(labels[i], downcast='float')) for i in range(len(labels))]
	testset = ImageDataset(testset_array, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize,
											shuffle=False, num_workers=opt.workers)

	print("Test Loader Finished")

	if opt.model == 'resnet34':
		net = resnet.resnet34(in_channels=in_channels, num_classes=num_classes)
	else: logger.info('no model exists')

	labels_array = np.array(trainset.imgs)[:, 1]
	weight = torch.FloatTensor(num_classes).cuda().zero_() + 1.
	# weight_neg = torch.FloatTensor(num_classes).cuda().zero_() + 1.
	for i in range(num_classes):
		# Changed
		temp_sum = 0
		temp_sum_neg = 0
		for image_label in labels_array:
			if image_label[i].astype(int) == 1:
				temp_sum += 1
			# elif image_label[i].astype(int) == 0:
			# 	temp_sum_neg += 1
		weight[i] = temp_sum
		# weight_neg[i] = temp_sum_neg
		# weight[i] = (torch.from_numpy(np.array(trainset.imgs)[:,1].astype(int)) == i).sum()
	weight = 1 / (weight / weight.max())
	# weight = weight / num_classes
	# weight_neg = weight_neg / num_classes
	# weight = torch.FloatTensor(num_classes).cuda().zero_() + 1.
	# weight = len(labels_array) / (num_classes * weight)
	logger.info('Weight')
	logger.info(weight.cpu().numpy())
	weight = weight.cuda()

	print("weight")
	print(weight)
	print("weight old")
	print(weight / num_classes)

	# https://stackoverflow.com/questions/46218566/pytorch-equivalence-for-softmax-cross-entropy-with-logits/63658068#63658068
	# cross_entropy(logits,ys) == -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))
	# criterion_nll = nn.NLLLoss()
	# TODO: idk if it is correct -> no as required by the paper it is cross entropy
	# criterion  	  = nn.CrossEntropyLoss(weight=weight)
	criterion  	  = nn.CrossEntropyLoss(weight=weight)
	criterion_nll = nn.NLLLoss()
	criterion_nr  = nn.CrossEntropyLoss(reduce=False)

	net          .cuda()
	criterion    .cuda()
	criterion_nll.cuda()
	criterion_nr .cuda()
	net = nn.DataParallel(net)
	# criterion_nr .cuda()

	optimizer = optim.SGD(net.parameters(), 
		lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

	train_preds  	  = torch.zeros(len(trainset), num_classes) - 1.
	# num_hist = 10
	num_hist = 10
	train_preds_hist  = torch.zeros(len(trainset), num_hist, num_classes)
	pl_ratio = 0.; nl_ratio = 1.-pl_ratio
	train_losses      = torch.zeros(len(trainset)) - 1.

	if opt.load_dir:
		ckpt = torch.load(opt.load_dir+'/'+opt.load_pth)
		net 	     .load_state_dict(ckpt['state_dict'])
		optimizer    .load_state_dict(ckpt['optimizer'])
		train_preds_hist      = ckpt['train_preds_hist']
		pl_ratio  			  = ckpt['pl_ratio']
		nl_ratio  			  = ckpt['nl_ratio']
		epoch_resume  		  = ckpt['epoch']
		logger.info('loading network SUCCESSFUL')
	else:
		epoch_resume = 0
		logger.info('loading network FAILURE')
	###################################################################################################
	# Start training
	pred_loss_train = []
	pred_loss_test = []
	pred_loss_train_neg = []
	pred_loss_test_neg = []
	pred_acc_train = []
	pred_acc_test = []
	epoch_number = []
		
	best_test_acc = 0.0
	for epoch in range(epoch_resume, opt.max_epochs):
		train_loss = train_loss_neg = train_acc = 0.0
		pl = 0.; nl = 0.; 
		if epoch in opt.epoch_step:
			for param_group in optimizer.param_groups:
				param_group['lr'] *= 0.1
				opt.lr = param_group['lr']
				
		for i, data in enumerate(trainloader, 0):

			net.zero_grad()
			imgs, labels, index = data
			labels_neg = (labels.unsqueeze(-1).repeat(1, opt.ln_neg)
			+ torch.LongTensor(len(labels), opt.ln_neg).random_(1, num_classes)) % num_classes
			
			assert labels_neg.max() <= num_classes-1 
			assert labels_neg.min() >= 0
			assert (labels_neg != labels.unsqueeze(-1).repeat(1, opt.ln_neg)).sum() == len(labels)*opt.ln_neg

			imgs = Variable(imgs.cuda()); labels = Variable(labels.cuda()); 
			labels_neg = Variable(labels_neg.cuda())

			logits = net(imgs)

			##
			s_neg = torch.log( torch.clamp(1.-F.softmax(logits, -1), min=1e-5, max=1.) )
			s_neg *= weight[labels].unsqueeze(-1).expand(s_neg.size()).cuda()

			_, pred = torch.max(logits.data, -1)
			acc = float((pred==labels.data).sum()) 
			train_acc += acc

			train_loss     += imgs.size(0)*criterion    (logits, labels         ).data
			train_loss_neg += imgs.size(0)*criterion_nll(s_neg , labels_neg[:,0]).data
			train_losses[index] = criterion_nr(logits, labels).cpu().data
			##

			if epoch >= opt.switch_epoch:
				if epoch == opt.switch_epoch and i == 0: logger.info('Switch to SelNL') 
				labels_neg[ train_preds_hist.mean(1)[index, labels] < 1/float(num_classes) ] = -100
				labels = labels*0 - 100
			else:
				labels = labels*0 - 100

			loss     = criterion    (logits    					, labels      					      ) * float((labels    >=0).sum())
			loss_neg = criterion_nll(s_neg.repeat(opt.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg>=0).sum())

			( (loss+loss_neg) / (float((labels>=0).sum())+float((labels_neg[:,0]>=0).sum())) ).backward()
			optimizer.step()

			train_preds[index.cpu()] = F.softmax(logits, -1).cpu().data
			pl += float((labels         >=0).sum())
			nl += float((labels_neg[:,0]>=0).sum())

		train_loss     /= len(trainset)
		train_loss_neg /= len(trainset)
		train_acc      /= len(trainset)
		pl_ratio  		= pl / float(len(trainset))
		nl_ratio  		= nl / float(len(trainset))
		noise_ratio = 1. - pl_ratio

		noise = (np.array(trainset.imgs)[:,1].astype(int) != np.array(clean_labels)).sum()
		logger.info('[%6d/%6d] loss: %5f, loss_neg: %5f, acc: %5f, lr: %5f, noise: %d, pl: %5f, nl: %5f, noise_ratio: %5f' 
			%(epoch, opt.max_epochs, train_loss, train_loss_neg, train_acc, opt.lr, noise, pl_ratio, nl_ratio, noise_ratio))
		###############################################################################################
		if epoch == 0:
			for i in range(in_channels): imgs.data[:,i] += mean[i].cuda()
			img = vutils.make_grid(imgs.data)
			vutils.save_image(img, '%s/x.jpg'%(opt.save_dir))
			logger.info('%s/x.jpg saved'%(opt.save_dir))

		net.eval()
		test_loss = test_acc = 0.0
		with torch.no_grad():
			for i, data in enumerate(testloader, 0):
				imgs, labels = data
				imgs = Variable(imgs.cuda()); labels = Variable(labels.cuda())

				logits = net(imgs)
				loss = criterion(logits, labels)
				test_loss += imgs.size(0)*loss.data

				_, pred = torch.max(logits.data, -1)
				acc = float((pred==labels.data).sum())
				test_acc += acc

		test_loss /= len(testset)
		test_acc  /= len(testset)

		inds  	  = np.argsort(np.array(train_losses))[::-1]
		rnge  	  = int(len(trainset)*noise_ratio)
		inds_filt = inds[:rnge]
		recall    = float( len(np.intersect1d(inds_filt, inds_noisy)) ) / float(len(inds_noisy))
		precision = float( len(np.intersect1d(inds_filt, inds_noisy)) ) / float(rnge)
		###############################################################################################
		logger.info('\tTESTING...loss: %5f, acc: %5f, best_acc: %5f, recall: %5f, precision: %5f'
			%(test_loss, test_acc, best_test_acc, recall, precision))
		net.train()
		###############################################################################################
		assert train_preds[train_preds<0].nelement() == 0
		train_preds_hist[:, epoch%num_hist] = train_preds
		train_preds = train_preds*0 - 1.
		assert train_losses[train_losses<0].nelement() == 0
		train_losses = train_losses*0 - 1.
		###############################################################################################
		is_best = test_acc > best_test_acc
		best_test_acc = max(test_acc, best_test_acc)
		state = ({
			'epoch' 		  : epoch,
			'state_dict' 	  : net 	 .state_dict(),
			'optimizer' 	  : optimizer.state_dict(),
			'train_preds_hist': train_preds_hist,
			'pl_ratio' 		  : pl_ratio,
			'nl_ratio' 		  : nl_ratio,
			})
		logger.info('saving model...')
		fn = os.path.join(opt.save_dir, 'checkpoint.pth.tar')
		torch.save(state, fn)
		if epoch % 100 == 0 or epoch == opt.switch_epoch-1 or epoch == opt.max_epochs-1:
			fn = os.path.join(opt.save_dir, 'checkpoint_epoch%d.pth.tar'%(epoch))
			torch.save(state, fn)
		# if is_best: 
		# 	fn_best = os.path.join(opt.save_dir, 'model_best.pth.tar')
		# 	logger.info('saving best model...')
		# 	shutil.copyfile(fn, fn_best)

		if epoch % 10 == 0:
			logger.info('saving histogram...')
			plt.hist(train_preds_hist.mean(1)[torch.arange(len(trainset)), np.array(trainset.imgs)[:,1].astype(int)], bins=20, range=(0., 1.), edgecolor='black', color='g')
			plt.xlabel('probability'); plt.ylabel('number of data')
			plt.grid()
			plt.savefig(opt.save_dir+'/histogram_epoch%03d.jpg'%(epoch))
			plt.clf()

			logger.info('saving separated histogram...')
			plt.hist(train_preds_hist.mean(1)[torch.arange(len(trainset))[inds_clean], np.array(trainset.imgs)[:,1].astype(int)[inds_clean]], bins=20, range=(0., 1.), edgecolor='black', alpha=0.5, label='clean')
			plt.hist(train_preds_hist.mean(1)[torch.arange(len(trainset))[inds_noisy], np.array(trainset.imgs)[:,1].astype(int)[inds_noisy]], bins=20, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy')
			plt.xlabel('probability'); plt.ylabel('number of data')
			plt.grid()
			plt.savefig(opt.save_dir+'/histogram_sep_epoch%03d.jpg'%(epoch))
			plt.clf()

if __name__ == "__main__":
	import args
	import os
	import sys
	import logging
	import torch
	torch.set_printoptions(profile="full")
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.backends.cudnn as cudnn
	import torch.optim as optim
	import torch.utils.data
	from torchvision.io import read_image
	import torchvision.datasets as dset
	import torchvision.transforms as transforms
	import torchvision.utils as vutils
	from torch.autograd import Variable
	from datetime import datetime
	import shutil
	import numpy as np
	import math
	import matplotlib
	matplotlib.use('Agg') # For error: _tkinter.TclError: couldn't connect to display "localhost:10.0"
	import matplotlib.pyplot as plt
	import pickle
	sys.path.append('models')
	import resnet
	import noisy_folder

	import pandas as pd
	from glob import glob
	from itertools import chain
	from sklearn.model_selection import train_test_split
	from utils import ImageDataset, NLNLCrossEntropyLossNL, NLNLCrossEntropyLossPL
	main()