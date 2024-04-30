from __future__ import print_function
import torch

def target_to_oh(target):
	# NUM_CLASS = 10  # hard code here, can do partial
	# one_hot = [0] * NUM_CLASS
	# one_hot[target] = 1
	# return one_hot
	return torch.eye(10)[target]

def main():
	opt = args.args_original()

	print("GPU information")
	print(torch.cuda.is_available())
	print(torch.cuda.current_device())
	print(torch.cuda.device_count())
	print(torch.cuda.get_device_name(0))
	print()

	if opt.load_dir: 
		assert os.path.isdir(opt.load_dir)
		opt.save_dir = opt.load_dir
	else: 			  		   		
		opt.save_dir = '{}/{}_{}_{}_{}'.format(opt.save_dir, opt.dataset, opt.model, opt.noise_type, int(opt.noise*100))
	try:
		os.makedirs(opt.save_dir)
	except OSError:
		pass
	cudnn.benchmark = True

	logger = logging.getLogger("ydk_logger")
	fileHandler = logging.FileHandler(opt.save_dir+'/train.log')
	streamHandler = logging.StreamHandler()

	logger.addHandler(fileHandler)
	logger.addHandler(streamHandler)

	logger.setLevel(logging.INFO)
	logger.info(opt)
	###################################################################################################
	if   opt.dataset == 'cifar10_wo_val' : num_classes = 10; in_channels=3
	else: logger.info('There exists no data')

	##
	# Computing mean
	trainset = dset.ImageFolder(root='{}/{}/train'.format(opt.dataroot, opt.dataset), 
		transform=transforms.ToTensor())
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
											shuffle=False, num_workers=opt.workers)
	mean = 0
	for i, data in enumerate(trainloader, 0):
		imgs, labels = data
		mean += torch.from_numpy(np.mean(np.asarray(imgs), axis=(2,3))).sum(0)
	mean = mean / len(trainset)
	##

	transform_train = transforms.Compose(
		[
		transforms.Resize(opt.imageSize),
		transforms.RandomCrop(opt.imageSize, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((mean[0], mean[1], mean[2]), (1.0, 1.0, 1.0))
		])

	transform_test = transforms.Compose(
		[
		transforms.Resize(opt.imageSize),
		transforms.ToTensor(),
		transforms.Normalize((mean[0], mean[1], mean[2]), (1.0, 1.0, 1.0))
		])

	logger.info(transform_train)
	logger.info(transform_test)

	with open ('noise/%s/train_labels_n%02d_%s'%(opt.noise_type, opt.noise*000, opt.dataset), 'rb') as fp:
		clean_labels = pickle.load(fp)
	with open ('noise/%s/train_labels_n%02d_%s'%(opt.noise_type, opt.noise*100, opt.dataset), 'rb') as fp:
		noisy_labels = pickle.load(fp)
	logger.info(float(np.sum(clean_labels != noisy_labels)) / len(clean_labels))

	trainset = noisy_folder.ImageFolder(root='{}/{}/train'.format(opt.dataroot, opt.dataset), 
		noisy_labels=noisy_labels, transform = transform_train, target_transform=target_to_oh)
	testset = dset.ImageFolder(root='{}/{}/test'.format(opt.dataroot, opt.dataset), 
		transform=transform_test, target_transform=target_to_oh)

	clean_labels = list(clean_labels.astype(int))
	noisy_labels = list(noisy_labels.astype(int))

	inds_noisy = np.asarray([ind for ind in range(len(trainset)) if trainset.imgs[ind][-1] != clean_labels[ind]])
	inds_clean = np.delete(np.arange(len(trainset)), inds_noisy)
	print(len(inds_noisy))

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
											shuffle=True, num_workers=opt.workers)
	testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize,
											shuffle= False, num_workers=opt.workers)

	if opt.model == 'resnet34':
		net = resnet.resnet34(in_channels=in_channels, num_classes=num_classes)
	else: logger.info('no model exists')

	weight = torch.FloatTensor(num_classes).zero_() + 1.
	for i in range(num_classes):
		weight[i] = (torch.from_numpy(np.array(trainset.imgs)[:,1].astype(int)) == i).sum()
	weight = 1 / (weight / weight.max())

	# https://stackoverflow.com/questions/46218566/pytorch-equivalence-for-softmax-cross-entropy-with-logits/63658068#63658068
	# cross_entropy(logits,ys) == -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))
	# criterion_nll = nn.NLLLoss()
	# TODO: idk if it is correct -> no as required by the paper it is cross entropy
	# criterion  	  = nn.CrossEntropyLoss(weight=weight)
	criterion  	  = NLNLCrossEntropyLossPL(weight, num_classes)
	criterion_nll = NLNLCrossEntropyLossNL(weight, num_classes)
	criterion_nr  = nn.CrossEntropyLoss(reduce=False)
	# criterion  	  = nn.BCELoss(weight=weight)
	# criterion_nll = nn.BCELoss(weight=weight)
	# criterion_nr  = nn.BCELoss(reduce=False)

	net = net.cuda()
	criterion_nr .cuda()

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
		train_loss = train_loss_neg = train_acc = num_no_change = 0.0
		pl = 0.; nl = 0.; 
		if epoch in opt.epoch_step:
			for param_group in optimizer.param_groups:
				param_group['lr'] *= 0.1
				opt.lr = param_group['lr']

		logger.info('Begin epoch [%6d]' %(epoch))
		epoch_number.append(epoch)
				
		for i, data in enumerate(trainloader, 0):

			net.zero_grad()
			imgs, labels, index = data
			
			# Get complementary labels
			labels_neg = torch.full(labels.shape, 1) - labels
			indices_neg = labels_neg.multinomial(opt.ln_neg, replacement=False)
			labels_neg = torch.full(labels.shape, 0.0)
			for index_neg, item in enumerate(indices_neg):
				for item_index in item:
					if labels[index_neg, item_index] == 1.0:
						break
					labels_neg[index_neg, item_index] = 1.0

			imgs = Variable(imgs.cuda()); labels = Variable(labels.cuda()); 
			labels_neg = Variable(labels_neg.cuda())

			# print("label")
			# print(labels)
			# print("neg")
			# print(labels_neg)
			# print("index")
			# print(index)

			logits = net(imgs)
			### Change from here

			# pred = logits.data
			# logits_pred = F.softmax(logits, -1)
			_, pred = torch.max(logits.data, -1)
			pred = pred.unsqueeze(1)
			pred_acc = labels.gather(1, pred)
			acc = torch.sum(pred_acc==1.0)

			train_acc += acc

			train_loss     += imgs.size(0)*criterion.loss(logits, labels         ).data
			# only one label
			train_loss_neg += imgs.size(0)*criterion_nll.loss(logits , labels_neg).data

			if epoch >= opt.max_epochs_NL:
				# selpl
				if epoch == opt.max_epochs_NL and i == 0: logger.info('Switch to SelPL')
				## SelPL
				for items_index, labels_sub in enumerate(labels):
					this_index = index[items_index].cpu() # get the index of the image
					set_invalid = True

					# one logit label
					for index_labels, labels_value in enumerate(labels_sub):
						# now treat as valid if one of the labels is correct (>= the value)
						if labels_value == 1 or labels_value == 1.0:
							if train_preds_hist.mean(1)[this_index, index_labels] >= opt.cut:
								set_invalid = False
								break
						
					if set_invalid:
						labels[items_index] = -100
				labels_neg = labels_neg*0 - 100
			elif epoch >= opt.switch_epoch:
				if epoch == opt.switch_epoch and i == 0: logger.info('Switch to SelNL') 
				for items_index, labels_sub in enumerate(labels):
					this_index = index[items_index].cpu()
					set_invalid = True

					# one logit label
					for index_labels, labels_value in enumerate(labels_sub):
						# now treat as valid if one of the labels is correct (>= the value)
						if labels_value == 1 or labels_value == 1.0:
							# print("============================")
							# print(items_index)
							# print(train_preds_hist.mean(1)[this_index, index_labels])
							# print(train_preds_hist.mean(1)[this_index, index_labels] < 1/float(num_classes))
							if train_preds_hist.mean(1)[this_index, index_labels] >= 1/float(num_classes):
								set_invalid = False
								break
						
					if set_invalid:
						labels_neg[items_index] = -100
				labels = labels*0 - 100
			else:
				if epoch == 0 and i == 0: logger.info('Begin to NL')
				labels = labels*0 - 100
			
			loss     = criterion.loss(logits    					, labels      					      ) * float((labels>=0).sum())
			loss_neg = criterion_nll.loss(logits, labels_neg.contiguous()) * float((labels_neg>=0).sum())

			divider = (float((labels>=0).sum()) 
												+ float((labels_neg>=0).sum()))
			
			# print("-------------")
			# print(divider)
			
			if divider > 0:
				cal_loss = (loss+loss_neg) / (float((labels>=0).sum()) 
													+ float((labels_neg>=0).sum()))
				# cal_loss.backward()
				cal_loss.backward()
				# print(net.layer4[2].conv2.weight.grad[0, 0, 0])
				# print(net.layer4[2].conv2.weight[0,0,0])
				# print("cal_loss: " + str(cal_loss))
				optimizer.step()

				train_preds[index.cpu()] = F.softmax(logits, -1).cpu().data
				pl += float((labels         >=0).sum() / float(num_classes))
				# TODO: I changed this
				nl += float((labels_neg >= 0).sum()) / float(num_classes)
			else:
				num_no_change += 1

		train_loss     /= len(trainset)
		train_loss_neg /= len(trainset)
		# train_loss_real /= len(trainset)
		train_acc      /= len(trainset)
		pl_ratio  		= pl / float(len(trainset))
		nl_ratio  		= nl / float(len(trainset))
		noise_ratio = 1. - pl_ratio

		try:
			pred_acc_train.append(train_acc.cpu())
		except:
			pred_acc_train.append(train_acc)

		try:
			pred_loss_train.append(train_loss.cpu())
		except:
			pred_loss_train.append(train_loss)

		try:
			pred_loss_train_neg.append(train_loss_neg.cpu())
		except:
			pred_loss_train_neg.append(train_loss_neg)

		plt.plot(epoch_number, pred_acc_train, color="r")
		plt.savefig("Train_Accuracy.png")
		plt.clf()
		plt.plot(epoch_number, pred_loss_train, color="r")
		plt.savefig("Train_Loss.png")
		plt.clf()
		plt.plot(epoch_number, pred_loss_train_neg, color="r")
		plt.savefig("Train_Negative_Loss.png")
		plt.clf()

		logger.info('All labels are 0[%6d]' %(num_no_change))

		logger.info('[%6d/%6d] loss: %5f, loss_neg: %5f, loss_real(cal_loss): %5f, acc: %5f, lr: %5f, pl: %5f, nl: %5f, noise_ratio: %5f' 
			%(epoch, opt.max_epochs, train_loss, train_loss_neg, cal_loss, train_acc, opt.lr, pl_ratio, nl_ratio, noise_ratio))
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
				loss = criterion.loss(logits, labels)
				test_loss += imgs.size(0)*loss.data

				# logits_pred = F.softmax(logits, -1)
				_, pred = torch.max(logits.data, -1)
				pred = pred.unsqueeze(1)
				pred_acc = labels.gather(1, pred)
				acc = float(torch.sum(pred_acc==1.0))
				test_acc += acc

		test_loss /= len(testset)
		test_acc  /= len(testset)

		try:
			pred_acc_test.append(test_acc.cpu())
		except:
			pred_acc_test.append(test_acc)

		try:
			pred_loss_test.append(test_loss.cpu())
		except:
			pred_loss_test.append(test_loss)
		
		plt.plot(epoch_number, pred_acc_test, color="r")
		plt.savefig("Test_Accuracy.png")
		plt.clf()
		plt.plot(epoch_number, pred_loss_test, color="r")
		plt.savefig("Test_Loss.png")
		plt.clf()
		###############################################################################################
		logger.info('\tTESTING...loss: %5f, acc: %5f, best_acc: %5f'
			%(test_loss, test_acc, best_test_acc))
		net.train()
		###############################################################################################
		# assert train_preds[train_preds<0].nelement() == 0
		print(train_preds.shape)
		train_preds_hist[:, epoch%num_hist] = train_preds
		train_preds = train_preds*0 - 1.
		# assert train_losses[train_losses<0].nelement() == 0
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
		if epoch % 100 == 0 or epoch == opt.switch_epoch-1 or epoch == opt.max_epochs_NL-1 or epoch == opt.max_epochs_NL-1:
			fn = os.path.join(opt.save_dir, 'checkpoint_epoch%d.pth.tar'%(epoch))
			torch.save(state, fn)
		if is_best: 
			fn_best = os.path.join(opt.save_dir, 'model_best.pth.tar')
			logger.info('saving best model...')
			shutil.copyfile(fn, fn_best)

if __name__ == "__main__":
	import args
	import os
	import sys
	import logging
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