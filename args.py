import argparse

def args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='cifar10_wo_val', help='what kind of dataset') 
	parser.add_argument('--dataroot', default='data', help='path to dataset')
	parser.add_argument('--realdataroot', default='real_data', help='path to dataset')
	parser.add_argument('--model', type=str, default='resnet34', help='type of model')
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=3)
	parser.add_argument('--batchSize', type=int, default=16, help='input batch size') # 32
	parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
	# parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
	# parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
	# parser.add_argument('--max_epochs', type=int, default=300, help='number of iterations to train for')
	# parser.add_argument('--max_epochs_NL', type=int, default=200, help='number of iterations to train for') 
	# parser.add_argument('--switch_epoch', type=int, default=100, help='epoch where training method changes')
	# parser.add_argument('--max_epochs', type=int, default=480, help='number of iterations to train for') # looks like x 2??
	# parser.add_argument('--max_epochs_NL', type=int, default=240, help='number of iterations to train for') 
	# parser.add_argument('--switch_epoch', type=int, default=120, help='epoch where training method changes')

	# parser.add_argument('--max_epochs', type=int, default=240, help='number of iterations to train for') # looks like x 2??
	# parser.add_argument('--max_epochs_NL', type=int, default=120, help='number of iterations to train for') 
	# parser.add_argument('--switch_epoch', type=int, default=60, help='epoch where training method changes')
	parser.add_argument('--max_epochs', type=int, default=88, help='number of iterations to train for') # looks like x 2??
	# parser.add_argument('--max_epochs_NL', type=int, default=44, help='number of iterations to train for') 
	# parser.add_argument('--switch_epoch', type=int, default=22, help='epoch where training method changes')
	parser.add_argument('--max_epochs_NL', type=int, default=0, help='number of iterations to train for') 
	parser.add_argument('--switch_epoch', type=int, default=0, help='epoch where training method changes')
 
	# parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
	# parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
	parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
	parser.add_argument('--epoch_step', type=int, nargs='+', default=[-1, -1], help='Learning Rate Decay Steps') 
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
	parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
	parser.add_argument('--save_dir', type=str, default='logs', help='Directory name to save the checkpoints')
	# parser.add_argument('--load_dir', type=str, default='./logs/NLNL', help='Directory name to load checkpoints')
	# parser.add_argument('--load_pth', type=str, default='checkpoint_epoch219.pth.tar', help='pth name to load checkpoints')
	parser.add_argument('--load_dir', type=str, default='', help='Directory name to load checkpoints')
	parser.add_argument('--load_pth', type=str, default='', help='pth name to load checkpoints')
	parser.add_argument('--pretrained', type=str, default='', help='Directory/pth name to load pretrained checkpoints')
	parser.add_argument('--noise', type=float, default=0.0, help='Noise ratio')
	parser.add_argument('--noise_type', type=str, default='val_split_symm_exc', help='Noise Type')
	# parser.add_argument('--ln_neg', type=int, default=1, help='number of negative labels on single image for training (ex. 110 for cifar100)') 
	parser.add_argument('--ln_neg', type=int, default=1, help='number of negative labels on single image for training (ex. 110 for cifar100)') 
	parser.add_argument('--cut', type=float, default=0.5, help='threshold value')
	opt = parser.parse_args() 
	
	print(opt)

	return opt

def args_test():

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='cifar10_wo_val', help='what kind of dataset') 
	parser.add_argument('--dataroot', default='data', help='path to dataset')
	parser.add_argument('--realdataroot', default='real_data', help='path to dataset')
	parser.add_argument('--model', type=str, default='resnet34', help='type of model')
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=3)
	parser.add_argument('--batchSize', type=int, default=8, help='input batch size') # 32
	parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
	# parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
	# parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
	# parser.add_argument('--max_epochs', type=int, default=300, help='number of iterations to train for')
	# parser.add_argument('--max_epochs_NL', type=int, default=200, help='number of iterations to train for') 
	# parser.add_argument('--switch_epoch', type=int, default=100, help='epoch where training method changes')
	# parser.add_argument('--max_epochs', type=int, default=480, help='number of iterations to train for') # looks like x 2??
	# parser.add_argument('--max_epochs_NL', type=int, default=240, help='number of iterations to train for') 
	# parser.add_argument('--switch_epoch', type=int, default=120, help='epoch where training method changes')
	parser.add_argument('--max_epochs', type=int, default=600, help='number of iterations to train for') # looks like x 2??
	parser.add_argument('--max_epochs_NL', type=int, default=300, help='number of iterations to train for') 
	parser.add_argument('--switch_epoch', type=int, default=150, help='epoch where training method changes')
	# parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
	# parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
	parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
	parser.add_argument('--epoch_step', type=int, nargs='+', default=[-1, -1], help='Learning Rate Decay Steps') 
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
	parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
	parser.add_argument('--save_dir', type=str, default='logs', help='Directory name to save the checkpoints')
	# parser.add_argument('--load_dir', type=str, default='./logs/NLNL', help='Directory name to load checkpoints')
	# parser.add_argument('--load_pth', type=str, default='checkpoint_epoch219.pth.tar', help='pth name to load checkpoints')
	parser.add_argument('--load_dir', type=str, default='', help='Directory name to load checkpoints')
	parser.add_argument('--load_pth', type=str, default='', help='pth name to load checkpoints')
	parser.add_argument('--pretrained', type=str, default='', help='Directory/pth name to load pretrained checkpoints')
	parser.add_argument('--noise', type=float, default=0.0, help='Noise ratio')
	parser.add_argument('--noise_type', type=str, default='val_split_symm_exc', help='Noise Type')
	# parser.add_argument('--ln_neg', type=int, default=1, help='number of negative labels on single image for training (ex. 110 for cifar100)') 
	parser.add_argument('--ln_neg', type=int, default=1, help='number of negative labels on single image for training (ex. 110 for cifar100)') 
	parser.add_argument('--cut', type=float, default=0.5, help='threshold value')
	opt = parser.parse_args() 
	
	print(opt)

	return opt

def args_original():

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='cifar10_wo_val', help='what kind of dataset') 
	parser.add_argument('--dataroot', default='data', help='path to dataset')
	parser.add_argument('--model', type=str, default='resnet34', help='type of model')
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
	parser.add_argument('--batchSize', type=int, default=128, help='input batch size') 
	parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
	parser.add_argument('--max_epochs', type=int, default=600, help='number of iterations to train for')
	parser.add_argument('--max_epochs_NL', type=int, default=300, help='number of iterations to train for') 
	parser.add_argument('--switch_epoch', type=int, default=150, help='epoch where training method changes')
	# parser.add_argument('--max_epochs', type=int, default=330, help='number of iterations to train for')
	# parser.add_argument('--max_epochs_NL', type=int, default=220, help='number of iterations to train for') 
	# parser.add_argument('--switch_epoch', type=int, default=110, help='epoch where training method changes')
	parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
	parser.add_argument('--epoch_step', type=int, nargs='+', default=[-1, -1], help='Learning Rate Decay Steps') 
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
	parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
	parser.add_argument('--save_dir', type=str, default='logs', help='Directory name to save the checkpoints')
	parser.add_argument('--load_dir', type=str, default='', help='Directory name to load checkpoints')
	parser.add_argument('--load_pth', type=str, default='', help='pth name to load checkpoints')
	parser.add_argument('--pretrained', type=str, default='', help='Directory/pth name to load pretrained checkpoints')
	parser.add_argument('--noise', type=float, default=0.0, help='Noise ratio')
	parser.add_argument('--noise_type', type=str, default='val_split_symm_exc', help='Noise Type')
	parser.add_argument('--ln_neg', type=int, default=1, help='number of negative labels on single image for training (ex. 110 for cifar100)') 
	parser.add_argument('--cut', type=float, default=0.5, help='threshold value')
	opt = parser.parse_args() 
	
	print(opt)

	return opt