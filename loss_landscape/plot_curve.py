import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--result_folder", "-r", required=True)
	args = parser.parse_args()


	event_acc = EventAccumulator(args.result_folder)
	event_acc.Reload()
	tmp_train_time, tmp_train_step , tmp_train_loss = zip(*event_acc.Scalars('train/loss'))
	_, test_step , test_loss = zip(*event_acc.Scalars('test/loss'))	
	_, acc_step , test_acc = zip(*event_acc.Scalars('test/acc'))


	#time = len([x for x in tmp_train_step if x<=test_step[70]])
	#print(time)
	#tmp_train_step = tmp_train_step[0:8175]
	#tmp_train_loss = tmp_train_loss[0:8175]

	#time = 70 
	#print(time)
	#test_step = test_step[0:70]
	#test_loss = test_loss[0:70]
	#acc_step = acc_step[0:70]
	#test_acc = test_acc[0:70]

	test_loss = list(test_loss)

	# plot the train/test losses in the same sacle 
	# plot the test acc in the other scale 
	fig, ax1 = plt.subplots()

	train_step = [tmp_train_step[i] for i in range(0,len(tmp_train_step),20)]
	train_loss = [tmp_train_loss[i] for i in range(0,len(tmp_train_loss),20)]

	total_train_steps = max(train_step) 
	train_step = [ 1.0/total_train_steps * x for x in train_step]

	total_test_steps = max(test_step) 
	test_step = [ 1.0/total_test_steps * x for x in test_step] 

	total_test_steps = max(acc_step) 
	acc_step = [ 1.0/total_test_steps * x for x in acc_step] 


	ax1.set_xlabel('Training Procedure in Percentage')
	ax1.xaxis.set_major_formatter(mtick.PercentFormatter())	
	ax1.set_xlim(0,1)

	color = 'tab:red'
	
	ax1.set_ylabel('loss', color=color)
	line1 = ax1.plot(train_step, train_loss, color='orange', label='train loss')
	line2 = ax1.plot(test_step, test_loss, color=color, label='test loss')
	ax1.tick_params(axis='y', labelcolor=color)

	ax1.set_ylim(0,max(train_loss+test_loss)*1.25)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
	line3 = ax2.plot(acc_step, test_acc, color=color, label='test accuracy')
	ax2.tick_params(axis='y', labelcolor=color)
	ax2.set_ylim(0,1.0)

	labs = [l.get_label() for l in line1+line2+line3]
	ax2.legend(line1+line2+line3, labs, loc = 'upper left')

	ax2.text(0.7, test_acc[-1],"Final Precision %.3lf%%"%(test_acc[-1]*100),ha="left",va="bottom")

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.savefig(args.result_folder + "/loss_acc_curve.png")
