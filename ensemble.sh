#!/bin/bash

set -e

#Number of folds
k_fold=1

#The detector you want to use
detector=res18

#How many epochs you want to train the model
epochs=50

#How frequent do you want to save the epoch
save_freq=5

#What is the batch size
batch_size=64

#Initial learning rate, decay 0.1 every 50% and 80% of epochs
lr=0.01

#Where to start the fold, used for debugging or error occurred
start_fold=0

#Store the ubmission.csv file
submission="submission"


for((i=${start_fold};i<${k_fold};i++))
do
	#Train the detector
	cmd="python main.py --optim sgd --model $detector --save-dir $i --train-filename filenames_train.txt --val-filename filenames_val.txt --lr $lr --workers 16 -b $batch_size --epochs $epochs --save-freq 5"
	echo "$cmd"
	echo "Training $i folder ..."
	$cmd
	# while [ 1 ]
	# do
	# 	$cmd
	# 	errono=$?
	# 	if [ $errono -eq 0 ]
	# 	then
	# 		exit $errono
	# 	fi
	# done
	
	best_epoch=`printf "%03d" $epochs`
	echo "Best epoch: $best_epoch"

	#Test on the test dataset
	cmd="python main.py --model $detector --resume results/${i}/${best_epoch}.ckpt --workers 1 --test-filename ./filenames_val.txt --test 1 --save-dir test/${i} --n_test 8"
	echo "$cmd"
        $cmd
	errono=$?
	if [ $errono -ne 0 ]
        then
                echo "Error occurred when finding best epoch of ${i}th fold"
                exit $errono
        fi
	
done

