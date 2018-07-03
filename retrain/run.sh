echo RATE=$1
echo STEPS=$2
echo NAME=$3
echo MODULE=$4

python retrain.py \
	--tfhub_module=$4 \
	--learning_rate=$1 \
	--how_many_training_steps=$2 \
	--result_dir=results/$3 \
	 2>&1 | tee results/$3/output.txt
