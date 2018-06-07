echo RATE=$1
echo STEPS=$2
echo NAME=$3
echo MODULE=$4

python retrain.py \
	--image_dir=cap_augmented \
	--tfhub_module=$4 \
	--learning_rate=$1 \
	--validation_batch_size=1000 \
	--eval_step_interval=1000 \
	--how_many_training_steps=$2 \
	--summaries_dir=train_summaries/$3_$2 \
	--saved_model_dir=train_models/$3_$2 \
	--misclassified_image_dir=train_misclassified/$3_$2 \
	--print_misclassified_test_images \
	 2>&1 | tee train_outputs/$3_$2.txt
