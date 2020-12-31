mkdir ./merced
cd merced
gdown http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
unzip -qq UCMerced_LandUse.zip
wait
cd ..

echo "downloaded merced data into current directory."
echo "Training SIMCLR Model"
python SpaceForceDataSearch/ssl_dali_distrib.py --DATA_PATH merced/UCMerced_LandUse/Images --encoder minicnn --batch_size 64 --num_workers 16 --epochs 10 --gpus 1 --log_name example_merced_SSL

echo "Stopped SSL training at epoch 10. Will resume now from checkpoint."
python SpaceForceDataSearch/ssl_dali_distrib.py --DATA_PATH merced/UCMerced_LandUse/Images --encoder models/SSL/SIMCLR_SSL_example_merced_SSL.ckpt --batch_size 64 --num_workers 16 --epochs 5 --gpus 1 --log_name example_merced_SSL_resumed
wait

echo "Typically we want to do SIMCLR SSL for > 400 epochs. For this example, let's just move to finetuning now earlier."
echo "We'll pass in our SSL Model to the finetuner"
python SpaceForceDataSearch/finetuner_dali_distrib.py --DATA_PATH merced/UCMerced_LandUse/Images --encoder models/SSL/SIMCLR_SSL_example_merced_SSL_resumed.ckpt --batch_size 64 --num_workers 16 --epochs 10 --gpus 1 --log_name example_merced_FT 

echo "Stopping finetuning at epoch 10. Will resume now from checkpoint."
python SpaceForceDataSearch/finetuner_dali_distrib.py --DATA_PATH merced/UCMerced_LandUse/Images --encoder models/FineTune/FineTune_example_merced_FT.ckpt --batch_size 64 --num_workers 16 --epochs 5 --gpus 1 --log_name example_merced_FT_resumed

echo "Great - to access your classifier, just: 
echo "from finetuner_dali_distrib import finetuner" 
echo "model = finetuner.load_from_checkpoint\('models/FineTune/FineTune_example_merced_FT_resumed.ckpt'\)"
echo "You can use this model like any old pytorch model."





