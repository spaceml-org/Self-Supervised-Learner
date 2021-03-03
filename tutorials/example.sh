
gdown http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
unzip -qq UCMerced_LandUse.zip
wait

echo "downloaded merced data into current directory."
echo "Training SIMCLR Model"
python Self-Supervised-Learner/train.py --technique SIMCLR --DATA_PATH /content/UCMerced_LandUse/Images --model imagenet_resnet18  --epochs 10 --batch_size 64 --log_name ssl

echo "Stopped SSL training at epoch 10. Will resume now from checkpoint."
python Self-Supervised-Learner/train.py --technique SIMCLR --DATA_PATH /content/UCMerced_LandUse/Images --model ./models/SIMCLR_ssl.ckpt  --epochs 20 --batch_size 64 --log_name ssl2
wait

echo "Typically we want to do SIMCLR SSL for > 100 epochs. For this example, let's just move to finetuning now earlier."
echo "We'll pass in our SSL Model to the finetuner"
python Self-Supervised-Learner/train.py --technique CLASSIFIER --DATA_PATH /content/UCMerced_LandUse/Images --model ./models/SIMCLR_ssl2.ckpt  --epochs 10  --log_name ft

echo "Stopping finetuning classifier at epoch 10. Will resume now from checkpoint."
python Self-Supervised-Learner/train.py --technique CLASSIFIER --DATA_PATH /content/UCMerced_LandUse/Images --model ./models/CLASSIFIER_ft.ckpt  --epochs 20  --log_name ft2

echo "Great - to access your classifier, just: "
echo "from models import CLASSIFIER" 
echo "model = CLASSIFIER.load_from_checkpoint\('CLASSIFIER_ft2.ckpt'\)"
echo "You can use this model like any old pytorch model."





