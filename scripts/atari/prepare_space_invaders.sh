if [ ! -f ./pretrained/a3c-space-invaders/model.80.tar ]
then
  echo "No model file, downloading..."
  wget https://github.com/greydanus/baby-a3c/raw/master/spaceinvaders-v4/model.80.tar -P ./pretrained/a3c-space-invaders/
else
  echo "Model is already present"
fi