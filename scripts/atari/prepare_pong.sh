if [ ! -f ./pretrained/a3c-pong/model.80.tar ]
then
  echo "No model file, downloading..."
  wget https://github.com/greydanus/baby-a3c/raw/master/pong-v4/model.80.tar -P ./pretrained/a3c-pong/
else
  echo "Model is already present"
fi
