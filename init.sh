BERT_HOME=~/torch_app_data/bert

SA_DATA_DIR=~/torch_app_data/dataset/sa
if [ ! -d $SA_DATA_DIR ];then
  mkdir -p $SA_DATA_DIR
fi
cp total.csv $SA_DATA_DIR

SA_BEST=~/torch_app_data/model/sa_best/
if [ ! -d $SA_BEST ];then
  mkdir -p $SA_BEST
fi
cp *.ckpt $SA_BEST

if [ ! -d $BERT_HOME ];then
  mkdir -p $BERT_HOME
  cd $BERT_HOME

  wget https://huggingface.co/bert-base-chinese/resolve/main/tokenizer.json
  wget https://huggingface.co/bert-base-chinese/resolve/main/tokenizer_config.json
  wget https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt
  wget https://huggingface.co/bert-base-chinese/resolve/main/config.json
  wget https://huggingface.co/bert-base-chinese/resolve/main/pytorch_model.bin

fi