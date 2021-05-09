# python pretrained_model_rename.py
# rm -r ./utils/__pycache__
# rm -r ./model/__pycache__
# python main.py --task AC --source Art --target Clipart 
# sleep 60

# rm -r ./utils/__pycache__
# rm -r ./model/__pycache__
# python main.py --task AP --source Art --target Product 
# sleep 60

rm -r ./utils/__pycache__
rm -r ./model/__pycache__
python train.py --task AR --source Art --target RealWorld --data_list_root /home/peter/Desktop/CV-Datasets/OfficeHome/list --data_root /home/peter/Desktop/CV-Datasets/OfficeHome/
sleep 60

# rm -r ./utils/__pycache__
# rm -r ./model/__pycache__
# python main.py --task CA --source Clipart --target Art 
# sleep 60

# rm -r ./utils/__pycache__
# rm -r ./model/__pycache__
# python main.py --task CP --source Clipart --target Product 
# sleep 60

# rm -r ./utils/__pycache__
# rm -r ./model/__pycache__
# python train.py --task CR --source Clipart --target RealWorld --data_list_root /home/peter/Desktop/CV-Datasets/OfficeHome/list --data_root /home/peter/Desktop/CV-Datasets/OfficeHome/ --epoch 120
# sleep 60
