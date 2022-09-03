echo "======DTB======" 'imb_ratio' 0.5
echo "--no--"
rm -rf ../data/TU/dtb/processed

python main.py --dataset='dtb' --imb_ratio=0.50 --setting='no' --num_training=1 --num_val=1 --epochs=100 --weight_decay=0.001

