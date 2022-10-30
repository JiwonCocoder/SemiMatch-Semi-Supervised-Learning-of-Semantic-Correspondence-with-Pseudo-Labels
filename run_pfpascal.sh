python train.py --benchmark pfpascal --batch_size 64 --c args.yaml --loss_sup semi --dynamic_unsup --lr 0.000001  
python train.py --benchmark pfpascal --batch_size 64 --c args.yaml --loss_sup ncnet --lr 0.000001
python train.py --benchmark pfpascal --batch_size 4 --c args.yaml --loss_sup semi --train_fe --dynamic_unsup 
python train.py --benchmark pfpascal --batch_size 4 --c args.yaml --loss_sup ncnet --train_fe

# python train.py