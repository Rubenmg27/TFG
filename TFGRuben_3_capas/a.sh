#bin/bash

EPOCHS=$1
MODELO=$2

#python3 my_train_torch.py /data/hook/3_capas /data/train.csv $EPOCHS fc $MODELO #--threshole 0.60

python3 my_test_torch.py /data/hook/3_capas /data/train.csv $EPOCHS $MODELO --train_data 0 --my_models 1

#python3 filtred_data.py /data/hook/3_capas /data/train.csv $EPOCHS $MODELO --train_data 0 --my_models 1

# tensor([[3.8615e-02, 2.6129e-03, 7.2283e-04, 2.5589e-04, 7.3373e-05]], device='cuda:0')
