

"""
AbdomenCT # train pairs : 380  | # test pairs: 0   | # val pairs: 45. | 200-epoch
"""


####. Abdomen. ####
cd demos/
python reg_sam_affine.py -o=results -d=data/processed/AbdomenCTCT --data_shape 192 160 256 --data_phase=val -e=SAME/affine -g=3 
python reg_sam_coarse.py -o=results -d=data/processed/AbdomenCTCT --data_shape 192 160 256 --data_phase=val -e=SAME/coarse -g=3 
python eval_NetInsO_abdomen.py


cd scripts/
python train.py -o=results -d=data/processed/AbdomenCTCT --data_shape 192 160 256 -e=SAME/deform/vm/unsupervised/svf --train_config=train_config/train_config_abdomen_wSAMLoss.py -g=0 --lr=5e-5 --epochs=200 --save_model_period=20

