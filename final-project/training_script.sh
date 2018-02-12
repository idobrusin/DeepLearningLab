#python run_agent.py algo_1_model_1 TRAIN 1 -s 100000
#python run_agent.py algo_1_model_2 TRAIN 1 -s 100000
#python run_agent.py algo_1_model_3 TRAIN 1 -s 100000
#python run_agent.py algo_1_hl_8_model_1 TRAIN 1 -s 100000 -l 8
#python run_agent.py algo_1_hl_8_model_2 TRAIN 1 -s 100000 -l 8
#python run_agent.py algo_2_model_1 TRAIN 2 -s 100000
#python run_agent.py algo_2_model_2 TRAIN 2 -s 100000
#python run_agent.py algo_2_model_3 TRAIN 2 -s 100000
#python run_agent.py algo_2_hl_8_model_1 TRAIN 2 -s 100000 -l 8
#python run_agent.py algo_2_hl_8_model_2 TRAIN 2 -s 100000 -l 8
#python run_agent.py algo_3_model_1 TRAIN 3 -s 100000
#python run_agent.py algo_3_model_2 TRAIN 3 -s 100000
#python run_agent.py algo_3_model_3 TRAIN 3 -s 100000
#python run_agent.py algo_3_hl_8_model_1 TRAIN 3 -s 100000 -l 8
#python run_agent.py algo_3_hl_8_model_2 TRAIN 3 -s 100000 -l 8
#python run_agent.py algo_4_model_1 TRAIN 3 -s 100000
#python run_agent.py algo_4_model_2 TRAIN 3 -s 100000
#python run_agent.py algo_4_model_3 TRAIN 3 -s 100000
#python run_agent.py algo_4_hl_8_model_1 TRAIN 3 -s 100000 -l 8
#python run_agent.py algo_4_hl_8_model_2 TRAIN 3 -s 100000 -l 8

for filename in ./config/*.cfg
do
	fname=$(basename "$filename" .cfg)
	python run_agent.py cfg "$fname" TRAIN
done
