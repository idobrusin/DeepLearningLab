for filename in ./config/*.cfg
do
	fname=$(basename "$filename" .cfg)
	python run_agent.py cfg "$fname" TEST
done
