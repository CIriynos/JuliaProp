echo "JuliaProp v0.1 starts..."

julia --project=. --threads 64 ./example/$1.jl

# command:
# nohup ./run.sh {example_name} > logging.txt &