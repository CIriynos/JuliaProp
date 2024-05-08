echo "JuliaProp v0.1 starts..."

julia --project=. --threads $2 ./example/$1.jl

# command:
# nohup ./run.sh {example_name} {core_number} > logging.txt &