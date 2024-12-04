echo "JuliaProp v0.1 starts..."

julia --project=. --threads $2 ./samples/$1.jl > ./samples/$1.log

# command:

# 1. Run it In Backend -->
# nohup ./doit.sh {example_name} {core_number} &

# 2. Run it In Shell -->
# ./doit.sh {example_name} {core_number}