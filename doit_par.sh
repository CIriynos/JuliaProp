echo "JuliaProp v0.1 starts..."

example_name=$1
core_number=$2
tasks_num=$3
core_per_task=`expr $core_number / $tasks_num`
echo "core_per_task = $core_per_task"

id=1
while(( $id <= $tasks_num))
do
    echo "Task starts. task id = "$id"."
    nohup julia --project=. --threads $core_per_task ./samples/${example_name}.jl $id > ./samples/${example_name}_${id}.log &
    let "id++"
done

# command:

# Run it In Backend -->
# ./doit_par.sh {example_name} {core_number} {tasks_num}