task=MNLI
task=muti_tno_g

# RTE MRPC CoLA STS-B SST-2 QNLI MNLI QQP
for task in pathfinder; do
    echo $task
    sentence=`spring.submit task -a  | grep $task`

    array=($sentence)
    for i in "${array[@]}"; do
        case "$i" in [1-9][0-9]*)
            echo $i
            scancel $i
        esac
    done
done