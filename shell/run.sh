vw="/usr/local/bin/vw"
function run() {
    train=$1
    test=$2
    $vw  -c -k -b 25 --oaa 9 --passes 100 --l2 0.00001 --learning_rate 0.05 --holdout_period 5   --loss_function logistic $train -f ${train}.model.vw --invert_hash read.model
    #$vw -t -i ${train}.model.vw --oaa 9 $test -p ${train}.pred -r ${train}.raw -a >xy
    $vw -t -i ${train}.model.vw --oaa 9 $test -p ${test}.pred -r ${test}.raw
}
run train.txt1 train.txt2
#run train.txt test.txt 
