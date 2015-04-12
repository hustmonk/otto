vw="/usr/local/bin/vw"
function run() {
    train=$1
    test=$2
    $vw  -c -k -b 25 --oaa 9  -q fv --loss_function logistic $train -f ${train}.model.vw
    #$vw -t -i ${train}.model.vw --oaa 9 $test -p ${train}.pred -r ${train}.raw -a >xy
    $vw -t -i ${train}.model.vw --oaa 9 $test -p ${train}.pred -r ${train}.raw
}
run train.txt1 train.txt2
run train.txt test.txt 
