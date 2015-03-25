vw="/usr/local/bin/vw"
function run() {
    train=$1
    test=$2
    $vw  -c -k -b 25 --oaa 9 --passes 30 --loss_function logistic  -q fv $train -f ${train}.model.vw
    #$vw -t -i ${train}.model.vw --oaa 9 $test -p ${train}.pred -r ${train}.raw -a >xy
    $vw -t -i ${train}.model.vw --oaa 9 $test -p ${train}.pred -r ${train}.raw
}
run train.txt test.txt 
