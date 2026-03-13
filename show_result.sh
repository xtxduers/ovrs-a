output=$1
if [ -z $output ]
then
    echo "No output directory found! Run with "sh show_result.sh [OUTPUT_DIR]""
    exit 0
fi


echo "========== iSAID Results =========="
cat $output/eval_iSAID/log.txt | grep copypaste

echo "========== DLRSD Results =========="
cat $output/eval_DLRSD/log.txt | grep copypaste

echo "========== Potsdam Results =========="
cat $output/eval_Potsdam/log.txt | grep copypaste

echo "========== Vaihingen Results =========="
cat $output/eval_Vaihingen/log.txt | grep copypaste