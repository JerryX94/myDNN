rm -f run.log
#bsub -J XJY_DNN -o run.log -q q_sw_share -n 1 -cgsp 64 -host_stack 1000 -share_size 6400 ./build/test_dnn
bsub -sw3runarg "-P -master" -J XJY_DNN -o run.log -q q_sw_share -n 1 -cgsp 64 -host_stack 1000 -share_size 6400 ./build/test_dnn
