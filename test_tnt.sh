for i in $(seq 1 16)
do
  echo bs is $i
  python3 test_tnt.py --data_root /private/wangchen/instance_model/instance_model_data_tnt_test_latency --save_dir run_eval/tnt  --batch_size $i --with_cuda -rc /data/wangchen/TNT-Trajectory-Prediction/run/tnt/07-10-18-19_train_tnt/checkpoint_iter8.ckpt
  echo "-------------------------------"
  echo "-------------------------------"
  echo "-------------------------------"
done
