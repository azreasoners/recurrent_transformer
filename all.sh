# Set the GPU index to run all experiments
gpu_idx=0
# Set the number of runs for each experiment
runs=1
# Set whether wandb is used for visualization
wandb=''
# wandb='--wandb'


echo -e "========== Textual Sudoku Experiments Start ==========" >> logs/textual_sudoku.txt
cd sudoku

echo -e "\n\n========== Textual Sudoku on SATNet 9k/1k dataset ==========\n\n" >> ../logs/textual_sudoku.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 ${wandb} --dataset satnet --gpu ${gpu_idx} --seed ${i}" >> ../logs/textual_sudoku.txt
    python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 ${wandb} --dataset satnet --gpu ${gpu_idx} --seed ${i} >> ../logs/textual_sudoku.txt
done

echo -e "\n\n========== Textual Sudoku on Palm 9k/1k dataset ==========\n\n" >> ../logs/textual_sudoku.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 ${wandb} --dataset palm --gpu ${gpu_idx} --seed ${i}" >> ../logs/textual_sudoku.txt
    python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 ${wandb} --dataset palm --gpu ${gpu_idx} --seed ${i} >> ../logs/textual_sudoku.txt
done

echo -e "\n\n========== Textual Sudoku on Palm 9k/1k dataset with L_attention ==========\n\n" >> ../logs/textual_sudoku.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 ${wandb} --dataset palm --gpu ${gpu_idx} --seed ${i} --loss att_c1 --hyper 0 1" >> ../logs/textual_sudoku.txt
    python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 ${wandb} --dataset palm --gpu ${gpu_idx} --seed ${i} --loss att_c1 --hyper 0 1 >> ../logs/textual_sudoku.txt
done

echo -e "\n\n========== Textual Sudoku on Palm 9k/1k dataset with L_sudoku ==========\n\n" >> ../logs/textual_sudoku.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 ${wandb} --dataset palm --gpu ${gpu_idx} --seed ${i} --loss c1 --hyper 1 0" >> ../logs/textual_sudoku.txt
    python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 ${wandb} --dataset palm --gpu ${gpu_idx} --seed ${i} --loss c1 --hyper 1 0 >> ../logs/textual_sudoku.txt
done

echo -e "\n\n========== Textual Sudoku on Palm 9k/1k dataset with L_attention+L_sudoku ==========\n\n" >> ../logs/textual_sudoku.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 ${wandb} --dataset palm --gpu ${gpu_idx} --seed ${i} --loss att_c1 c1 --hyper 0.5 0.5" >> ../logs/textual_sudoku.txt
    python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 ${wandb} --dataset palm --gpu ${gpu_idx} --seed ${i} --loss att_c1 c1 --hyper 0.5 0.5 >> ../logs/textual_sudoku.txt
done

cd ..
echo -e "\n\n========== Textual Sudoku Experiments End ==========" >> logs/textual_sudoku.txt




echo -e "========== Visual Sudoku Experiments Start ==========" >> logs/visual_sudoku.txt
cd visual_sudoku

echo -e "\n\n========== Visual Sudoku on SATNet-V 9k/1k dataset ==========\n\n" >> ../logs/visual_sudoku.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 500 --eval_interval 1 ${wandb} --dataset satnet --gpu ${gpu_idx} --seed ${i}" >> ../logs/visual_sudoku.txt
    python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 500 --eval_interval 1 ${wandb} --dataset satnet --gpu ${gpu_idx} --seed ${i} >> ../logs/visual_sudoku.txt
done

echo -e "\n\n========== Visual Sudoku on Palm-V 18k/1k dataset ==========\n\n" >> ../logs/visual_sudoku.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 500 --eval_interval 1 ${wandb} --dataset palm --n_train 18000 --gpu ${gpu_idx} --seed ${i}" >> ../logs/visual_sudoku.txt
    python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 500 --eval_interval 1 ${wandb} --dataset palm --n_train 18000 --gpu ${gpu_idx} --seed ${i} >> ../logs/visual_sudoku.txt
done

echo -e "\n\n========== Visual Sudoku on Palm-V 18k/1k dataset with L_attention ==========\n\n" >> ../logs/visual_sudoku.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 500 --eval_interval 1 ${wandb} --dataset palm --n_train 18000 --gpu ${gpu_idx} --seed ${i} --loss att_c1" >> ../logs/visual_sudoku.txt
    python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 500 --eval_interval 1 ${wandb} --dataset palm --n_train 18000 --gpu ${gpu_idx} --seed ${i} --loss att_c1 >> ../logs/visual_sudoku.txt
done

echo -e "\n\n========== Visual Sudoku on Palm-V 18k/1k dataset with L_sudoku ==========\n\n" >> ../logs/visual_sudoku.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 500 --eval_interval 1 ${wandb} --dataset palm --n_train 18000 --gpu ${gpu_idx} --seed ${i} --loss c1" >> ../logs/visual_sudoku.txt
    python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 500 --eval_interval 1 ${wandb} --dataset palm --n_train 18000 --gpu ${gpu_idx} --seed ${i} --loss c1 >> ../logs/visual_sudoku.txt
done

echo -e "\n\n========== Visual Sudoku on Palm-V 18k/1k dataset with L_attention+L_sudoku ==========\n\n" >> ../logs/visual_sudoku.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 500 --eval_interval 1 ${wandb} --dataset palm --n_train 18000 --gpu ${gpu_idx} --seed ${i} --loss att_c1 c1" >> ../logs/visual_sudoku.txt
    python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 500 --eval_interval 1 ${wandb} --dataset palm --n_train 18000 --gpu ${gpu_idx} --seed ${i} --loss att_c1 c1 >> ../logs/visual_sudoku.txt
done

cd ..
echo -e "\n\n========== Visual Sudoku Experiments End ==========" >> logs/visual_sudoku.txt




echo -e "========== 16x16 Textual Sudoku Experiments Start ==========" >> logs/16x16_textual_sudoku.txt
cd sudoku_16

echo -e "\n\n========== 16x16 Textual Sudoku on easy 9k/1k dataset ==========\n\n" >> ../logs/16x16_textual_sudoku.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --dataset easy ${wandb} --gpu ${gpu_idx} --seed ${i}" >> ../logs/16x16_textual_sudoku.txt
    python main.py --dataset easy ${wandb} --gpu ${gpu_idx} --seed ${i} >> ../logs/16x16_textual_sudoku.txt
done

echo -e "\n\n========== 16x16 Textual Sudoku on medium 9k/1k dataset ==========\n\n" >> ../logs/16x16_textual_sudoku.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --dataset medium ${wandb} --gpu ${gpu_idx} --seed ${i}" >> ../logs/16x16_textual_sudoku.txt
    python main.py --dataset medium ${wandb} --gpu ${gpu_idx} --seed ${i} >> ../logs/16x16_textual_sudoku.txt
done

cd ..
echo -e "\n\n========== 16x16 Textual Sudoku Experiments End ==========" >> logs/16x16_textual_sudoku.txt




echo -e "========== Nonogram Experiments Start ==========" >> logs/nonogram.txt
cd nonogram

echo -e "\n\n========== Nonogram on 7x7 board with 9k/1k dataset ==========\n\n" >> ../logs/nonogram.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --game_size 7 ${wandb} --gpu ${gpu_idx} --seed ${i}" >> ../logs/nonogram.txt
    python main.py --game_size 7 ${wandb} --gpu ${gpu_idx} --seed ${i} >> ../logs/nonogram.txt
done

echo -e "\n\n========== Nonogram on 15x15 board with 9k/1k dataset ==========\n\n" >> ../logs/nonogram.txt
for i in $(seq 0 $((runs-1))); do
    echo "python main.py --game_size 15 ${wandb} --gpu ${gpu_idx} --seed ${i}" >> ../logs/nonogram.txt
    python main.py --game_size 15 ${wandb} --gpu ${gpu_idx} --seed ${i} >> ../logs/nonogram.txt
done

cd ..
echo -e "\n\n========== Nonogram Experiments End ==========" >> logs/nonogram.txt
