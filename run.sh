# a bash script to evaluate the hyperparameter beta
echo "for politic"
for ((i=0; i<=10; i++))
do
    fl=$(bc <<< "scale=1; $i/10")
    echo $fl
    nohup env CUDA_LAUNCH_BLOCKING=1 python main_1.py -beta $fl -dn "politic" > bash_$fl.txt
done
