#!/bin/bash
# echo "scale=1;$a*0.8"|bc
# 循环10次
# echo "for politic"
# for ((i=2; i<=10; i++))
# do
#     # 计算a的值
#     fl=$(bc <<< "scale=1; $i/10")
#     echo $fl
#     nohup python main_1.py -alpha $fl > Res4/BO_politic_0$fl.txt
# done

echo "for politic"
for ((i=2; i<=10; i++))
do
    # 计算a的值
    fl=$(bc <<< "scale=1; $i/10")
    echo $fl
    nohup python main_1.py -alpha $fl > Res5/BO_gossip_0$fl.txt
done

# echo "for politic"
# for ((i=1; i<=5; i++))
# do
#     # 计算a的值
#     fl=$(bc <<< "scale=1; $i/10")
#     echo $fl
#     nohup python main_1.py -alpha 0.2 > Res4/1_BO_politic_0$fl.txt
# done
# echo "for gossip BO-two-infer"
# for ((i=1; i<=10; i++))
# do
#     # 计算a的值
#     fl=$(bc <<< "scale=1; $i/10")
#     echo "for gossip BO-two-infer"
#     echo $fl
#     nohup python main_1.py -alpha $fl -dn 'gossip' > Res4/BO_gossip_0$fl.txt
# done

# echo "for politic BO-two-atten"
# for ((i=1; i<=10; i++))
# do
#     # 计算a的值
#     fl=$(bc <<< "scale=1; $i/10")
#     echo "for gossip BO-two-atten"
#     echo $fl
#     nohup python main_atten.py -alpha $fl > Res4/BO_politic-atten_0$fl.txt
# done

# echo "for politic-x-noweight"
# for ((i=2; i<=10; i++))
# do
#     # 计算a的值
#     fl=$(bc <<< "scale=1; $i/10")
#     echo "for politic BO-politic-x-noweight"
#     echo $fl
#     nohup python main_xnowe.py -alpha $fl > Res4/BO_politic_x-noweight_0$fl.txt
# done

# echo "for gossip-x-noweight"
# for ((i=2; i<=10; i++))
# do
#     # 计算a的值
#     fl=$(bc <<< "scale=1; $i/10")
#     echo "for gossip BO-gossip-x-noweight"
#     echo $fl
#     nohup python main_xnowe.py -alpha $fl > Res4/BO_gossip_x-noweight_0$fl.txt
# done

# echo "for politic-one-infer-former"
# for ((i=1; i<=10; i++))
# do
#     # 计算a的值
#     fl=$(bc <<< "scale=1; $i/10")
#     echo "for politic-one-infer-former"
#     echo $fl
#     nohup python main_oneformer.py -fl $fl > Res4/BO_politic_one_infer_former_0$fl
# done

# echo "for politic-one-infer-later"
# for ((i=1; i<=10; i++))
# do
#     # 计算a的值
#     fl=$(bc <<< "scale=1; $i/10")
#     echo "for politic-one-infer-later"
#     echo $fl
#     nohup python main_onelater.py -fl $fl > Res4/BO_politic_one_infer_later_0$fl
# done

