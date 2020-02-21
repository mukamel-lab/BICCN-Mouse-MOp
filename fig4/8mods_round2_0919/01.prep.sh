src=$1
num=$2 


for ((k=2; k<=$num; k++))
do 
	echo $k
	dst=${src/"sub1"/"sub"$k}
	echo $dst
	cp $src $dst
	sed -i "s/sub1/sub$k/g"  $dst
done
