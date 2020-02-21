src=$1
data_dir=$2
clst_nums=(`ls $data_dir`) 

for k in "${clst_nums[@]}"
do 
	echo $k
	dst=`basename "${src/sub1/$k}"`
	dst=${dst//-/_} # replace - by _
	echo $dst
	cp $src $dst
	sed -i "s/sub1/$k/g"  $dst
done
