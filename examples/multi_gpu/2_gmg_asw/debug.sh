rm *.txt || echo "can't remove *.txt files"

make 1_single_asw
./1_single_asw.out --nxe 4 >> out1.txt

make 2_multi_asw
./2_multi_asw.out 4 >> out2.txt