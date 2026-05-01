rm *.txt || echo "can't remove *.txt files"
rm *.out || echo "can't remove *.out files"

export NXE=4
# export NXE=10

make 1_single_asw
echo "run 1_single_asw.out"
./1_single_asw.out --nxe ${NXE} >> out1.txt

make 2_multi_asw
echo "run 2_multi_asw.out"
./2_multi_asw.out ${NXE} >> out2.txt