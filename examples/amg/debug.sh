rm *txt
make 1_plate
./1_plate.out --nxe 4 --solver rn_amg --omegap 0.5 --threshold 0.1 --SR 1e3 >> out.txt
cd _py_demo/3_rn_amg/
python 1_plate.py --nxe 4 --noplot >> "../../out2.txt"
cd ../../