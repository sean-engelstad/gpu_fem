# rm *txt
# python 1_plate.py >> out.txt
# make 3_plate
# ./3_plate.out >> out3.txt

rm *txt
./4_plate_ilu.out --nxe 6 --subdomain 2 --thick 1e-2 --omega 1.0 --nsmooth 1 >> out.txt
./4_plate_ilu.out --nxe 6 --subdomain 2 --thick 1e-2 --omega 0.5 --nsmooth 2 >> out2.txt
