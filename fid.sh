NAME="forest_subtraction_1_ks"
python -m pytorch_fid results/organized/$NAME/fake_B results/organized/$NAME/real_B
python -m pytorch_fid results/organized/$NAME/fake_A results/organized/$NAME/real_A