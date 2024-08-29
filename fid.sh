NAME="n2n_16_timed"
python -m pytorch_fid results/organized/$NAME/fake_B results/organized/$NAME/real_B
python -m pytorch_fid results/organized/$NAME/fake_A results/organized/$NAME/real_A