for x in `seq 1 100`; do
    cp test_image.jpg $(printf "test_image-%03d.jpg" $x)
done
