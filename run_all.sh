#!/bin/bash -l

pairs=( "8,15" "10,20" "15,30" )
for p in "${pairs[@]}"; do
  # split on the comma:
  IFS=',' read -r x y <<< "$p"
  echo "x=$x, y=$y"
  # pass to your program:
  qsub run_multi_finite.sh $x $y
  qsub run_multi_rewardfree.sh $x $y
done

