#!/bin/bash

INPUT_DIR=testcases

for each in `seq 1 50`; do
	if [[ -f $INPUT_DIR/input$each.txt ]]; then
		cp $INPUT_DIR/input$each.txt ./input.txt
		time python2.7 hw3cs561s2017.py
		diff -w ./output.txt $INPUT_DIR/output$each.txt
		exit_code=$?
		if (($exit_code == 0)); then
		    echo Testcase $each passed.
		else
		    echo Testcase $each is not correct.
		fi
	fi
done
