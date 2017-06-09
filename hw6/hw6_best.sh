#!/bin/bash

user_path="$1/users.csv"
movie_path="$1/movies.csv"
test_path="$1/test.csv"

python3.5 hw6_best_test.py $user_path $movie_path $test_path $2