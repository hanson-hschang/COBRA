#!/bin/bash

## Loop over all povray files and produce images
for i in {8..11} # {4..7} # {0..3} # 
do
	echo "Processing " $i "th file..."
	python collect_br2_data2.py $i
done


# #!/bin/bash

# # Check if the correct number of arguments is provided
# if [ "$#" -ne 2 ]; then
#     echo "Usage: $0 <integer1> <integer2>"
#     exit 1
# fi

# # Check if the provided arguments are integers
# re='^[0-9]+$'
# if ! [[ $1 =~ $re ]] || ! [[ $2 =~ $re ]]; then
#     echo "Error: Both arguments must be integers."
#     exit 1
# fi

# # Assign arguments to variables
# integer1="$1"
# integer2="$2"

# # Perform some operation with the integers (example: addition)
# result=$((integer1 + integer2))

# # Display the result
# echo "Sum of $integer1 and $integer2 is: $result"

# ## Loop over all povray files and produce images
# for i in {0..120} # 175
# do
# 	echo "Processing " $i "th file..."
# 	# python test.py $i
# 	# python main_smooth.py $i
# 	python run_simulation_stat1.py $i
# done