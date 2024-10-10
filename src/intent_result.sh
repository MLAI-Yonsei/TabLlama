#!/bin/bash

# Define the list of models
models=("rnn" "lstm" "vanillatf" "tabbert" "tabgpt2" "tabllama")


# Path to store the test results
output_file="./intent_results.txt"

# Clear the results file if it exists
> $output_file

# Loop through each model and run the test
for model in "${models[@]}"
do
    echo "Running test for model: $model" | tee -a $output_file

    # Choose the appropriate config file
        if [ "$model" == "tabllama" ]; then
            config_file="./configs/utils/default_intent.yaml"
        else
            config_file="./configs/utils/config_intent.yaml"
        fi
    
    # Modify the YAML config file with the current model
    sed -i "s/^model_type:.*/model_type: \"$model\"/" $config_file
    sed -i "s#^load_path:.*#load_path: \"./training_results/${model}_best_model.pth\"#" $config_file
    
    # Run the test
    python main.py --config $config_file >> $output_file 2>&1
    
    # Separate the results for each model in the output file
    echo -e "\n\n" >> $output_file
done

echo "All tests completed. Results saved to $output_file."
