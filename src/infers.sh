#!/bin/bash

# Define the list of models
models=("rnn" "lstm"  "vanillatf" "tabbert" "tabgpt2" "tabllama")

# Path to store the comparison results
result_file="inference_results.txt"

# Clear the results file if it exists
> $result_file

# Loop through each init_seq and model, modifying the config and running inference
for init_seq in {1..4}; do
    for model_type in "${models[@]}"; do
        echo "Running test for model: $model_type, init_seq: $init_seq" | tee -a $result_file
        
        # Choose the appropriate config file
        if [ "$model_type" == "tabllama" ]; then
            config_file="./configs/utils/default_infer.yaml"
        else
            config_file="./configs/utils/config_infer.yaml"
        fi
        
        # Modify the YAML config file with the current model and init_seq
        sed -i "s/^model_type:.*/model_type: \"$model_type\"/" $config_file
        sed -i "s#^load_path:.*#load_path: \"./training_results/${model_type}_best_model.pth\"#" $config_file
        sed -i "s/^init_seq:.*/init_seq: $init_seq/" $config_file
        
        # Run the inference with the modified config
        python main.py --config $config_file >> $result_file 2>&1
        
        # Separate results with a divider
        echo -e "\n----------------------------------------\n" >> $result_file
    done
done

echo "Script execution completed. Results saved in $result_file."