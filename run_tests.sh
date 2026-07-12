# Define your base configurations
base_exp="exp/fql/Debug/sd000_s_11165720.0.20260705_163909/"
strategies=("random" "quality" "state_region")

# Loop through each strategy
for strategy in "${strategies[@]}"; do
    # Loop through percentages from 10 to 90
    for i in {10..90..10}; do
        
        # Convert percentage to decimal format (e.g., 0.10)
        frac=$(echo "scale=2; $i/100" | bc)
        
        # Define output directory based on current parameters
        save_dir="analysis_humanoidmaze_medium_navigate/${strategy}_${i}pct"
        
        echo "Running: strategy=$strategy, removal_frac=$frac, save_dir=$save_dir"
        
        # Execute the command
        python run_subset_analysis.py \
            --exp_dir "$base_exp" \
            --strategy "$strategy" \
            --removal_frac "$frac" \
            --save_dir "$save_dir"
            
    done
done