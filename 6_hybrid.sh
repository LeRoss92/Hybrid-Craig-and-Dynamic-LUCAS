#!/bin/bash

source ~/.bashrc
micromamba activate DPL_JAX_copy

temps=('2015' '2018' '2015_2018')
microbial_decompositions=('linear' 'MM' 'RMM')
microbial_turnovers=('linear' 'density_dependent')
saturations=('no' 'Langmuir')
targets_combinations=('Cp,Cb,Cm' 'Cp,Cb' 'Cp,Cm' 'Cp')
folds=(0 1 2 3 4 5 6 7 8 9)

mkdir -p 6_hybrid_outputs

combinations=()
for md in "${microbial_decompositions[@]}"; do
    for mt in "${microbial_turnovers[@]}"; do
        for sat in "${saturations[@]}"; do
            for targets in "${targets_combinations[@]}"; do
                for fold in "${folds[@]}"; do
                    for temp in "${temps[@]}"; do
                        combinations+=("${temp}:${fold}:${md}:${mt}:${sat}:${targets}")
                    done
                done
            done
        done
    done
done

total=${#combinations[@]}
combo_index=0
finished=0
start_time=$(date +%s)
declare -a job_ids=()

while [ $combo_index -lt $total ] || [ ${#job_ids[@]} -gt 0 ]; do
    running=0
    pending=0
    new_job_ids=()
    for job_id in "${job_ids[@]}"; do
        state=$(squeue -h -j "$job_id" -o %T 2>/dev/null | head -n 1)
        if [ -n "$state" ]; then
            new_job_ids+=("$job_id")
            if [ "$state" == "RUNNING" ]; then
                ((running++))
            elif [ "$state" == "PENDING" ]; then
                ((pending++))
            fi
        else
            ((finished++))
        fi
    done
    job_ids=("${new_job_ids[@]}")
    
    tracked=${#job_ids[@]}
    remaining=$((total - combo_index))
    
    if [ $finished -gt 0 ]; then
        elapsed=$(($(date +%s) - start_time))
        if [ $elapsed -gt 0 ]; then
            if [ $remaining -gt 0 ]; then
                eta_seconds=$((remaining * elapsed / finished))
                eta_hours=$((eta_seconds / 3600))
                eta_mins=$(((eta_seconds % 3600) / 60))
                eta_str="${eta_hours}h ${eta_mins}m"
            else
                eta_str="0h 0m"
            fi
        else
            eta_str="calculating..."
        fi
    else
        eta_str="calculating..."
    fi
    
    printf "\rRunning: %d | Pending: %d | Tracked: %d | Finished: %d | Remaining: %d | ETA: %s" "$running" "$pending" "$tracked" "$finished" "$remaining" "$eta_str"
    
    while [ $combo_index -lt $total ] && [ $tracked -lt 24 ]; do
        combo="${combinations[$combo_index]}"
        job_output=$(sbatch 6_hybrid_single.sh "$combo" 2>&1)
        job_id=$(echo "$job_output" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*')
        if [ -n "$job_id" ]; then
            job_ids+=($job_id)
            tracked=${#job_ids[@]}
            ((combo_index++))
        else
            echo ""
            echo "sbatch failed for combo: $combo"
            echo "sbatch output: $job_output"
            sleep 10
            break
        fi
    done
    
    sleep 5
done

echo ""
total_elapsed=$(($(date +%s) - start_time))
total_hours=$((total_elapsed / 3600))
total_mins=$(((total_elapsed % 3600) / 60))
total_secs=$((total_elapsed % 60))
echo "Total time: ${total_hours}h ${total_mins}m ${total_secs}s"