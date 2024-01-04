#!/bin/bash


echo "--------------------------------------------------------------------------------------"
echo "|                                    CUDA test                                       |"
echo "--------------------------------------------------------------------------------------"   
echo "| Thread num || Total time | SpeedUp. || DFT time | SpeedUp. || IDFT time | SpeedUp. |"
echo "--------------------------------------------------------------------------------------"

# Run the serial version
serial_output=$(./sequential)
serial_dft=$(echo "$serial_output" | grep -m 1 "DFT execution time:" | awk '{print $4}')
serial_idft=$(echo "$serial_output" | grep "IDFT execution time:" | awk '{print $4}')
serial_total=$(echo "$serial_output" | grep "Total execution time:" | awk '{print $4}')

printf "| %-10s || %-10s | %-8s || %-8s | %-8s || %-9s | %-8s |\n" "Serial" "$serial_total" "N/A" "$serial_dft" "N/A" "$serial_idft" "N/A"


# Run the OpenMP version with $threads
omp_output=$(./cuda)
omp_dft=$(echo "$omp_output" | grep -m 1 "DFT execution time" | awk '{print $4}' | tr -d '\n')
omp_idft=$(echo "$omp_output" | grep "IDFT execution time" | awk '{print $4}' | tr -d '\n')
omp_total=$(echo "$omp_output" | grep "Total execution time" | awk '{print $4}' | tr -d '\n')

# Calculate speedupiciencies
speedup_total=$(echo "$serial_total $omp_total" | awk '{printf "%.2f", $1 / $2}')
speedup_dft=$(echo "$serial_dft $omp_dft" | awk '{printf "%.2f", $1 / $2}')
speedup_idft=$(echo "$serial_idft $omp_idft" | awk '{printf "%.2f", $1 / $2}')

# Print the results
printf "| %-10s || %-10s | %-8s || %-8s | %-8s || %-9s | %-8s |\n" "CUDA" "$omp_total" "$speedup_total" "$omp_dft" "$speedup_dft" "$omp_idft" "$speedup_idft"

# Calculate MD5 values of the images
md5_parallel=$(md5sum ./img/watermarked_image.png | awk '{print $1}')
md5_serial=$(md5sum ./img/watermarked_image_serial.png | awk '{print $1}')

# Compare MD5 values and print the result
if [ "$md5_parallel" != "$md5_serial" ]; then
    echo "The images are different."
fi

echo "--------------------------------------------------------------------------------------"