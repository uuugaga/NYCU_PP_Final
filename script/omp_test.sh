#!/bin/bash


echo "--------------------------------------------------------------------------"
echo "|                             OpenMP test                                |"
echo "--------------------------------------------------------------------------"
echo "| Thread num || Total time | Eff. || DFT time | Eff. || IDFT time | Eff. |"
echo "--------------------------------------------------------------------------"

# Run the serial version
serial_output=$(./sequential)
serial_dft=$(echo "$serial_output" | grep -m 1 "DFT execution time:" | awk '{print $4}')
serial_idft=$(echo "$serial_output" | grep "IDFT execution time:" | awk '{print $4}')
serial_total=$(echo "$serial_output" | grep "Total execution time:" | awk '{print $4}')

printf "| %-10s || %-10s | %-4s || %-8s | %-4s || %-9s | %-4s |\n" "Serial" "$serial_total" "N/A" "$serial_dft" "N/A" "$serial_idft" "N/A"

for threads in 2 4 8 16; do
    # Run the OpenMP version with $threads
    omp_output=$(./omp $threads)
    omp_dft=$(echo "$omp_output" | grep -m 1 "DFT execution time" | awk '{print $4}' | tr -d '\n')
    omp_idft=$(echo "$omp_output" | grep "IDFT execution time" | awk '{print $4}' | tr -d '\n')
    omp_total=$(echo "$omp_output" | grep "Total execution time" | awk '{print $4}' | tr -d '\n')

    # Calculate efficiencies
    eff_total=$(echo "$serial_total $omp_total $threads" | awk '{printf "%.2f", $1 / $2 / $3}')
    eff_dft=$(echo "$serial_dft $omp_dft $threads" | awk '{printf "%.2f", $1 / $2 / $3}')
    eff_idft=$(echo "$serial_idft $omp_idft $threads" | awk '{printf "%.2f", $1 / $2 / $3}')
    
    # Print the results
    printf "| %-10s || %-10s | %-4s || %-8s | %-4s || %-9s | %-4s |\n" "$threads" "$omp_total" "$eff_total" "$omp_dft" "$eff_dft" "$omp_idft" "$eff_idft"

    # Calculate MD5 values of the images
    md5_parallel=$(md5sum ./img/watermarked_image.png | awk '{print $1}')
    md5_serial=$(md5sum ./img/watermarked_image_serial.png | awk '{print $1}')

    # Compare MD5 values and print the result
    if [ "$md5_parallel" != "$md5_serial" ]; then
        echo "The images are different."
    fi

done
echo "--------------------------------------------------------------------------"