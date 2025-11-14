rm -rf build
mkdir build
cd build
cmake ..
make

# Unset all SIM_ environment variables to avoid interference
for v in $(env | grep '^SIM_' | cut -d= -f1); do
    unset "$v"
done


sys=2
while ((sys<=1024)); do
    export SIM_NUM_SYSTEMS=$sys
    p=8
    while (( p <= 4224 )); do
        export SIM_PARTICLES_PER_SYSTEM=$p
        echo "Running: systems=$SIM_NUM_SYSTEMS particles=$SIM_PARTICLES_PER_SYSTEM"
        ./gas_simulation -e
        ((p *= 2))
    done
    ((sys *= 2))
done
