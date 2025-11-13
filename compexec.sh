rm -rf build
mkdir build
cd build
cmake ..
make

# Unset all SIM_ environment variables to avoid interference
for v in $(env | grep '^SIM_' | cut -d= -f1); do
    unset "$v"
done


./gas_simulation -e
