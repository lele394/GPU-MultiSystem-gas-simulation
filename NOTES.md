

Initialization will actually happen on HOST => Makes loading specific starting conditions easier.


We don't care about system ordering in the big matrix at initialization since the distributions for every system is the same. Might need to modify that later for multi-system with different distribution for each of them.




OKKKKKKKKK SOOOOOO
I can do ping ponging directly into L1 cache (great but) I need to be careful to the number of particles per systems
=> dependent on cache size
    => RTX4060 => 1536 particles, pad to 1500 to leave some room around, we dealing with 128kb here

