

// loop for each grid

// use multi-GPU partition typically for assembler
// BUT: if on coarsest problem, vector uses multi-GPU partition
// and assembler will use single-GPU partition 
// and then copy into LU pattern of SingleGPUDirectLU class