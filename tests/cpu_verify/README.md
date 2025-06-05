## Checking Kmat matches exactly from CPU to GPU
* rowp, cols, and perm don't match after AMD reordering
* was difference in the rowp, cols even before fillin, before AMD reordering
* found out the elem_conn used were different
    * since I go straight from the TACSMeshLoader to TACSAssembler skipping the TACSCreator, I don't get the same elem_conn and then rowp, cols in the comparison rn
    * basically there is a sorting of elem_conn in the TACSCreator, I will try and add this to my code in order to have a one-to-one comparison and eventually add the Kmat

