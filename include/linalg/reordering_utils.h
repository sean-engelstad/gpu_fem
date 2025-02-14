
// the following are some reordering utils

// q-ordering (writing it myself),
// https://arc.aiaa.org/doi/epdf/10.2514/6.2020-3022 how to apply ILU(k) from
// below RCM reorderings (reverrse cuthill-mckee)? here's how to implement
// q-ordering 1) reorder the CSR data structure with bandwidth minimizing step
// like RCM 2) compute the bandwidth of the new reordered structure 3) random
// reodering reorder every groups of rows with #rows=1/p * bandwidth where p =
// prune width
//      normal prune widths are 2,1,1/2,1/4 (lower prune width results in more
//      nnz but more stable convergence of iterative solvers like GMRES)

int TacsIntegerComparator(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

/*
  Sort an array and remove duplicate entries from the array. Negative
  values are removed from the array.

  This function is useful when trying to determine the number of
  unique design variable from a set of unsorted variable numbers.

  The algorithm proceeds by first sorting the array, then scaning from
  the beginning to the end of the array, copying values back only once
  duplicate entries are discovered.

  input:
  len:     the length of the array
  array:   the array of values to be sorted

  returns:
  the size of the unique list <= len
*/
int TacsUniqueSort(int len, int *array) {
    // Sort the array
    qsort(array, len, sizeof(int), TacsIntegerComparator);

    int i = 0; // The location from which to take the entires
    int j = 0; // The location to place the entries

    // Remove the negative entries
    while (i < len && array[i] < 0)
        i++;

    for (; i < len; i++, j++) {
        while ((i < len - 1) && (array[i] == array[i + 1])) {
            i++;
        }

        if (i != j) {
            array[j] = array[i];
        }
    }

    return j;
}

// cuthill-mckee and reverse cuthill-mckee are shown here

/*!
  Given an unsorted CSR data structure, with duplicate entries,
  sort/uniquify each array, and recompute the rowp values on the
  fly.

  For each row, sort the array and remove duplicates.  Copy the
  values if required and skip the diagonal.  Note that the copy may
  be overlapping so memcpy cannot be used.
*/
void TacsSortAndUniquifyCSR(int nvars, int *rowp, int *cols, int remove_diag) {
    // Uniquify each column of the array
    int old_start = 0;
    int new_start = 0;
    for (int i = 0; i < nvars; i++) {
        // sort cols[start:rowp[i]]
        int rsize = TacsUniqueSort(rowp[i + 1] - old_start, &cols[old_start]);

        if (remove_diag) {
            int end = old_start + rsize;
            for (int j = old_start, k = new_start; j < end; j++, k++) {
                if (cols[j] == i) {
                    rsize--;
                    k--;
                } else if (j != k) {
                    cols[k] = cols[j];
                }
            }
        } else if (old_start != new_start) {
            int end = old_start + rsize;
            for (int j = old_start, k = new_start; j < end; j++, k++) {
                cols[k] = cols[j];
            }
        }

        old_start = rowp[i + 1];
        rowp[i] = new_start;
        new_start += rsize;
    }

    rowp[nvars] = new_start;
}

/*
  Compute the RCM level sets andreordering of the graph given by the
  symmetric CSR data structure rowp/cols.

  rowp/cols represents the non-zero structure of the matrix to be
  re-ordered

  Here levset is a unique, 0 to nvars array containing the level
  sets
*/
static int TacsComputeRCMLevSetOrder(const int nvars, const int *rowp,
                                     const int *cols, int *rcm_vars,
                                     int *levset, int root) {
    int start = 0; // The start of the current level
    int end = 0;   // The end of the current level

    // Set all the new variable numbers to -1
    for (int k = 0; k < nvars; k++) {
        rcm_vars[k] = -1;
    }

    int var_num = 0;
    while (var_num < nvars) {
        // If the current level set is empty, find any un-ordered variables
        if (end - start == 0) {
            if (rcm_vars[root] >= 0) {
                // Find an appropriate root
                int i = 0;
                for (; i < nvars; i++) {
                    if (rcm_vars[i] < 0) {
                        root = i;
                        break;
                    }
                }
                if (i >= nvars) {
                    return var_num;
                }
            }

            levset[end] = root;
            rcm_vars[root] = var_num;
            var_num++;
            end++;
        }

        while (start < end) {
            int next = end;
            // Iterate over the nodes added to the previous level set
            for (int current = start; current < end; current++) {
                int node = levset[current];

                // Add all the nodes in the next level set
                for (int j = rowp[node]; j < rowp[node + 1]; j++) {
                    int next_node = cols[j];

                    if (rcm_vars[next_node] < 0) {
                        rcm_vars[next_node] = var_num;
                        levset[next] = next_node;
                        var_num++;
                        next++;
                    }
                }
            }

            start = end;
            end = next;
        }
    }

    // Go through and reverse the ordering of all the variables
    for (int i = 0; i < var_num; i++) {
        int node = levset[i];
        rcm_vars[node] = var_num - 1 - i;
    }

    return var_num;
}

/*!
  Perform Reverse Cuthill-McKee ordering.

  Input:
  ------
  nvars, rowp, cols == The CSR data structure containing the
  graph representation of the matrix

  root: The root node to perform the reordering from
  Returns:
  --------
  rcm_vars == The new variable ordering

  The number of variables ordered in this pass of the RCM reordering
*/
int TacsComputeRCMOrder(const int nvars, const int *rowp, const int *cols,
                        int *rcm_vars, int root, int n_rcm_iters) {
    if (n_rcm_iters < 1) {
        n_rcm_iters = 1;
    }

    int *levset = new int[nvars];
    int rvars = 0;
    for (int k = 0; k < n_rcm_iters; k++) {
        rvars = TacsComputeRCMLevSetOrder(nvars, rowp, cols, rcm_vars, levset,
                                          root);
        if (nvars != rvars) {
            return rvars;
        }
        root = rcm_vars[0];
    }

    delete[] levset;
    return rvars;
}