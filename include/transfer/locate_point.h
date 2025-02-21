#ifndef LOCATE_POINT_H
#define LOCATE_POINT_H

#include <math.h>
#include <stdio.h>

#include "_funtofemlapack.h"

/*!
  Given a set of points in R^3, locate the closest one to a given
  point in O(log(N)) time -- after an initial O(N) setup time.

  Copyright (c) 2010 Graeme Kennedy. All rights reserved.
  Not for commercial purposes.
*/

template <typename T>
class LocatePoint {
   public:
    LocatePoint(const T *_Xpts, int _npts, int _max_num_points) {
        Xpts = _Xpts;
        npts = _npts;
        max_num_points = _max_num_points;

        // Calculate approximately how many nodes there should be
        max_nodes = (2 * npts) / max_num_points;
        if (max_nodes < 1) {
            max_nodes = 1;
        }
        num_nodes = 0;

        // The point indicies
        indices = new int[npts];
        for (int i = 0; i < npts; i++) {
            indices[i] = i;
        }

        // Set up the data structure that represents the
        // splitting planes
        nodes = new int[2 * max_nodes];
        indices_ptr = new int[max_nodes];
        num_indices = new int[max_nodes];

        // The base point and normal direction for the splitting
        // planes
        node_xav = new T[3 * max_nodes];
        node_normal = new T[3 * max_nodes];

        for (int i = 0; i < max_nodes; i++) {
            nodes[2 * i] = nodes[2 * i + 1] = -1;
            indices_ptr[i] = -1;
            num_indices[i] = -1;

            for (int k = 0; k < 3; k++) {
                node_xav[3 * i + k] = 0.0;
                node_normal[3 * i + k] = 0.0;
            }
        }

        // Recursively split the points
        split(0, npts);
    }
    ~LocatePoint() {
        delete[] indices;
        delete[] nodes;
        delete[] indices_ptr;
        delete[] num_indices;
        delete[] node_xav;
        delete[] node_normal;
    }

    // Return the index of the point in the array
    // ------------------------------------------
    int locateClosest(const T xpt[]);
    int locateExhaustive(const T xpt[]);

    // Locate the K-closest points (note that dist/indices must of length K)
    // ---------------------------------------------------------------------
    void locateKClosest(int K, int indices[], T dist[], const T xpt[]) {
        int start = indices_ptr[root];
        int left_node = nodes[2 * root];
        int right_node = nodes[2 * root + 1];

        if (start != -1) {  // This node is a leaf
            // Do an exhaustive search of the points at the node

            int end = start + num_indices[root];
            for (int k = start; k < end; k++) {
                int n = indices[k];

                T t = ((Xpts[3 * n] - xpt[0]) * (Xpts[3 * n] - xpt[0]) +
                       (Xpts[3 * n + 1] - xpt[1]) * (Xpts[3 * n + 1] - xpt[1]) +
                       (Xpts[3 * n + 2] - xpt[2]) * (Xpts[3 * n + 2] - xpt[2]));

                if ((*nk < K) || (F2FRealPart(t) < F2FRealPart(dist[K - 1]))) {
                    insertIndex(dist, indx, nk, t, n, K);
                }
            }
        } else {
            T *xav = &node_xav[3 * root];
            T *normal = &node_normal[3 * root];

            // The normal distance
            T ndist = ((xpt[0] - xav[0]) * normal[0] + (xpt[1] - xav[1]) * normal[1] +
                       (xpt[2] - xav[2]) * normal[2]);

            if (F2FRealPart(ndist) < 0.0) {  // The point lies to the 'left' of the plane
                locateKClosest(K, left_node, xpt, dist, indx, nk);

                // If the minimum distance to the plane is less than the minimum
                // distance then search the other branch too - there could be a
                // point on that branch that lies closer than *dist
                if (*nk < K || F2FRealPart(ndist * ndist) < F2FRealPart(dist[*nk - 1])) {
                    locateKClosest(K, right_node, xpt, dist, indx, nk);
                }
            } else {  // The point lies to the 'right' of the plane
                locateKClosest(K, right_node, xpt, dist, indx, nk);

                // If the minimum distance to the plane is less than the minimum
                // distance then search the other branch too - there could be a
                // point on that branch that lies closer than *dist
                if (*nk < K || F2FRealPart(ndist * ndist) < F2FRealPart(dist[*nk - 1])) {
                    locateKClosest(K, left_node, xpt, dist, indx, nk);
                }
            }
        }
    }
    void locateKExhaustive(int K, int indices[], T dist[], const T xpt[]);

    // // Find the point with the closest taxi-cab distance to the plane
    // // --------------------------------------------------------------
    // void locateClosestTaxi(int K, int indices[], T dist[], const T xpt[], const T n[]);

   private:
    // // The recursive versions of the above functions
    // void locateClosest(int root, const T xpt[], T *dist, int *index);
    // void locateKClosest(int K, int root, const T xpt[], T *dist, int *indices, int *nk);

    // // Insert the index into the sorted list of indices
    // void insertIndex(T *dist, int *indices, int *nk, T d, int dindex, int K);

    // // Sort the list of initial indices into the tree data structure
    int split(int start, int end) {
        int root = num_nodes;

        num_nodes++;
        if (num_nodes >= max_nodes) {
            extendArrays(num_nodes, 2 * (num_nodes + 1));
            max_nodes = 2 * (num_nodes + 1);
        }

        if (end - start <= max_num_points) {
            nodes[2 * root] = -1;
            nodes[2 * root + 1] = -1;

            for (int k = 0; k < 3; k++) {
                node_xav[3 * root + k] = 0.0;
                node_normal[3 * root + k] = 0.0;
            }

            indices_ptr[root] = start;
            num_indices[root] = end - start;

            return root;
        }

        indices_ptr[root] = -1;
        num_indices[root] = 0;

        int mid =
            splitList(&node_xav[3 * root], &node_normal[3 * root], &indices[start], end - start);

        if (mid == 0 || mid == end - start) {
            fprintf(stderr,
                    "LocatePoint: Error, splitting points did nothing \
-- problem with your nodes?\n");
            return root;
        }

        // Now, split the right and left hand sides of the list
        int left_node = split(start, start + mid);
        int right_node = split(start + mid, end);

        nodes[2 * root] = left_node;
        nodes[2 * root + 1] = right_node;

        return root;
    }
    int splitList(T xav[], T normal[], int *indices, int npts) {
        xav[0] = xav[1] = xav[2] = T(0.0);
        normal[0] = normal[1] = normal[2] = T(0.0);

        // lwork  = 1 + 6*N + 2*N**2
        // liwork = 3 + 5*N
        double eigs[3];
        int N = 3;
        int lwork = 1 + 6 * N + 2 * N * N;
        double work[1 + 6 * 3 + 2 * 3 * 3];
        int liwork = 3 + 5 * N;
        int iwork[3 + 5 * 3];

        double I[9];
        for (int i = 0; i < 9; i++) {
            I[i] = 0.0;
        }

        // Find the average point and the moment of inertia about the average point
        for (int i = 0; i < np; i++) {
            int n = ind[i];
            for (int k = 0; k < 3; k++) {
                xav[k] += doublePart(Xpts[3 * n + k]);
            }

            // I[0] = Ix = y^2 + z^2
            I[0] +=
                doublePart(Xpts[3 * n + 1] * Xpts[3 * n + 1] + Xpts[3 * n + 2] * Xpts[3 * n + 2]);
            // I[4] = Iy = x^2 + z^2
            I[4] += doublePart(Xpts[3 * n] * Xpts[3 * n] + Xpts[3 * n + 2] * Xpts[3 * n + 2]);
            // I[8] = Iz = x^2 + y^2
            I[8] += doublePart(Xpts[3 * n] * Xpts[3 * n] + Xpts[3 * n + 1] * Xpts[3 * n + 1]);

            I[1] += -doublePart(Xpts[3 * n] * Xpts[3 * n + 1]);      // Ixy = - xy
            I[2] += -doublePart(Xpts[3 * n] * Xpts[3 * n + 2]);      // Ixz = - xz
            I[5] += -doublePart(Xpts[3 * n + 1] * Xpts[3 * n + 2]);  // Ixz = - yz
        }

        for (int k = 0; k < 3; k++) {
            xav[k] = xav[k] / (1.0 * np);
        }

        // Ix(cm) = Ix - np*(yav^2 + zav^2) ... etc
        I[0] = I[0] - np * doublePart(xav[1] * xav[1] + xav[2] * xav[2]);
        I[4] = I[4] - np * doublePart(xav[0] * xav[0] + xav[2] * xav[2]);
        I[8] = I[8] - np * doublePart(xav[0] * xav[0] + xav[1] * xav[1]);

        I[1] = I[1] + np * doublePart(xav[0] * xav[1]);
        I[2] = I[2] + np * doublePart(xav[0] * xav[2]);
        I[5] = I[5] + np * doublePart(xav[1] * xav[2]);

        I[3] = I[1];
        I[6] = I[2];
        I[7] = I[5];

        // Find the eigenvalues/eigenvectors
        int info;
        const char *jobz = "V";
        const char *uplo = "U";

        LAPACKsyevd(jobz, uplo, &N, I, &N, eigs, work, &lwork, iwork, &liwork, &info);

        normal[0] = I[0];
        normal[1] = I[1];
        normal[2] = I[2];

        int low = 0;
        int high = np - 1;

        // Now, split the index array such that
        while (high > low) {
            // (dot(Xpts[ind] - xav, n ) < 0 ) < 0.0 for i < low
            while (high > low && doublePart((Xpts[3 * ind[low]] - xav[0]) * normal[0] +
                                            (Xpts[3 * ind[low] + 1] - xav[1]) * normal[1] +
                                            (Xpts[3 * ind[low] + 2] - xav[2]) * normal[2]) < 0.0) {
                low++;
            }

            // (dot(Xpts[ind] - xav, n ) < 0 ) >= 0.0 for i >= high
            while (high > low &&
                   doublePart((Xpts[3 * ind[high]] - xav[0]) * normal[0] +
                              (Xpts[3 * ind[high] + 1] - xav[1]) * normal[1] +
                              (Xpts[3 * ind[high] + 2] - xav[2]) * normal[2]) >= 0.0) {
                high--;
            }

            if (high > low) {
                // Switch the two indices that don't match
                int temp = ind[high];
                ind[high] = ind[low];
                ind[low] = temp;
            }
        }

        if (low == 0 || low == np) {
            fprintf(stderr, "LocatePoint: Error split points\n");
        }

        return low;
    }

    // // Functions for array management
    void extendArrays(int old_len, int new_len) {
        nodes = newIntArray(nodes, 2 * old_len, 2 * new_len);
        indices_ptr = newIntArray(indices_ptr, old_len, new_len);
        num_indices = newIntArray(num_indices, old_len, new_len);

        node_xav = newDoubleArray(node_xav, 3 * old_len, 3 * new_len);
        node_normal = newDoubleArray(node_normal, 3 * old_len, 3 * new_len);
    }
    int *newIntArray(int *array, int old_len, int new_len) {
        int *temp = new int[new_len];

        for (int i = 0; i < old_len; i++) {
            temp[i] = array[i];
        }
        for (int i = old_len; i < new_len; i++) {
            temp[i] = -1;
        }

        delete[] array;

        return temp;
    }

    T *newDoubleArray(T *array, int old_len, int new_len) {
        T *temp = new T[new_len];

        for (int i = 0; i < old_len; i++) {
            temp[i] = array[i];
        }
        for (int i = old_len; i < new_len; i++) {
            temp[i] = 0.0;
        }
        delete[] array;

        return temp;
    }

    // The cloud of points to match
    const T *Xpts;
    int npts;

    int max_num_points;  // Maximum number of points stored at a leaf

    // Keep track of the nodes that have been created
    int max_nodes;
    int num_nodes;

    int *indices;      // Indices into the array of points
    int *nodes;        // Indices from the current node to the two child nodes
    int *indices_ptr;  // Pointer into the global indices array
    int *num_indices;  // Number of indices associated with this node
    T *node_xav;       // Origin point for the array
    T *node_normal;    // Normal direction of the plane
};

#endif
