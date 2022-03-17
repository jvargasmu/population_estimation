#cython: boundscheck=False, wraparound=False, nonecheck=False
import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.math cimport log
from libc.math cimport exp
from libc.math cimport copysign
from libc.math cimport fabs

def count_matches(np.ndarray[np.uint32_t, ndim=2] raster1, np.ndarray[np.uint32_t, ndim=2] raster2,
	np.ndarray[np.uint32_t, ndim=2] valid_mask, int num_labels_raster1, int num_labels_raster2):

	if np.isfortran(raster1):
		raise ValueError("The input image is not C-contiguous")

	cdef int i, j
	cdef int h = raster1.shape[0]
	cdef int w = raster1.shape[1]

	cdef np.ndarray[np.uint32_t, ndim=2] matches = np.zeros((num_labels_raster1, num_labels_raster2), dtype=np.uint32)
	for i in range(h):
		for j in range(w):
			if valid_mask[i, j] == 1:
				matches[raster1[i, j], raster2[i, j]] =  matches[raster1[i, j], raster2[i, j]] + 1

	print("count matches done")
	return matches


def compute_area_of_regions(np.ndarray[np.uint32_t, ndim=2] regions, np.ndarray[np.uint32_t, ndim=2] valid_mask, int num_labels):

	if np.isfortran(regions):
		raise ValueError("The input image is not C-contiguous")

	cdef int h = regions.shape[0]
	cdef int w = regions.shape[1]

	cdef np.ndarray[np.uint32_t, ndim=1] areas = np.zeros(num_labels, dtype=np.uint32)

	for i in range(h):
		for j in range(w):
			if valid_mask[i, j] == 1:
				areas[regions[i, j]] = areas[regions[i, j]] + 1

	return areas


def compute_map_with_new_labels(np.ndarray[np.uint32_t, ndim=2] regions, np.ndarray[np.uint32_t, ndim=1] conversion, np.ndarray[np.uint32_t, ndim=2] valid_mask):

	if np.isfortran(regions):
		raise ValueError("The input image is not C-contiguous")

	cdef int h = regions.shape[0]
	cdef int w = regions.shape[1]

	cdef np.ndarray[np.uint32_t, ndim=2] relabel = np.zeros((h, w), dtype=np.uint32)

	for i in range(h):
		for j in range(w):
			if valid_mask[i, j] == 1:
				relabel[i, j] = conversion[regions[i, j]]

	return relabel


def compute_accumulated_values_by_region(np.ndarray[np.uint32_t, ndim=2] regions, np.ndarray[np.float32_t, ndim=2] input_data,
	np.ndarray[np.uint32_t, ndim=2] valid_mask, int num_labels):

	if np.isfortran(regions):
		raise ValueError("The input image is not C-contiguous")

	cdef int h = regions.shape[0]
	cdef int w = regions.shape[1]

	cdef np.ndarray[np.float32_t, ndim=1] features = np.zeros(num_labels, dtype=np.float32)

	for i in range(h):
		for j in range(w):
			if valid_mask[i, j] == 1:
				features[regions[i, j]] = features[regions[i, j]] + input_data[i, j]

	return features


def compute_disagg_weights(np.ndarray[np.uint32_t, ndim=2] regions, np.ndarray[np.float32_t, ndim=2] pred_map,
	np.ndarray[np.float32_t, ndim=1] pred_map_per_region, np.ndarray[np.uint32_t, ndim=2] valid_mask):

	if np.isfortran(regions):
		raise ValueError("The input image is not C-contiguous")

	cdef int h = regions.shape[0]
	cdef int w = regions.shape[1]

	cdef np.ndarray[np.float32_t, ndim=2] weights = np.zeros((h, w), dtype=np.float32)

	for i in range(h):
		for j in range(w):
			if valid_mask[i, j] == 1 and pred_map_per_region[regions[i, j]] > 0:
				weights[i, j] = pred_map[i, j] / pred_map_per_region[regions[i, j]]

	return weights


def set_value_for_each_region(np.ndarray[np.uint32_t, ndim=2] regions, np.ndarray[np.float32_t, ndim=1] value_per_region, np.ndarray[np.uint32_t, ndim=2] valid_mask):

	if np.isfortran(regions):
		raise ValueError("The input image is not C-contiguous")

	cdef int h = regions.shape[0]
	cdef int w = regions.shape[1]

	cdef np.ndarray[np.float32_t, ndim=2] output = np.zeros((h, w), dtype=np.float32)

	for i in range(h):
		for j in range(w):
			if valid_mask[i, j] == 1:
				output[i, j] = value_per_region[regions[i, j]]

	return output


def compute_map_with_new_labels(np.ndarray[np.uint32_t, ndim=2] regions, np.ndarray[np.uint32_t, ndim=1] conversion, np.ndarray[np.uint32_t, ndim=2] valid_mask):

	if np.isfortran(regions):
		raise ValueError("The input image is not C-contiguous")

	cdef int h = regions.shape[0]
	cdef int w = regions.shape[1]

	cdef np.ndarray[np.uint32_t, ndim=2] relabel = np.zeros((h, w), dtype=np.uint32)

	for i in range(h):
		for j in range(w):
			if valid_mask[i, j] == 1:
				relabel[i, j] = conversion[regions[i, j]]

	return relabel


def bool_arr_to_seq_of_indices(np.ndarray[np.uint32_t, ndim=1] bool_arr):
	cdef int i, size, current_index
	size = bool_arr.shape[0]
	cdef np.ndarray[np.uint32_t, ndim=1] seq_of_indices = np.zeros(size, dtype=np.uint32)
	current_index = 0
	for i in xrange(size):
		if bool_arr[i] == 1:
			seq_of_indices[i] = current_index
			current_index += 1
	return seq_of_indices


def cy_fast_ICM_with_pop_target(np.ndarray[np.float32_t, ndim=1] initial_y, np.ndarray[np.uint32_t, ndim=2] neigh_ind,
		np.ndarray[np.uint32_t, ndim=1] valid_g_regions, np.ndarray[np.float32_t, ndim=1] g_target,
		int num_groups, float perc_change, int max_iter, float lambda_val):

	cdef int it, i, j, k, t, g, num_samples, num_neigh, num_changes
	cdef float energy_diff_neigh, energy_region_group, total_energy, new_val_i, val_i, val_j, avg_neigh_val

	print "Pre-processing..."
	num_neigh = neigh_ind.shape[1]
	num_samples = initial_y.shape[0]
	cdef np.ndarray[np.int_t, ndim=1] best_t = np.zeros(num_samples, dtype=np.int)
	cdef np.ndarray[np.float32_t, ndim=1] energy = np.zeros(num_samples, dtype=np.float32)
	cdef np.ndarray[np.float32_t, ndim=1] total_sum_per_group = np.zeros(num_groups, dtype=np.float32)
	cdef np.ndarray[np.float32_t, ndim=1] final_vals = np.zeros(num_samples, dtype=np.float32)

	for i in xrange(num_samples):
		g = valid_g_regions[i]
		total_sum_per_group[g] += initial_y[i]

	print "num_samples {}".format(num_samples)
	print "num_neigh {}".format(num_neigh)

	for i in xrange(num_samples):
		avg_neigh_val = 0
		energy_diff_neigh = 0
		g = valid_g_regions[i]
		energy_region_group = fabs(g_target[g] - total_sum_per_group[g]) / g_target[g]
		for k in xrange(num_neigh):
			j = neigh_ind[i, k]
			energy_diff_neigh += fabs(initial_y[i] - initial_y[j])
			avg_neigh_val += initial_y[j]
		avg_neigh_val /= num_neigh
		total_energy = energy_diff_neigh + lambda_val * energy_region_group
		energy[i] = total_energy

	#print "Energy values initialized"

	num_changes = 0
	for it in xrange(max_iter):
		print "ICM iter {}".format(it)
		for i in xrange(num_samples):
			for t in xrange(-10, 11, 1):
				new_val_i = initial_y[i]*(1+perc_change*copysign(1, t))**fabs(t)
				val_i = initial_y[i]*(1+perc_change*copysign(1, best_t[i]))**fabs(best_t[i])
				energy_diff_neigh = 0
				for k in xrange(num_neigh):
					j = neigh_ind[i, k]
					val_j = initial_y[j]*(1+perc_change*copysign(1, best_t[j]))**fabs(best_t[j])
					energy_diff_neigh += fabs(new_val_i - val_j)

				energy_region_group = fabs(g_target[g] - (total_sum_per_group[g] + new_val_i - val_i)) / g_target[g]
				total_energy = energy_diff_neigh + lambda_val * energy_region_group
				if total_energy < energy[i]:
					total_sum_per_group[g] += (new_val_i - val_i)
					best_t[i] = t
					num_changes += 1
					energy[i] = total_energy

	print "ICM iters DONE , changes {}".format(num_changes)
	for i in xrange(num_samples):
		final_vals[i] = initial_y[i]*(1+perc_change*copysign(1, best_t[i]))**fabs(best_t[i])

	return final_vals