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
			if valid_mask[i, j] == 1:
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
