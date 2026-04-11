#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // CRITICAL: Enables automatic std::vector <-> Python list conversion
#include <algorithm>
#include "kmeans.hpp"     // Assuming your C++ code is in this header
#include "spectral.hpp"
#include <omp.h>

namespace py = pybind11;

PYBIND11_MODULE(specbridge, m) {
    m.doc() = "KMeans and SpectralClustering module";

    m.def("get_max_threads", []() {
        return omp_get_max_threads();
    }, "Get the maximum number of OpenMP threads available");

    m.def("get_num_procs", []() {
        return omp_get_num_procs();
    }, "Get the number of processors available");

    m.def("set_num_threads", [](int num_threads) {
        omp_set_num_threads(num_threads);
    }, "Set the number of OpenMP threads to use");

    // 1. Bind the Result Struct
    py::class_<KMeansResult>(m, "KMeansResult")
        .def_property_readonly("centroids", [](const KMeansResult& r) {
            if (r.centroid_rows <= 0 || r.centroid_cols <= 0) {
                // Fallback for any legacy result values that may not carry shape metadata.
                py::array_t<float> out(r.centroids.size());
                auto out_buf = out.request();
                auto* out_ptr = static_cast<float*>(out_buf.ptr);
                std::copy(r.centroids.begin(), r.centroids.end(), out_ptr);
                return out;
            }
            py::array_t<float> out({r.centroid_rows, r.centroid_cols});
            auto out_buf = out.request();
            auto* out_ptr = static_cast<float*>(out_buf.ptr);
            std::copy(r.centroids.begin(), r.centroids.end(), out_ptr);
            return out;
        })
        .def_property_readonly("centroids_flat", [](const KMeansResult& r) {
            py::array_t<float> out(r.centroids.size());
            auto out_buf = out.request();
            auto* out_ptr = static_cast<float*>(out_buf.ptr);
            std::copy(r.centroids.begin(), r.centroids.end(), out_ptr);
            return out;
        })
        .def_readonly("labels", &KMeansResult::labels)
        .def_readonly("centroid_rows", &KMeansResult::centroid_rows)
        .def_readonly("centroid_cols", &KMeansResult::centroid_cols);

    py::class_<SpectralResult>(m, "SpectralResult")
        .def_property_readonly("labels", [](const SpectralResult& r) {
            py::array_t<int> out(r.labels.size());
            auto out_buf = out.request();
            auto* out_ptr = static_cast<int*>(out_buf.ptr);
            std::copy(r.labels.begin(), r.labels.end(), out_ptr);
            return out;
        })
        .def_property_readonly("eigvals", [](const SpectralResult& r) {
            py::array_t<float> out(r.eigvals.size());
            auto out_buf = out.request();
            auto* out_ptr = static_cast<float*>(out_buf.ptr);
            std::copy(r.eigvals.begin(), r.eigvals.end(), out_ptr);
            return out;
        })
        .def_readonly("ngap", &SpectralResult::ngap);

    py::class_<SBResult>(m, "SBResult")
        .def_readonly("clusterPointIndices", &SBResult::clusterPointIndices)
        .def_property_readonly("labels", [](const SBResult& r) {
            py::array_t<int> out(r.labels.size());
            auto out_buf = out.request();
            auto* out_ptr = static_cast<int*>(out_buf.ptr);
            std::copy(r.labels.begin(), r.labels.end(), out_ptr);
            return out;
        })
        .def_property_readonly("eigvals", [](const SBResult& r) {
            py::array_t<float> out(r.eigvals.size());
            auto out_buf = out.request();
            auto* out_ptr = static_cast<float*>(out_buf.ptr);
            std::copy(r.eigvals.begin(), r.eigvals.end(), out_ptr);
            return out;
        })
        .def_readonly("ngap", &SBResult::ngap);

    // 2. Bind the KMeans Class
    py::class_<KMeans>(m, "KMeans")
        // Bind the constructor, mirroring the default arguments from C++
        .def(py::init<int, int, int, uint64_t>(),
             py::arg("n_clusters"),
             py::arg("n_iter") = 20,
             py::arg("n_local_trials") = -1,
             py::arg("random_state") = 42)
        // Bind public member variables (allows Python to read AND write them like properties)
        .def_readwrite("n_clusters", &KMeans::n_clusters)
        .def_readwrite("n_iter", &KMeans::n_iter)
        .def_readwrite("n_local_trials", &KMeans::n_local_trials)
        .def_readwrite("random_state", &KMeans::random_state)

        // NumPy-friendly fit: infer n and d from a 2D float array.
        // NumPy-friendly fit: infer n and d from a 2D float array.
        .def("fit", [](KMeans& self,
                        py::array_t<float, py::array::c_style | py::array::forcecast> X_in) {
            
            // --- PHASE 1: Python Interaction (GIL is held) ---
            py::buffer_info buf = X_in.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Input must be a 2D NumPy array");
            }

            const int n_points = static_cast<int>(buf.shape[0]);
            const int dimensions = static_cast<int>(buf.shape[1]);
            const auto* ptr = static_cast<const float*>(buf.ptr);

            // --- PHASE 2: C++ Execution (Release the GIL!) ---
            // From this line down, NO Python objects can be touched.
            // Other Python threads can now run freely.
            py::gil_scoped_release release;

            // 1. Perform the mandatory memory copy across the C++/Python boundary
            Matrix X(ptr, ptr + (n_points * dimensions));
            
            // 2. Call your OpenMP-accelerated C++ fit function
            return self.fit(X, n_points, dimensions);

        }, py::arg("X"))

        // Backward-compatible signature used in older notebooks: fit(X, n, d).
        .def("fit", [](KMeans& self,
                        py::array_t<float, py::array::c_style | py::array::forcecast> X_in,
                        int n, int d) {
            py::buffer_info buf = X_in.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Input must be a 2D NumPy array");
            }

            const int n_points = static_cast<int>(buf.shape[0]);
            const int dimensions = static_cast<int>(buf.shape[1]);
            if (n != n_points || d != dimensions) {
                throw std::runtime_error("Provided n/d do not match X.shape");
            }

            const auto* ptr = static_cast<const float*>(buf.ptr);
            return self.fit(Matrix(ptr, ptr + n_points * dimensions), n_points, dimensions);
        }, py::arg("X"), py::arg("n"), py::arg("d"))

        // Keep raw API available for std::vector-backed inputs/tests.
        .def("fit_raw", &KMeans::fit,
             py::arg("X"), py::arg("n"), py::arg("d"));

    py::class_<SpectralClustering>(m, "SpectralClustering")
        .def(py::init<int, int, uint64_t>(),
             py::arg("n_clusters"),
             py::arg("n_iter") = 20,
             py::arg("random_state") = 42)
        .def_readwrite("n_clusters", &SpectralClustering::n_clusters)
        .def_readwrite("n_iter", &SpectralClustering::n_iter)
        .def_readwrite("random_state", &SpectralClustering::random_state)
        .def("fit", [](const SpectralClustering& self,
                        py::array_t<float, py::array::c_style | py::array::forcecast> A_in) {
            py::buffer_info buf = A_in.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Input affinity must be a 2D NumPy array");
            }
            const int m = static_cast<int>(buf.shape[0]);
            const int cols = static_cast<int>(buf.shape[1]);
            if (m != cols) {
                throw std::runtime_error("Affinity matrix must be square");
            }

            const auto* ptr = static_cast<const float*>(buf.ptr);
            Matrix affinity(ptr, ptr + m * m);
            return self.fit(affinity, m);
        }, py::arg("affinity"))
        .def("fit", [](const SpectralClustering& self,
                        py::array_t<float, py::array::c_style | py::array::forcecast> A_in,
                        int m) {
            py::buffer_info buf = A_in.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Input affinity must be a 2D NumPy array");
            }
            const int rows = static_cast<int>(buf.shape[0]);
            const int cols = static_cast<int>(buf.shape[1]);
            if (rows != cols) {
                throw std::runtime_error("Affinity matrix must be square");
            }
            if (rows != m) {
                throw std::runtime_error("Provided m does not match affinity shape");
            }

            const auto* ptr = static_cast<const float*>(buf.ptr);
            Matrix affinity(ptr, ptr + m * m);
            return self.fit(affinity, m);
        }, py::arg("affinity"), py::arg("m"))
        .def("fit_raw", &SpectralClustering::fit,
             py::arg("affinity"), py::arg("m"));

    m.def("spectral_bridges", [](py::array_t<float, py::array::c_style | py::array::forcecast> X_in,
                                  int k,
                                  int m,
                                  float p,
                                  float M,
                                  int n_iter,
                                  uint64_t random_state) {
        py::buffer_info buf = X_in.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Input X must be a 2D NumPy array");
        }

        const int n = static_cast<int>(buf.shape[0]);
        const int d = static_cast<int>(buf.shape[1]);
        const auto* ptr = static_cast<const float*>(buf.ptr);
        Matrix X(ptr, ptr + n * d);
        return spectralBridges(X, n, d, k, m, p, M, n_iter, random_state);
    },
    py::arg("X"),
    py::arg("k"),
    py::arg("m"),
    py::arg("p") = 0.5f,
    py::arg("M") = 1.0f,
    py::arg("n_iter") = 20,
    py::arg("random_state") = 42);
};