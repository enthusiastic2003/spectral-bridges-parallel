#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include "kmeans.hpp"
#include "kmeans_cuda.hpp"
#include "spectral.hpp"
#include <omp.h>

namespace py = pybind11;

PYBIND11_MODULE(specbridge, m) {
    m.doc() = "KMeans module";

    m.def("get_max_threads", []() {
        return omp_get_max_threads();
        }, "Get the maximum number of OpenMP threads available");

    m.def("get_num_procs", []() {
        return omp_get_num_procs();
        }, "Get the number of processors available");

    m.def("set_num_threads", [](int num_threads) {
        omp_set_num_threads(num_threads);
        }, "Set the number of OpenMP threads to use");

    py::class_<KMeansResult>(m, "KMeansResult")
        .def_property_readonly("centroids", [](const KMeansResult& r) {
        if (r.centroid_rows <= 0 || r.centroid_cols <= 0) {
            py::array_t<float> out(r.centroids.size());
            auto out_buf = out.request();
            std::copy(r.centroids.begin(), r.centroids.end(),
                static_cast<float*>(out_buf.ptr));
            return out;
        }
        py::array_t<float> out({ r.centroid_rows, r.centroid_cols });
        auto out_buf = out.request();
        std::copy(r.centroids.begin(), r.centroids.end(),
            static_cast<float*>(out_buf.ptr));
        return out;
            })
        .def_property_readonly("centroids_flat", [](const KMeansResult& r) {
        py::array_t<float> out(r.centroids.size());
        auto out_buf = out.request();
        std::copy(r.centroids.begin(), r.centroids.end(),
            static_cast<float*>(out_buf.ptr));
        return out;
            })
        .def_readonly("labels", &KMeansResult::labels)
        .def_readonly("centroid_rows", &KMeansResult::centroid_rows)
        .def_readonly("centroid_cols", &KMeansResult::centroid_cols);

    py::class_<SBResult>(m, "SBResult")
        .def_property_readonly("cluster_point_indices", [](const SBResult& r) {
            py::list result;
            for (const auto& cluster : r.clusterPointIndices) {
                py::array_t<int> arr(cluster.size());
                auto buf = arr.request();
                std::copy(cluster.begin(), cluster.end(),
                    static_cast<int*>(buf.ptr));
                result.append(arr);
            }
            return result;
        })
        .def_property_readonly("labels", [](const SBResult& r) {
            return py::cast(r.labels);
        })
        .def_property_readonly("eigvals", [](const SBResult& r) {
            return py::cast(r.eigvals);
        })
        .def_readonly("ngap", &SBResult::ngap);

        py::class_<KMeans>(m, "KMeans")
        .def(py::init<int, int, int, uint64_t>(),
            py::arg("n_clusters"),
            py::arg("n_iter") = 20,
            py::arg("n_local_trials") = -1,
            py::arg("random_state") = 42)
        .def_readwrite("n_clusters", &KMeans::n_clusters)
        .def_readwrite("n_iter", &KMeans::n_iter)
        .def_readwrite("n_local_trials", &KMeans::n_local_trials)
        .def_readwrite("random_state", &KMeans::random_state)
        .def("fit", [](KMeans& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> X_in) {
                py::buffer_info buf = X_in.request();
                if (buf.ndim != 2)
                    throw std::runtime_error("Input must be a 2D NumPy array");

                const int n = static_cast<int>(buf.shape[0]);
                const int d = static_cast<int>(buf.shape[1]);
                const auto* ptr = static_cast<const float*>(buf.ptr);
                Matrix X(ptr, ptr + n * d);

                py::gil_scoped_release release;
                return self.fit(X, n, d);
            }, py::arg("X"))
        .def("fit_raw", &KMeans::fit,
            py::arg("X"), py::arg("n"), py::arg("d"));

    // CUDA KMeans - free function binding
    m.def("fit_kmeans_cuda", [](
        py::array_t<float, py::array::c_style | py::array::forcecast> X_in,
        int n_clusters,
        int n_iter,
        uint64_t random_state) {
            py::buffer_info buf = X_in.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Input must be a 2D NumPy array");

            const int n = static_cast<int>(buf.shape[0]);
            const int d = static_cast<int>(buf.shape[1]);
            const auto* ptr = static_cast<const float*>(buf.ptr);
            Matrix X(ptr, ptr + n * d);

            py::gil_scoped_release release;
            return fitKMeansCuda(X, n, d, n_clusters, n_iter, random_state);
        },
        py::arg("X"),
        py::arg("n_clusters"),
        py::arg("n_iter") = 20,
        py::arg("random_state") = 42,
        "Run KMeans clustering on GPU via CUDA");

    // Also expose the raw version for callers that already have flat data
    m.def("fit_kmeans_cuda_raw", &fitKMeansCuda,
        py::arg("X"), py::arg("n"), py::arg("d"),
        py::arg("n_clusters"), py::arg("n_iter"), py::arg("random_state"),
        "Run KMeans clustering on GPU via CUDA (raw flat vector input)");

    py::class_<SpectralClustering>(m, "SpectralClustering")
        .def(py::init<int, int, int, float, uint64_t>(),
            py::arg("n_clusters"),
            py::arg("num_voronoi"),
            py::arg("n_iter") = 20,
            py::arg("target_perplexity") = 2.0f,
            py::arg("random_state") = 42)
        .def_readwrite("n_clusters", &SpectralClustering::n_clusters)
        .def_readwrite("n_iter", &SpectralClustering::n_iter)
        .def_readwrite("num_vornoi", &SpectralClustering::num_vornoi)
        .def_readwrite("random_state", &SpectralClustering::random_state)
        .def_readwrite("target_perplexity", &SpectralClustering::target_perplexity)
        .def("fit", [](SpectralClustering& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> X_in) {
                py::buffer_info buf = X_in.request();
                if (buf.ndim != 2)
                    throw std::runtime_error("Input must be a 2D NumPy array");

                const int n = static_cast<int>(buf.shape[0]);
                const int d = static_cast<int>(buf.shape[1]);
                const auto* ptr = static_cast<const float*>(buf.ptr);
                Matrix X(ptr, ptr + n * d);

                py::gil_scoped_release release;
                return self.fit(X, n, d);
            }, py::arg("X"));

}