// Copyright (c) 2025–present Siyan Dong - siyandong.3 [at] gmail.com
// CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/  
// Contributors: Siyan Dong and Zijun Wang

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "opt_fus.h"

namespace py = pybind11;

py::array_t<float> mat_to_numpy_float(const cv::Mat& mat) { 
    if (mat.empty()) {
        return py::array_t<float>();
    }
    std::vector<ssize_t> shape;
    if (mat.dims == 2) {
        shape = {mat.rows, mat.cols};
        if (mat.channels() > 1) {
            shape.push_back(mat.channels());
        }
    }
    cv::Mat mat_copy = mat.clone();
    return py::array_t<float>(
        shape,
        mat_copy.ptr<float>()
    );
}
py::array_t<uint8_t> mat_to_numpy_uint8(const cv::Mat& mat) {
    if (mat.empty()) {
        return py::array_t<uint8_t>();
    }
    std::vector<ssize_t> shape = {mat.rows, mat.cols};
    if (mat.channels() > 1) {
        shape.push_back(mat.channels());
    }
    cv::Mat mat_copy = mat.clone();   
    return py::array_t<uint8_t>(
        shape,
        mat_copy.ptr<uint8_t>()
    );
}
template<typename T>
cv::Mat numpy_to_mat(py::array_t<T> arr) {
    py::buffer_info buf = arr.request();
    int cv_type;
    if (std::is_same<T, float>::value) {
        cv_type = CV_32F;
    } else if (std::is_same<T, uint8_t>::value) {
        cv_type = CV_8U;
    } else {
        throw std::runtime_error("Unsupported numpy dtype");
    }
    if (buf.ndim == 2) {
        cv::Mat mat(buf.shape[0], buf.shape[1], cv_type, (void*)buf.ptr);
        return mat.clone();
    } else if (buf.ndim == 3) {
        int channels = buf.shape[2];
        if (std::is_same<T, uint8_t>::value && channels == 3) {
            cv_type = CV_8UC3;
        } else if (std::is_same<T, float>::value && channels == 3) {
            cv_type = CV_32FC3;
        }
        cv::Mat mat(buf.shape[0], buf.shape[1], cv_type, (void*)buf.ptr);
        return mat.clone();
    }
    throw std::runtime_error("Unsupported array dimensions");
}

PYBIND11_MODULE(optimization_fusion, m) { 
    m.doc() = "Python Bindings for Optimization and Fusion";

    // only bind properties that are used in python code
    py::class_<optfus::DataConfiguration>(m, "DataConfiguration")  
        .def(py::init<const std::string&>(), py::arg("config_file"), "Load data config from file")
        .def_readwrite("fx", &optfus::DataConfiguration::focal_x)
        .def_readwrite("fy", &optfus::DataConfiguration::focal_y)
        .def_readwrite("cx", &optfus::DataConfiguration::principal_x)
        .def_readwrite("cy", &optfus::DataConfiguration::principal_y)
        .def_readwrite("image_width", &optfus::DataConfiguration::image_width)
        .def_readwrite("image_height", &optfus::DataConfiguration::image_height)
        .def_readwrite("depth_ratio", &optfus::DataConfiguration::depth_ratio)
        .def_readwrite("data_root", &optfus::DataConfiguration::data_root)
        .def_readwrite("output_folder", &optfus::DataConfiguration::output_folder); 
    py::class_<optfus::ControllerConfiguration>(m, "ControllerConfiguration")
        .def(py::init<const std::string&>(), py::arg("config_file"), "Load controller config from file")
        .def_readwrite("save_trajectory", &optfus::ControllerConfiguration::save_trajectory)
        .def_readwrite("save_scene", &optfus::ControllerConfiguration::save_scene);

    // scene points properties
    py::class_<optfus::PointCloud>(m, "PointCloud")
        .def(py::init<>())
        .def_property("vertices",
            [](optfus::PointCloud& pc) { return mat_to_numpy_float(pc.vertices); },
            [](optfus::PointCloud& pc, py::array_t<float> arr) { 
                pc.vertices = numpy_to_mat<float>(arr); 
            })
        .def_property("normals",
            [](optfus::PointCloud& pc) { return mat_to_numpy_float(pc.normals); },
            [](optfus::PointCloud& pc, py::array_t<float> arr) { 
                pc.normals = numpy_to_mat<float>(arr); 
            })
        .def_property("color",
            [](optfus::PointCloud& pc) { return mat_to_numpy_uint8(pc.color); },
            [](optfus::PointCloud& pc, py::array_t<uint8_t> arr) { 
                pc.color = numpy_to_mat<uint8_t>(arr); 
            })
        .def_readwrite("num_points", &optfus::PointCloud::num_points);

    // optimization and fusion
    py::class_<optfus::OptFus>(m, "OptFus")
        .def(py::init<const optfus::DataConfiguration, const optfus::ControllerConfiguration>(),
             py::arg("data_config"), py::arg("controller_config"), "Initialize OptFus")
        
        .def("process_frame",
             [](optfus::OptFus& self,
                py::array_t<float> depth_map,
                py::array_t<uint8_t> color_map,
                py::object rel_pose) {
                    cv::Mat depth = numpy_to_mat<float>(depth_map);
                    cv::Mat color = numpy_to_mat<uint8_t>(color_map);
                    Eigen::Matrix4d* pose_ptr = nullptr;
                    Eigen::Matrix4d pose_mat;
                    if (!rel_pose.is_none()) {
                        pose_mat = rel_pose.cast<Eigen::Matrix4d>();
                        pose_ptr = &pose_mat;
                    }
                    return self.process_frame(depth, color, pose_ptr);
                },
             py::arg("depth_map"),
             py::arg("color_map"),
             py::arg("rel_pose") = py::none(),
             "Optimization and fusion")
        
        .def("save_poses",
             &optfus::OptFus::save_poses,
             py::arg("filename"),
             "Save camera poses to file")

        .def("extract_pointcloud",
             &optfus::OptFus::extract_pointcloud,
             "Extract point cloud from TSDF")

        .def("export_ply",
              &optfus::OptFus::export_ply,
              py::arg("filename"),
              py::arg("point_cloud"),
              "Save point cloud to file with .ply format");
}
