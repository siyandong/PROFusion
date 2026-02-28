#ifndef OPTFUS_DATA_TYPES_H
#define OPTFUS_DATA_TYPES_H

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Eigen>
#include <time.h>
#include <sstream>
#include <iomanip>
#else
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Dense>
#endif

using cv::cuda::GpuMat;

namespace optfus {

    struct DataConfiguration {

        float focal_x {.0};
        float focal_y {.0};
        float principal_x {.0};
        float principal_y {.0};
        int image_width {0};
        int image_height {0};
        float depth_ratio {.0};
        std::string data_root { "data" };
        std::string output_folder { "tmp" };

        float voxel_scale { 20.f };
        float truncation_distance { 60.f };
        int3 volume_size { make_int3(712, 712, 712) };
        float3 init_pos { volume_size.x / 2 * voxel_scale, volume_size.x / 2 * voxel_scale ,  volume_size.x / 2 * voxel_scale };
        float depth_cutoff_distance { 8000.f };
        int pointcloud_buffer_size { 3 * 2000000 };
        Eigen::Matrix3d initial_rotation { Eigen::Matrix3d::Identity() };  // rotate the scene points in TSDF
        
    private:

        Eigen::Matrix3d rotation_matrix_x(double angle_deg) const {
            double angle_rad = angle_deg * M_PI / 180.0;
            double cos_a = std::cos(angle_rad);
            double sin_a = std::sin(angle_rad);
            Eigen::Matrix3d R;
            R << 1,     0,      0,
                 0, cos_a, -sin_a,
                 0, sin_a,  cos_a;
            return R;
        }
        Eigen::Matrix3d rotation_matrix_y(double angle_deg) const {
            double angle_rad = angle_deg * M_PI / 180.0;
            double cos_a = std::cos(angle_rad);
            double sin_a = std::sin(angle_rad);
            Eigen::Matrix3d R;
            R <<  cos_a, 0, sin_a,
                      0, 1,     0,
                 -sin_a, 0, cos_a;
            return R;
        }
        Eigen::Matrix3d rotation_matrix_z(double angle_deg) const {
            double angle_rad = angle_deg * M_PI / 180.0;
            double cos_a = std::cos(angle_rad);
            double sin_a = std::sin(angle_rad);
            Eigen::Matrix3d R;
            R << cos_a, -sin_a, 0,
                 sin_a,  cos_a, 0,
                    0,      0, 1;
            return R;
        }
        Eigen::Matrix3d calculate_rotation_matrix(double rx, double ry, double rz, const std::string& order) const {
            Eigen::Matrix3d Rx = rotation_matrix_x(rx);
            Eigen::Matrix3d Ry = rotation_matrix_y(ry);
            Eigen::Matrix3d Rz = rotation_matrix_z(rz);
            if (order == "xyz") {
                return Rz * Ry * Rx;
            } else if (order == "zyx") {
                return Rx * Ry * Rz;
            } else if (order == "xzy") {
                return Ry * Rz * Rx;
            } else if (order == "zxy") {
                return Ry * Rx * Rz;
            } else if (order == "yxz") {
                return Rz * Rx * Ry;
            } else if (order == "yzx") {
                return Rx * Rz * Ry;
            } else {
                std::cerr << "Format not supported '" << order << "', using identity rotation" << std::endl;
                return Eigen::Matrix3d::Identity();
            }
        }

    public:

        DataConfiguration(const std::string &config_file){

            cv::FileStorage dataSetting(config_file.c_str(),cv::FileStorage::READ);
            std::string temp_str;

            dataSetting["fx"]>>focal_x;
            dataSetting["fy"]>>focal_y;
            dataSetting["cx"]>>principal_x;
            dataSetting["cy"]>>principal_y;
            dataSetting["image_width"]>>image_width;
            dataSetting["image_height"]>>image_height;
            dataSetting["depth_ratio"]>>depth_ratio;
            dataSetting["data_root"]>>data_root;
            dataSetting["output_folder"]>>output_folder;

            voxel_scale=dataSetting["voxel_size"];
            truncation_distance=dataSetting["truncated_size"];
            int voxel_x=dataSetting["voxel_x"];
            int voxel_y=dataSetting["voxel_y"];
            int voxel_z=dataSetting["voxel_z"];
            volume_size=make_int3(voxel_x,voxel_y,voxel_z);

            float init_x=dataSetting["init_x"];
            float init_y=dataSetting["init_y"];
            float init_z=dataSetting["init_z"];

            float init_pos_x=volume_size.x / 2 * voxel_scale - init_x;
            float init_pos_y=volume_size.y / 2 * voxel_scale - init_y;
            float init_pos_z=volume_size.z / 2 * voxel_scale - init_z;
            init_pos=make_float3(init_pos_x,init_pos_y,init_pos_z);

            cv::FileNode depth_cutoff_distance_node = dataSetting["depth_cutoff_distance"];
            if (!depth_cutoff_distance_node.empty()) {
                depth_cutoff_distance = depth_cutoff_distance_node;
            }
            
            cv::FileNode rotation_node = dataSetting["rotation"];
            if (!rotation_node.empty()) {
                double rx = rotation_node["rx"];
                double ry = rotation_node["ry"];
                double rz = rotation_node["rz"];
                std::string order = rotation_node["order"];
                initial_rotation = calculate_rotation_matrix(rx, ry, rz, order);
            } 
            dataSetting.release();
        }
    };

    struct ControllerConfiguration {
        std::string pose_initialization_method { "mix" };
        float translation_initialize_scale { 1.0f };
        float rotation_initialize_scale { 1.0f };
        std::string PST_path { "~" };
        int max_iteration {20};
        float scaling_coefficient1 {0.12};
        float scaling_coefficient2 {0.12};
        float init_fitness {0.5};
        float momentum {0.9};
        bool scaling_inherit_directly {false};
        float enough_depth_ratio {0.5};

        bool save_trajectory {false};
        bool save_scene {false};
        
        ControllerConfiguration(const std::string &config_file){
            cv::FileStorage controllerSetting(config_file.c_str(),cv::FileStorage::READ);
            controllerSetting["pose_initialization_method"]>>pose_initialization_method;
            translation_initialize_scale=controllerSetting["translation_initialize_scale"];
            rotation_initialize_scale=controllerSetting["rotation_initialize_scale"];
            controllerSetting["PST_path"]>>PST_path;
            max_iteration=controllerSetting["max_iteration"];
            scaling_coefficient1=controllerSetting["scaling_coefficient1"];
            scaling_coefficient2=controllerSetting["scaling_coefficient2"];
            init_fitness=controllerSetting["init_fitness"];
            momentum=controllerSetting["momentum"];
            scaling_inherit_directly=bool(int(controllerSetting["scaling_inherit_directly"]));
            enough_depth_ratio=controllerSetting["enough_depth_ratio"];

            save_trajectory=bool(int(controllerSetting["save_trajectory"]));
            save_scene=bool(int(controllerSetting["save_scene"]));

            if (pose_initialization_method != "pro" && pose_initialization_method != "mix") {
                std::cerr << "Mode not supported '" << pose_initialization_method << "', using 'mix' mode" << std::endl;
                pose_initialization_method = "mix";
            }
        }
    };

    struct CameraParameters {
        int image_width {0};
        int image_height {0};
        float focal_x {.0};
        float focal_y {.0};
        float principal_x {.0};
        float principal_y {.0};
        CameraParameters(const int w, const int h, 
                         const float fx, const float fy, 
                         const float cx, const float cy)
        {   
            image_width = w; 
            image_height = h; 
            focal_x = fx; 
            focal_y = fy;
            principal_x = cx;
            principal_y = cy; 
        }
    };

    struct PointCloud {
        cv::Mat vertices;
        cv::Mat normals;
        cv::Mat color;
        int num_points;
    };

    namespace internal {
  
        struct FrameData {
            GpuMat depth_map;
            GpuMat color_map;
            GpuMat vertex_map;
            GpuMat normal_map;
            int image_size;
            cv::Mat valid_pixels_count;
            GpuMat gpu_valid_pixels_count;

            explicit FrameData(const int image_height,const int image_width) 
            { 
                image_size = image_height * image_width;
                depth_map = cv::cuda::createContinuous(image_height, image_width, CV_32FC1);
                color_map = cv::cuda::createContinuous(image_height, image_width, CV_8UC3);
                vertex_map = cv::cuda::createContinuous(image_height, image_width, CV_32FC3);
                normal_map = cv::cuda::createContinuous(image_height, image_width, CV_32FC3);
                valid_pixels_count = cv::Mat::zeros(1, 1, CV_32FC1);
                gpu_valid_pixels_count = cv::cuda::createContinuous(1, 1, CV_32FC1);
            }
        };

        struct VolumeData {
            GpuMat tsdf_volume; 
            GpuMat weight_volume; 
            GpuMat color_volume; 
            int3 volume_size;
            float voxel_scale;

            VolumeData(const int3 _volume_size, const float _voxel_scale) :
                    tsdf_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_16SC1)),
                    weight_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_16SC1)),
                    color_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_8UC3)),
                    volume_size(_volume_size), voxel_scale(_voxel_scale)
            {
                tsdf_volume.setTo(32767);
                weight_volume.setTo(0);
                color_volume.setTo(0);
            }
        };

        struct QuaternionData{
            std::vector<GpuMat> q;
            std::vector<cv::Mat> q_trans;
            int num=20;

            QuaternionData(std::vector<int> particle_level, std::string PST_path):
            q(60),q_trans(60)
            {   
                for (int i=0;i<num;i++){
                    q_trans[i]=cv::Mat(particle_level[0],6,CV_32FC1);
                    q[i]=cv::cuda::createContinuous(particle_level[0], 6, CV_32FC1);

                    q_trans[i]=cv::imread(PST_path+"pst_10240_"+std::to_string(i)+".tiff",cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
                    q_trans[i].ptr<float>(0)[0]=0;
                    q_trans[i].ptr<float>(0)[1]=0;
                    q_trans[i].ptr<float>(0)[2]=0;
                    q_trans[i].ptr<float>(0)[3]=0;
                    q_trans[i].ptr<float>(0)[4]=0;
                    q_trans[i].ptr<float>(0)[5]=0;
                    q[i].upload(q_trans[i]);
                }

                for (int i=num;i<num*2;i++){
                    q_trans[i]=cv::Mat(particle_level[1],6,CV_32FC1);
                    q[i]=cv::cuda::createContinuous(particle_level[1], 6, CV_32FC1);

                    q_trans[i]=cv::imread(PST_path+"pst_3072_"+std::to_string(i-20)+".tiff",cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
                    q_trans[i].ptr<float>(0)[0]=0;
                    q_trans[i].ptr<float>(0)[1]=0;
                    q_trans[i].ptr<float>(0)[2]=0;
                    q_trans[i].ptr<float>(0)[3]=0;
                    q_trans[i].ptr<float>(0)[4]=0;
                    q_trans[i].ptr<float>(0)[5]=0;
                    q[i].upload(q_trans[i]);
                }

                for (int i=num*2;i<num*3;i++){
                    q_trans[i]=cv::Mat(particle_level[2],6,CV_32FC1);
                    q[i]=cv::cuda::createContinuous(particle_level[2], 6, CV_32FC1);

                    q_trans[i]=cv::imread(PST_path+"pst_1024_"+std::to_string(i-40)+".tiff",cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
                    q_trans[i].ptr<float>(0)[0]=0;
                    q_trans[i].ptr<float>(0)[1]=0;
                    q_trans[i].ptr<float>(0)[2]=0;
                    q_trans[i].ptr<float>(0)[3]=0;
                    q_trans[i].ptr<float>(0)[4]=0;
                    q_trans[i].ptr<float>(0)[5]=0;
                    q[i].upload(q_trans[i]);
                }
            }
        };

        struct SearchData{
            std::vector<GpuMat> gpu_search_count;
            std::vector<cv::Mat> search_count;
            std::vector<GpuMat> gpu_search_value;
            std::vector<cv::Mat> search_value;
            std::vector<GpuMat> gpu_valid_pixels_count;
            std::vector<cv::Mat> valid_pixels_count;

            SearchData(std::vector<int> particle_level):
            gpu_search_count(3),search_count(3),gpu_search_value(3),search_value(3),gpu_valid_pixels_count(3),valid_pixels_count(3)
            {
                search_count[0]=cv::Mat::zeros(particle_level[0],1,CV_32FC1);
                search_count[1]=cv::Mat::zeros(particle_level[1],1,CV_32FC1);
                search_count[2]=cv::Mat::zeros(particle_level[2],1,CV_32FC1);

                gpu_search_count[0]=cv::cuda::createContinuous(particle_level[0], 1, CV_32FC1);
                gpu_search_count[1]=cv::cuda::createContinuous(particle_level[1], 1, CV_32FC1);
                gpu_search_count[2]=cv::cuda::createContinuous(particle_level[2], 1, CV_32FC1);

                search_value[0]=cv::Mat::zeros(particle_level[0],1,CV_32FC1);
                search_value[1]=cv::Mat::zeros(particle_level[1],1,CV_32FC1);
                search_value[2]=cv::Mat::zeros(particle_level[2],1,CV_32FC1);

                gpu_search_value[0]=cv::cuda::createContinuous(particle_level[0], 1, CV_32FC1);
                gpu_search_value[1]=cv::cuda::createContinuous(particle_level[1], 1, CV_32FC1);
                gpu_search_value[2]=cv::cuda::createContinuous(particle_level[2], 1, CV_32FC1);

                valid_pixels_count[0]=cv::Mat::zeros(particle_level[0],1,CV_32FC1);
                valid_pixels_count[1]=cv::Mat::zeros(particle_level[1],1,CV_32FC1);
                valid_pixels_count[2]=cv::Mat::zeros(particle_level[2],1,CV_32FC1);

                gpu_valid_pixels_count[0]=cv::cuda::createContinuous(particle_level[0], 1, CV_32FC1);
                gpu_valid_pixels_count[1]=cv::cuda::createContinuous(particle_level[1], 1, CV_32FC1);
                gpu_valid_pixels_count[2]=cv::cuda::createContinuous(particle_level[2], 1, CV_32FC1);
            }
        };

        struct CloudData {
            GpuMat vertices;
            GpuMat normals;
            GpuMat color;

            cv::Mat host_vertices;
            cv::Mat host_normals;
            cv::Mat host_color;

            int* point_num;
            int host_point_num;

            explicit CloudData(const int max_number) :
                    vertices{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
                    normals{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
                    color{cv::cuda::createContinuous(1, max_number, CV_8UC3)},
                    host_vertices{}, host_normals{}, host_color{}, point_num{nullptr}, host_point_num{}
            {
                vertices.setTo(0.f);
                normals.setTo(0.f);
                color.setTo(0.f);

                cudaMalloc(&point_num, sizeof(int));
                cudaMemset(point_num, 0, sizeof(int));
            }

            CloudData(const CloudData&) = delete;
            CloudData& operator=(const CloudData& data) = delete;

            void download()
            {
                vertices.download(host_vertices);
                normals.download(host_normals);
                color.download(host_color);

                cudaMemcpy(&host_point_num, point_num, sizeof(int), cudaMemcpyDeviceToHost);
            }
        };
    }
}

#endif 