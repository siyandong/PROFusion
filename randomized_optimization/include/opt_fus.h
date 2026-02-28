#ifndef OPT_FUS_H
#define OPT_FUS_H

#include "data_types.h"
using Matf61da = Eigen::Matrix<double, 6, 1, Eigen::DontAlign>;

namespace optfus {

    class OptFus {
    public:
        OptFus(const DataConfiguration _data_config, const ControllerConfiguration _controller_config);
        ~OptFus() = default;
        
        bool process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map, Eigen::Matrix4d* rel_pose = nullptr);
        
        void save_poses(const std::string& filename) const;
        
        PointCloud extract_pointcloud() const;

        void export_ply(const std::string& filename, const PointCloud& point_cloud);

    private:
        
        const DataConfiguration data_config;
        const ControllerConfiguration controller_config;

        const CameraParameters camera_parameters;
        const std::vector<int> particle_leve; 

        internal::VolumeData volume;
        internal::QuaternionData PST;         
        internal::SearchData search_data;
        internal::FrameData frame_data;

        Eigen::Matrix4d current_pose;
        Eigen::Matrix4d previous_pose;
        std::vector<Eigen::Matrix4d> poses;        
        bool previous_frame_success=false;

        Matf61da initialize_search_size;    
        size_t frame_id;    

        float iter_tsdf;
    };

    namespace internal {
        void surface_measurement(const cv::Mat_<cv::Vec3b>& color_map,
                                 const cv::Mat_<float>& depth_map,
                                 FrameData& frame_data,
                                 const CameraParameters& camera_params,
                                 const float depth_cutoff);

        bool pose_estimation(const VolumeData& volume,
                             const QuaternionData& quaternions,
                             SearchData& search_data,
                             Eigen::Matrix4d& pose,
                             FrameData& frame_data,
                             const CameraParameters& cam_params,
                             const ControllerConfiguration& controller_config,
                             const std::vector<int> particle_level,
                             float * iter_tsdf,
                             bool * previous_frame_success,
                             Matf61da& initialize_search_size,
                             const Eigen::Matrix4d* rel_pose = nullptr,
                             const float translation_initialize_scale = 1,
                             const float rotation_initialize_scale = 1,                                                       
                             const Eigen::Matrix4d& previous_pose = Eigen::Matrix4d::Identity());

        bool pose_estimation_mix(const VolumeData& volume,
                                 const QuaternionData& quaternions,
                                 SearchData& search_data,
                                 Eigen::Matrix4d& pose,
                                 FrameData& frame_data,
                                 const CameraParameters& cam_params,
                                 const ControllerConfiguration& controller_config,
                                 const std::vector<int> particle_level,
                                 float * iter_tsdf,
                                 bool * previous_frame_success,
                                 Matf61da& initialize_search_size,
                                 const Eigen::Matrix4d* rel_pose = nullptr,
                                 const float translation_initialize_scale = 1,
                                 const float rotation_initialize_scale = 1,
                                 const Eigen::Matrix4d& previous_pose = Eigen::Matrix4d::Identity());

        namespace cuda {

            void surface_reconstruction(const cv::cuda::GpuMat& depth_image,
                                        const cv::cuda::GpuMat& color_image,
                                        VolumeData& volume,
                                        const CameraParameters& cam_params,
                                        const float truncation_distance,
                                        const Eigen::Matrix4d& model_view);

            void surface_prediction(const VolumeData& volume,
                                    cv::cuda::GpuMat& shading_buffer,
                                    const CameraParameters& cam_parameters,
                                    const float truncation_distance,
                                    const float3 init_pos,
                                    cv::Mat& shaded_img,
                                    const Eigen::Matrix4d& pose);

            PointCloud extract_points(const VolumeData& volume, const int buffer_size);
        }
    }
}

#endif 