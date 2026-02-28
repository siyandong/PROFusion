#include<opt_fus.h>
#include<fstream>
#include<dirent.h>
#include<sys/stat.h>
#include<algorithm>
using cv::cuda::GpuMat;
using Matf61da = Eigen::Matrix<double, 6, 1, Eigen::DontAlign>;

namespace optfus {

    OptFus::OptFus(const DataConfiguration _data_config, const ControllerConfiguration _controller_config): 
        data_config(_data_config),
        controller_config(_controller_config),
        camera_parameters(_data_config.image_width, _data_config.image_height, 
            _data_config.focal_x, _data_config.focal_y, _data_config.principal_x, _data_config.principal_y), 
        volume(data_config.volume_size, data_config.voxel_scale),
        frame_data(_data_config.image_height,_data_config.image_width),
        particle_leve{10240,3072,1024},PST(particle_leve,"randomized_optimization/cam_pose_templates/"),search_data(particle_leve),
        current_pose{}, previous_pose{}, poses{}, frame_id{0}, initialize_search_size{},
        iter_tsdf{_controller_config.init_fitness}
    {
        current_pose.setIdentity();
        current_pose.block(0, 0, 3, 3) = data_config.initial_rotation;
        current_pose(0, 3) = data_config.init_pos.x;
        current_pose(1, 3) = data_config.init_pos.y;
        current_pose(2, 3) = data_config.init_pos.z;
    }

    bool OptFus::process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map, Eigen::Matrix4d* rel_pose){
        
        internal::surface_measurement(color_map, depth_map, frame_data, camera_parameters, data_config.depth_cutoff_distance);
        const bool enough_depth_value = frame_data.valid_pixels_count.ptr<int>(0)[0] >= controller_config.enough_depth_ratio * frame_data.image_size;

        bool tracking_success { true };

        if (frame_id > 0) {
            if (!enough_depth_value){
                tracking_success = internal::pose_estimation(volume, PST, search_data, current_pose, frame_data, 
                camera_parameters, controller_config, particle_leve, &iter_tsdf, &previous_frame_success, initialize_search_size, nullptr, controller_config.translation_initialize_scale, controller_config.rotation_initialize_scale, previous_pose);
            }            
            else if (controller_config.pose_initialization_method == "pro"){
                tracking_success = internal::pose_estimation(volume, PST, search_data, current_pose, frame_data, 
                camera_parameters, controller_config, particle_leve, &iter_tsdf, &previous_frame_success, initialize_search_size, rel_pose, controller_config.translation_initialize_scale, controller_config.rotation_initialize_scale, previous_pose);
            }
            else if (controller_config.pose_initialization_method == "mix"){                
                tracking_success = internal::pose_estimation_mix(volume, PST, search_data, current_pose, frame_data, 
                camera_parameters, controller_config, particle_leve, &iter_tsdf, &previous_frame_success, initialize_search_size, rel_pose, controller_config.translation_initialize_scale, controller_config.rotation_initialize_scale, previous_pose);
            }
        }
        
        if (!tracking_success){
            current_pose = current_pose * *rel_pose;
            poses.push_back(current_pose);
            return false;
        }

        poses.push_back(current_pose);

        if (frame_id == 0 || enough_depth_value){
            previous_pose = current_pose;
            internal::cuda::surface_reconstruction(frame_data.depth_map, frame_data.color_map,
                volume, camera_parameters, data_config.truncation_distance,
                current_pose.inverse());
        }

        ++frame_id;
        return true;
    }

    void OptFus::save_poses(const std::string& filename) const
    {
        Eigen::Matrix4d init_pose=poses[0];
        std::ofstream trajectory;
        trajectory.open(filename);
        int iter_count=0;
        for (auto pose : poses){
            Eigen::Matrix4d temp_pose=init_pose.inverse()*pose;
            Eigen::Matrix3d rotation_m=temp_pose.block(0,0,3,3);
            Eigen::Vector3d translation=temp_pose.block(0,3,3,1)/1000;
            Eigen::Quaterniond q(rotation_m);
            trajectory<<iter_count<<" "<<translation.x()<<" "<<translation.y()<<" "<<translation.z()<<\
            " "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<std::endl;
            iter_count++;
        }
        trajectory.close();
    }

    PointCloud OptFus::extract_pointcloud() const
    {
        PointCloud cloud_data = internal::cuda::extract_points(volume, data_config.pointcloud_buffer_size);
        return cloud_data;
    }

    void OptFus::export_ply(const std::string& filename, const PointCloud& point_cloud)
    {
        std::ofstream file_out { filename };
        if (!file_out.is_open())
            return;

        file_out << "ply" << std::endl;
        file_out << "format ascii 1.0" << std::endl;
        file_out << "element vertex " << point_cloud.num_points << std::endl;
        file_out << "property float x" << std::endl;
        file_out << "property float y" << std::endl;
        file_out << "property float z" << std::endl;
        file_out << "property float nx" << std::endl;
        file_out << "property float ny" << std::endl;
        file_out << "property float nz" << std::endl;
        file_out << "property uchar red" << std::endl;
        file_out << "property uchar green" << std::endl;
        file_out << "property uchar blue" << std::endl;
        file_out << "end_header" << std::endl;

        for (int i = 0; i < point_cloud.num_points; ++i) {
            float3 vertex = point_cloud.vertices.ptr<float3>(0)[i];
            float3 normal = point_cloud.normals.ptr<float3>(0)[i];
            uchar3 color = point_cloud.color.ptr<uchar3>(0)[i];
            file_out << vertex.x << " " << vertex.y << " " << vertex.z << " " << normal.x << " " << normal.y << " "
                     << normal.z << " ";
            file_out << static_cast<int>(color.x) << " " << static_cast<int>(color.y) << " "
                     << static_cast<int>(color.z) << std::endl;
        }
    }
}
