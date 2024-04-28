from projectaria_tools.core import data_provider, calibration,mps
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image, ImageDraw
from tqdm import tqdm

from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
)
class AriaDataset():

    def __init__(self,
                 file_name,
                 save_path="./data/aria_dataset",
                 selected_frequence=100,
                 clip_length=10000
                 ) -> None:
        print(f"Creating data provider from {file_name}")
        self.file_name=file_name
        self.save_path=save_path
        self.clip_length=clip_length
        try:
            self.provider = data_provider.create_vrs_data_provider(file_name)
            self.frame_num=self.provider.get_num_data(StreamId("214-1"))
            self.selected_frame_indices = [a for a in range(self.frame_num) if a % selected_frequence == 0]
            self.stream_dict=self.get_all_possible_stream()
            self.stream_mappings={}
            for id,stream in self.stream_dict.items():
                if "camera"in stream:
                    self.stream_mappings[stream]=StreamId(id)
        except Exception as e:
            print(e)

    def gettime(self): # Use camera frame for calibration
        rgb_stream_id = StreamId('231-1')
        time_domain = TimeDomain.DEVICE_TIME  
        start_time = self.provider.get_first_time_ns(rgb_stream_id, time_domain)
        end_time = self.provider.get_last_time_ns(rgb_stream_id, time_domain)
        print("start_time:{} nanoseconds".format(start_time))
        print("end time:{} nanoseconds".format(end_time))
        return start_time,end_time
    
    def get_all_possible_stream(self)->dict:
        # Goals:
        # In a vrs file, each sensor data is identitied through stream_id
        # Learn mapping between stream_id and label for each sensor

        # Key learnings:
        # VRS is using Unique Identifier for each stream called stream_id.
        # For each sensor data, it is attached with a stream_id, which contains two parts [RecordableTypeId, InstanceId].
        # To get the actual readable name of each sensor, we can use get_label_from_stream_id vise versa get_stream_id_from_label

        stream_dict={}
        streams = self.provider.get_all_streams()
        for stream_id in streams:
            label = self.provider.get_label_from_stream_id(stream_id)
            stream_dict[str(stream_id)]=label
        return stream_dict
    
    def _undistort(self,image_array,sensor_name):
        device_calib = self.provider.get_device_calibration()
        src_calib = device_calib.get_camera_calib(sensor_name)

        dst_calib = calibration.get_linear_camera_calibration(1408, 1408, 465, sensor_name)
        # undistort image
        rectified_array = calibration.distort_by_calibration(image_array, dst_calib, src_calib, InterpolationMethod.BILINEAR)
        return rectified_array

    def get_image(self,frequence=1):
        #frequence: The frequence of sampling.

        # stream_mappings = {
        #     "camera-slam-left": StreamId("1201-1"),
        #     "camera-slam-right":StreamId("1201-2"),
        #     "camera-rgb":StreamId("214-1"),
        #     "camera-eyetracking":StreamId("211-1"),
        # }
        # Query data with index
        frame_num = self.provider.get_num_data(StreamId("214-1"))
        for frame_index in self.selected_frame_indices:
            if frame_index % frequence == 0:
                for idx, [stream_name, stream_id] in enumerate(list(self.stream_mappings.items())):
                    
                    target_save_path=os.path.join(self.save_path,stream_name)
                    if not os.path.exists(target_save_path):
                        os.makedirs(target_save_path)
                    image_data,image_data_record = self.provider.get_image_data_by_index(stream_id, frame_index)
                    try:
                        if "et" not in stream_name:
                            undistort_image_data=self._undistort(image_data.to_numpy_array(),stream_name)
                        else:
                            undistort_image_data=image_data.to_numpy_array()
                        image_data=Image.fromarray(undistort_image_data)
                        image_path=os.path.join(target_save_path,f"{frame_index}.png")
                        image_data.save(image_path)
                    except:
                        continue

    def load_eye_gaze(self):

        generalized_eye_gaze_path = os.path.join(self.save_path, "eye_gaze", "general_eye_gaze.csv")
        generalized_eye_gazes = mps.read_eyegaze(generalized_eye_gaze_path)

        rgb_stream_id = StreamId("214-1")
        rgb_stream_label = self.provider.get_label_from_stream_id(rgb_stream_id)
        device_calibration = self.provider.get_device_calibration()
        cam_calibration = device_calibration.get_camera_calib(rgb_stream_label)
        

        for frame_index in tqdm(range(self.selected_frame_indices)):
            rgb_frame = self.provider.get_image_data_by_index(rgb_stream_id, frame_index)
            image = rgb_frame[0].to_numpy_array()
            
            capture_timestamp_ns = rgb_frame[1].capture_timestamp_ns
            generalized_eye_gaze = get_nearest_eye_gaze(generalized_eye_gazes, capture_timestamp_ns)
            if not generalized_eye_gaze:
                continue
            generalized_gaze_center_in_pixels = get_gaze_vector_reprojection(generalized_eye_gaze, rgb_stream_label, device_calibration, cam_calibration, depth_m = generalized_eye_gaze.depth)
            # Draw the crosshair
            img=Image.fromarray(self._undistort(image,self.stream_dict[str(rgb_stream_id)]))
            img=img.rotate(-90, expand=True)

            draw = ImageDraw.Draw(img)
            ori_x, ori_y = generalized_gaze_center_in_pixels
            x,y=self._eye_gaze_rotate90(ori_x,ori_y,img.size[0]) #gaze rotate 90


            frame_with_crosshair = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


    def visualize_load_eye_gaze(self):

        generalized_eye_gaze_path = os.path.join(self.save_path, "eye_gaze", "general_eye_gaze.csv")
        generalized_eye_gazes = mps.read_eyegaze(generalized_eye_gaze_path)

        rgb_stream_id = StreamId("214-1")
        rgb_stream_label = self.provider.get_label_from_stream_id(rgb_stream_id)
        device_calibration = self.provider.get_device_calibration()
        cam_calibration = device_calibration.get_camera_calib(rgb_stream_label)
        

        import cv2
        output_file = "sample.mp4"
        fps = 30  # Frames per second
        video_width, video_height = 1408, 1408 
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_out = cv2.VideoWriter(output_file, fourcc, fps, (video_width, video_height))

        for frame_index in tqdm(range(self.frame_num)):
            rgb_frame = self.provider.get_image_data_by_index(rgb_stream_id, frame_index)
            image = rgb_frame[0].to_numpy_array()
            
            capture_timestamp_ns = rgb_frame[1].capture_timestamp_ns
            generalized_eye_gaze = get_nearest_eye_gaze(generalized_eye_gazes, capture_timestamp_ns)
            if not generalized_eye_gaze:
                continue
            generalized_gaze_center_in_pixels = get_gaze_vector_reprojection(generalized_eye_gaze, rgb_stream_label, device_calibration, cam_calibration, depth_m = generalized_eye_gaze.depth)
            # Draw the crosshair
            img=Image.fromarray(self._undistort(image,self.stream_dict[str(rgb_stream_id)]))
            img=img.rotate(-90, expand=True)

            draw = ImageDraw.Draw(img)
            ori_x, ori_y = generalized_gaze_center_in_pixels
            x,y=self._eye_gaze_rotate90(ori_x,ori_y,img.size[0]) #gaze rotate 90

            draw.line([(x, y - 10), (x, y + 10)], fill="red", width=6)
            draw.line([(x - 10, y), (x + 10, y)], fill="red", width=6)

            frame_with_crosshair = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            video_out.write(frame_with_crosshair) 
                       
        video_out.release()
            
    def _eye_gaze_rotate90(self,original_x,original_y,image_width):
        rotated_x =image_width -  original_y
        rotated_y = original_x
        return rotated_x,rotated_y
    
    def load_mps(self):
        vrsfile =self.file_name

        # Trajectory and global points
        # closed_loop_trajectory = os.path.join(self.save_path, "trajectory","closed_loop_trajectory.csv")
        # global_points = os.path.join(self.save_path, "trajectory", "semidense_points.csv.gz")
        # Eye Gaze
        generalized_eye_gaze_path = os.path.join(self.save_path, "eye_gaze", "general_eye_gaze.csv")


        # Since we want to display the position of the RGB camera, we are querying its relative location
        # from the device and will apply it to the device trajectory.
        T_device_RGB = self.provider.get_device_calibration().get_transform_device_sensor("camera-rgb")

        # ## Load trajectory and global points
        # mps_trajectory = mps.read_closed_loop_trajectory(closed_loop_trajectory)
        # points = mps.read_global_point_cloud(global_points)

        ## Load eyegaze
        generalized_eye_gazes = mps.read_eyegaze(generalized_eye_gaze_path)





file_name="/mnt/data/liushuai/llava-video/data/adt/0322_2.vrs"
a=AriaDataset(file_name=file_name,selected_frequence=50)
a.load_eye_gaze()

