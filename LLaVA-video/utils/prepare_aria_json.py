from projectaria_tools.core import data_provider, calibration,mps
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
import json

AUDIOVOLUME=30 #30db

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
                 clip_length=int(5e9),
                 clip_out_dir="./data/clip"
                 ) -> None:
        print(f"Creating data provider from {file_name}")
        self.PROJECT_NAME=file_name.split("/")[-1].split(".vrs")[0]
        self.file_name=file_name
        self.save_path=save_path
        self.clip_length=clip_length
        self.candidate_clip={}
        ###group of path
        self.cam_rgb_group=[]
        self.eye_gaze_group=[]
        self.mic_group=[]
        self.imu_left_group=[]
        self.imu_right_group=[]
        ###
        try:
            self.provider = data_provider.create_vrs_data_provider(file_name)
            self.frame_num=self.provider.get_num_data(StreamId("214-1"))
            self.stream_dict=self.get_all_possible_stream()
            self.stream_mappings={}
            for id,stream in self.stream_dict.items():
                self.stream_mappings[stream]=StreamId(id)
            self.start_time,self.end_time=self.gettime("214-1") # Use camera to adjust
            self.clip_range=self.generate_clip_range(self.start_time,self.end_time,int(self.clip_length))
            self.candidate_clip=self.get_all_candidate_clip(clip_range=self.clip_range)
        except Exception as e:
            print(e)

    def gettime(self,stream_id):
        rgb_stream_id = StreamId(stream_id)
        time_domain = TimeDomain.DEVICE_TIME  
        start_time = self.provider.get_first_time_ns(rgb_stream_id, time_domain)
        end_time = self.provider.get_last_time_ns(rgb_stream_id, time_domain)
        return start_time,end_time    
    
    def get_all_possible_stream(self)->dict:
        stream_dict={}
        streams = self.provider.get_all_streams()
        for stream_id in streams:
            label = self.provider.get_label_from_stream_id(stream_id)
            if "rgb" in label or "mic" in label or "imu" in label:
                stream_dict[str(stream_id)]=label
        return stream_dict

    def generate_clip_range(self,start_time, end_time, clip_length):
        intervals = []
        current_time = start_time
        clip_range={}
        while current_time < end_time:
            next_time = min(current_time + clip_length, end_time)
            intervals.append([current_time, next_time])
            current_time = next_time
        for i,time in enumerate(intervals):
            clip_range[f'clip{i}']=time
        return clip_range

    def get_all_candidate_clip(self,clip_range):
        candidate_clip={}
        def get_candidate_clip(label):
            #{'214-1': 'camera-rgb', '231-1': 'mic', '1202-1': 'imu-right', '1202-2': 'imu-left'}
            stream_id=self.provider.get_stream_id_from_label(label)
            
            stamps=self.provider.get_timestamps_ns(stream_id,TimeDomain.DEVICE_TIME)
            for stamp in stamps:
                for clip_id,[sub_start,sub_end] in clip_range.items():
                    if clip_id not in candidate_clip.keys():
                        candidate_clip[clip_id]={'camera-rgb':[],'mic':[],'imu-right':[],'imu-left':[]}
                    if sub_start<=stamp<sub_end:
                        candidate_clip[clip_id][label].append(stamp)
                        break
        get_candidate_clip("camera-rgb")
        get_candidate_clip("mic")
        get_candidate_clip("imu-right")
        get_candidate_clip("imu-left")
        return candidate_clip

    def _undistort(self,image_array,sensor_name):
        device_calib = self.provider.get_device_calibration()
        src_calib = device_calib.get_camera_calib(sensor_name)

        dst_calib = calibration.get_linear_camera_calibration(1408, 1408, 465, sensor_name)
        # undistort image
        rectified_array = calibration.distort_by_calibration(image_array, dst_calib, src_calib, InterpolationMethod.BILINEAR)
        return rectified_array

    def write_modalities(self,modalities):
        # {'camera-rgb': 214-1, 'mic': 231-1, 'imu-right': 1202-1, 'imu-left': 1202-2}
        PROJECT_PATH=os.path.join(self.save_path,self.PROJECT_NAME)
        if not os.path.exists(PROJECT_PATH):
            os.makedirs(PROJECT_PATH)

        def _get_eye_gaze(modalities,time_stamps,video_width):
            generalized_eye_gaze_path = os.path.join(self.save_path, "eye_gaze", "general_eye_gaze.csv")
            generalized_eye_gazes = mps.read_eyegaze(generalized_eye_gaze_path)
            device_calibration = self.provider.get_device_calibration()
            cam_calibration = device_calibration.get_camera_calib(modalities)
            generalized_eye_gaze = get_nearest_eye_gaze(generalized_eye_gazes, time_stamps)
            if not generalized_eye_gaze:
                return None,None
            generalized_gaze_center_in_pixels = get_gaze_vector_reprojection(generalized_eye_gaze,\
                                                                              modalities, device_calibration, \
                                                                                cam_calibration, \
                                                                                    depth_m = generalized_eye_gaze.depth)
            ori_x, ori_y = generalized_gaze_center_in_pixels
            eye_gaze_rotate90 = lambda original_x, original_y, image_width: (image_width - original_y, original_x)
            rot_x,rot_y=eye_gaze_rotate90(ori_x,ori_y,video_width) #gaze rotate 90
            return rot_x,rot_y
        
        if modalities=="camera-rgb":
            MODALITY_PATH=os.path.join(PROJECT_PATH,modalities)
            if not os.path.exists(MODALITY_PATH):
                os.makedirs(MODALITY_PATH)

            EYEGAZE_dict={}
            stream_id=self.provider.get_stream_id_from_label(modalities)
            print(f"Now Write {modalities}:{stream_id}")
            for clip_id,value in self.candidate_clip.items():
                if clip_id not in EYEGAZE_dict.keys():
                    EYEGAZE_dict[clip_id]={}
                video_path = f"{self.PROJECT_NAME}_{modalities}_{clip_id}.mp4"
                output_file=os.path.join(MODALITY_PATH,video_path)
                fps = int(len(value[modalities]) /  (self.clip_length//1e9))
                sample_image, _ = self.provider.get_image_data_by_index(stream_id, 1)
                video_width, video_height,_ = sample_image.to_numpy_array().shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_out = cv2.VideoWriter(output_file, fourcc, fps, (video_width, video_height))

                self.cam_rgb_group.append(output_file)

                print(f"Start Write {modalities}:{clip_id}")
                for time_stamp in tqdm(value[modalities]):
                    EYEGAZE_dict[clip_id][time_stamp]={"eyegaze_x":None,"eyegaze_y":None}
                    eyegaze_x,eyegaze_y=_get_eye_gaze(modalities,time_stamp,video_width)
                    EYEGAZE_dict[clip_id][time_stamp]['eyegaze_x']=eyegaze_x
                    EYEGAZE_dict[clip_id][time_stamp]['eyegaze_y']=eyegaze_y
                    image_data,image_data_record=self.provider.get_image_data_by_time_ns(stream_id,time_stamp,TimeDomain.DEVICE_TIME)
                    try:
                        img=Image.fromarray(self._undistort(image_data.to_numpy_array(),modalities))
                        img=img.rotate(-90, expand=True)
                        frame_with_crosshair = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        video_out.write(frame_with_crosshair) 
                    except:
                        continue
                video_out.release()
            
            #Write EYEGAZE JSON
            MODALITY_PATH=os.path.join(PROJECT_PATH,"EyeGaze")
            if not os.path.exists(MODALITY_PATH):
                os.makedirs(MODALITY_PATH)
            EYEGAZE_JSON_PATH=f"{MODALITY_PATH}/{self.PROJECT_NAME}_EyeGaze.json"

            for i in range(len(self.candidate_clip.keys())):
                self.eye_gaze_group.append(EYEGAZE_JSON_PATH)

            with open(EYEGAZE_JSON_PATH,"w")as f:
                f.write(json.dumps(EYEGAZE_dict))
                

        if modalities=="mic":
            from pydub import AudioSegment #ffmpeg
            MODALITY_PATH=os.path.join(PROJECT_PATH,modalities)
            if not os.path.exists(MODALITY_PATH):
                os.makedirs(MODALITY_PATH)

            stream_id=self.provider.get_stream_id_from_label(modalities)

            print(f"Now Write {modalities}:{stream_id}")

            for clip_id,value in self.candidate_clip.items():
                audio_path = f"{self.PROJECT_NAME}_{modalities}_{clip_id}.wav"
                output_file=os.path.join(MODALITY_PATH,audio_path)
                
                self.mic_group.append(output_file)

                print(f"Start Write {modalities}:{clip_id}")

                clip_audio=[]
                for time_stamp in tqdm(value[modalities]):
                    audio_data,audio_data_record = self.provider.get_audio_data_by_time_ns(stream_id, time_stamp,TimeDomain.DEVICE_TIME)
                    try: #TODO(Choiszt) need to check this if raise expection
                        clip_audio.extend(a for a in audio_data.data)
                    except:
                        continue

                audio = [[] for c in range(0, 7)]
                for c in range(0, 7):
                    audio[c] += clip_audio[c::7] # list
                audio_array = np.array(audio).transpose()
                audio_segment = AudioSegment(audio_array.tobytes(),frame_rate = 96000, sample_width = 4, channels = 7)
                # audio_segment = audio_segment + AUDIOVOLUME
                audio_segment.export(output_file,format= "wav")

        if modalities=="imu-right":
            MODALITY_PATH=os.path.join(PROJECT_PATH,modalities)
            if not os.path.exists(MODALITY_PATH):
                os.makedirs(MODALITY_PATH)
            stream_id = self.provider.get_stream_id_from_label(modalities)
            print(f"Now Write {modalities}:{stream_id}")

            for clip_id,value in self.candidate_clip.items():
                imu_R_path = f"{self.PROJECT_NAME}_{modalities}_{clip_id}.npy"
                output_file=os.path.join(MODALITY_PATH,imu_R_path)
                print(f"Start Write {modalities}:{clip_id}")
                IMU_R_file=[]
                self.imu_right_group.append(output_file)

                for time_stamp in tqdm(value[modalities]):
                    imu_data = self.provider.get_imu_data_by_time_ns(stream_id,time_stamp ,TimeDomain.DEVICE_TIME)
                    accel_x=imu_data.accel_msec2[0]
                    accel_y=imu_data.accel_msec2[1]
                    accel_z=imu_data.accel_msec2[2]
                    gyro_x=imu_data.gyro_radsec[0]
                    gyro_y=imu_data.gyro_radsec[1]
                    gyro_z=imu_data.gyro_radsec[2]
                    IMU_R_file.append([time_stamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z])
                np.save(output_file,IMU_R_file)

        if modalities=="imu-left":
            MODALITY_PATH=os.path.join(PROJECT_PATH,modalities)
            if not os.path.exists(MODALITY_PATH):
                os.makedirs(MODALITY_PATH)
            stream_id = self.provider.get_stream_id_from_label(modalities)
            print(f"Now Write {modalities}:{stream_id}")

            for clip_id,value in self.candidate_clip.items():
                imu_L_path = f"{self.PROJECT_NAME}_{modalities}_{clip_id}.npy"
                output_file=os.path.join(MODALITY_PATH,imu_L_path)
                print(f"Start Write {modalities}:{clip_id}")
                self.imu_left_group.append(output_file)
                IMU_L_file=[]
                for time_stamp in tqdm(value[modalities]):
                    imu_data = self.provider.get_imu_data_by_time_ns(stream_id,time_stamp ,TimeDomain.DEVICE_TIME)
                    accel_x=imu_data.accel_msec2[0]
                    accel_y=imu_data.accel_msec2[1]
                    accel_z=imu_data.accel_msec2[2]
                    gyro_x=imu_data.gyro_radsec[0]
                    gyro_y=imu_data.gyro_radsec[1]
                    gyro_z=imu_data.gyro_radsec[2]
                    IMU_L_file.append([int(time_stamp),accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z])
                np.save(output_file,IMU_L_file)
                
    def get_image(self,frequence=1):

        frame_num = self.provider.get_num_data(StreamId("214-1"))
        print(f"Frame Number:{frame_num}")

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

    def _eye_gaze_rotate90(self,original_x,original_y,image_width):
        rotated_x =image_width -  original_y
        rotated_y = original_x
        return rotated_x,rotated_y
    
    def write_all(self):
        result={}
        #Write meta
        result['meta']={"device_id":None,
                        "username": None,
                        "date": None,
                        "start_time": self.start_time,
                        "end_time": self.end_time,
                        "clip_length": self.clip_length}
        #Write Video Clip
        result["video_clip"]={}
        assert len(self.cam_rgb_group)==len(self.eye_gaze_group)==len(self.mic_group)==len(self.imu_left_group)==len(self.imu_right_group)==len(self.candidate_clip.keys())
        for i,clip_name in enumerate(self.candidate_clip.keys()):
            result['video_clip'][clip_name]={}
            result['video_clip'][clip_name]["start_time"]=self.clip_range[clip_name][0]
            result['video_clip'][clip_name]["end_time"]=self.clip_range[clip_name][1]
            result['video_clip'][clip_name]["video"]=self.cam_rgb_group[i]
            result['video_clip'][clip_name]["audio"]=self.mic_group[i]
            result['video_clip'][clip_name]["imu_left"]=self.imu_left_group[i]
            result['video_clip'][clip_name]["imu_right"]=self.imu_right_group[i]
            result['video_clip'][clip_name]["gaze"]=self.eye_gaze_group[i]
            result['video_clip'][clip_name]["speech"]=None
            result['video_clip'][clip_name]["caption"]=None
        PROJECT_PATH=os.path.join(self.save_path,self.PROJECT_NAME)
        RESULT_PATH=os.path.join(PROJECT_PATH,f"{self.PROJECT_NAME}_result.json")
        with open(RESULT_PATH,"w")as f:
            f.write(json.dumps(result))
        
file_name="/mnt/data/liushuai/llava-video/data/adt/0322_2.vrs"
a=AriaDataset(file_name=file_name,selected_frequence=50)
a.write_modalities("mic")
a.write_modalities("camera-rgb") #rgb and eye gaze
a.write_modalities("imu-right")
a.write_modalities("imu-left")
a.write_all()

