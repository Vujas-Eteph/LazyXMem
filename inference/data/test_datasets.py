import os
from os import path
import json

from inference.data.video_reader import VideoReader
from icecream import ic

from eteph_tools.third_party.detection_video_reader import DetectionVideoReader # For BURST


class LongTestDataset:
    def __init__(self, data_root, size=-1):
        self.image_dir = path.join(data_root, 'JPEGImages')
        self.mask_dir = path.join(data_root, 'Annotations')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                to_save = [
                    name[:-4] for name in os.listdir(path.join(self.mask_dir, video))
                ],
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)

class DAVISTestDataset:
    def __init__(self, data_root, imset='2017/val.txt', size=-1):
        if size != 480:
            self.image_dir = path.join(data_root, 'JPEGImages', 'Full-Resolution')
            self.mask_dir = path.join(data_root, 'Annotations', 'Full-Resolution')
            if not path.exists(self.image_dir):
                print(f'{self.image_dir} not found. Look at other options.')
                self.image_dir = path.join(data_root, 'JPEGImages', '1080p')
                self.mask_dir = path.join(data_root, 'Annotations', '1080p')
            assert path.exists(self.image_dir), 'path not found'
        else:
            self.image_dir = path.join(data_root, 'JPEGImages', '480p')
            self.mask_dir = path.join(data_root, 'Annotations', '480p')
        self.size_dir = path.join(data_root, 'JPEGImages', '480p')
        self.size = size

        with open(path.join(data_root, 'ImageSets', imset)) as f:
            self.vid_list = sorted([line.strip() for line in f])

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                size_dir=path.join(self.size_dir, video),
            )

    def __len__(self):
        return len(self.vid_list)


class YouTubeVOSTestDataset:
    def __init__(self, data_root, split, size=480):
        self.image_dir = path.join(data_root, 'all_frames', split+'_all_frames', 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))
        self.req_frame_list = {}

        with open(path.join(data_root, split, 'meta.json')) as f:
            # read meta.json to know which frame is required for evaluation
            meta = json.load(f)['videos']

            for vid in self.vid_list:
                req_frames = []
                objects = meta[vid]['objects']
                for value in objects.values():
                    req_frames.extend(value['frames'])

                req_frames = list(set(req_frames))
                self.req_frame_list[vid] = req_frames

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                to_save=self.req_frame_list[video], 
                use_all_mask=True
            )

    def __len__(self):
        return len(self.vid_list)


class MOSETestDataset:
    def __init__(self, data_root, split='valid', size=-1):
        self.image_dir = path.join(data_root, split, 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations')
        self.size = size
        ic(self.image_dir)
        self.vid_list = os.listdir(self.image_dir)

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video,
                              path.join(self.image_dir, video),
                              path.join(self.mask_dir, video),
                              size=self.size,
                              size_dir=path.join(self.size_dir, video))

    def __len__(self):
        return len(self.vid_list)


class BURSTDetectionTestDataset:
    def __init__(self, image_dir, split='val', size=-1, *, start=None, count=None):
        self.image_dir = image_dir
        self.mask_dir = path.join(image_dir, split, 'Annotations')
        self.size = size

        gt_json_dir = os.path.join(image_dir,'annotations',split,'all_classes.json')

        # read the json file to get a list of videos and frames to save
        with open(gt_json_dir, 'r') as f:
            json_file = json.load(f)
            sequences = json_file['sequences']

        assert split == 'test' or split == 'val'

        # load a randomized ordering of BURST videos for a balanced load
        with open(f'./util/burst_{split}.txt', mode='r') as f:
            randomized_videos = list(f.read().splitlines())

        # subsample a list of videos for processing
        if start is not None and count is not None:
            randomized_videos = randomized_videos[start:start + count]
            print(f'Start: {start}, Count: {count}, End: {start+count}')

        self.vid_list = []
        self.frames_to_save = {}
        for sequence in sequences:
            dataset = sequence['dataset']
            seq_name = sequence['seq_name']
            video_name = path.join(dataset, seq_name)
            if video_name not in randomized_videos:
                continue
            self.vid_list.append(video_name)

            annotated_image_paths = sequence['annotated_image_paths']
            self.frames_to_save[video_name] = [p[:-4] for p in annotated_image_paths]
            assert path.exists(path.join(image_dir, video_name))
            assert path.exists(path.join(mask_dir, video_name))

        assert len(self.vid_list) == len(randomized_videos)
        # to use the random ordering
        self.vid_list = randomized_videos

        print(f'Actual total: {len(self.vid_list)}')

    def get_datasets(self):
        for video in self.vid_list:
            yield DetectionVideoReader(
                video,
                path.join(self.image_dir, video),
                path.join(self.mask_dir, video),
                to_save=self.frames_to_save[video],
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)
