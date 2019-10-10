# Max limits, used to modify the file index to store that information as well
MAX_CROPS_PER_VIDEO = 1000
MAX_SPATIAL_CROPS_PER_CROP = 10


def label_id_to_parts(full_id):
    vid_id = full_id // MAX_CROPS_PER_VIDEO
    temporal_crop_id = ((full_id % MAX_CROPS_PER_VIDEO) //
                        MAX_SPATIAL_CROPS_PER_CROP)
    spatial_crop_id = ((full_id % MAX_CROPS_PER_VIDEO) %
                       MAX_SPATIAL_CROPS_PER_CROP)
    return vid_id, temporal_crop_id, spatial_crop_id


def parts_to_full_id(vid_id, temporal_crop_id, spatial_crop_id,
                     total_temporal_crops, total_spatial_crops):
    assert (total_temporal_crops * total_spatial_crops <
            MAX_CROPS_PER_VIDEO)
    assert (total_spatial_crops < MAX_SPATIAL_CROPS_PER_CROP)
    return (vid_id * MAX_CROPS_PER_VIDEO +
            temporal_crop_id * MAX_SPATIAL_CROPS_PER_CROP +
            spatial_crop_id)
