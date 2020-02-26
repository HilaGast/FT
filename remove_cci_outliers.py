from dipy.tracking.streamline import Streamlines, cluster_confidence


def remove_cci_outliers(streamlines):
    s = Streamlines(streamlines)
    cci = cluster_confidence(s,subsample=7)
    keep_streamlines = Streamlines()
    keep_streamlines_idx = list()
    for i, sl in enumerate(s):
        if cci[i] >= 1:
            keep_streamlines.append(sl)
            keep_streamlines_idx.append(i)

    return keep_streamlines,keep_streamlines_idx
