import os
import cv2
import shutil
from math import cos, sin, pi

import subprocess
import tempfile

from progress.bar import IncrementalBar

FNULL = open(os.devnull, 'w')

def generate_video(angles,
                   images_path,
                   video_path,
                   parent_temp_dir):
    assert video_path.endswith('.mp4'), 'h264 pls'
    safe_makedirs(os.path.dirname(video_path))

    temp_dir = os.path.join(parent_temp_dir, 'temp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    safe_makedirs(temp_dir)

    filename_angles = []
    for filename,angle in zip(list(sorted(os.listdir(images_path))), angles):
        if filename.endswith('.jpg'):
            filename = filename[:-4]
        filename_angles.append((filename, angle))

    progress_bar = IncrementalBar(
        'Generating overlay',
        max=len(filename_angles),
        suffix='%(percent).1f%% - %(eta)ds')

    for filename, angle in filename_angles:
        img_path = os.path.join(images_path, filename + '.jpg')
        cv_image = overlay_angle(img_path, float(angle))
        cv2.imwrite(os.path.join(temp_dir, filename + '.png'), cv_image)
        progress_bar.next()

    print '\nGenerating mpg video'
    _, mpg_path = tempfile.mkstemp()
    print mpg_path
    subprocess.check_call([
        'mencoder',
        'mf://%s/*.png' % temp_dir,
        '-mf',
        'type=png:fps=20',
        '-o', mpg_path,
        '-speed', '1',
        '-ofps', '20',
        '-ovc', 'lavc',
        '-lavcopts', 'vcodec=mpeg2video:vbitrate=2500',
        '-oac', 'copy',
        '-of', 'mpeg'
    ], stdout=FNULL, stderr=subprocess.STDOUT)

    if os.path.exists(video_path):
        os.remove(video_path)

    print 'Converting mpg video to mp4'
    try:
        subprocess.check_call([
            'ffmpeg',
            '-i', mpg_path,
            video_path
        ], stdout=FNULL, stderr=subprocess.STDOUT)
    finally:
        os.remove(mpg_path)

    print 'Wrote final overlay video to', video_path

def point_on_circle(center, radius, angle):
    """ Finding the x,y coordinates on circle, based on given angle
    """
    # center of circle, angle in degree and radius of circle
    shift_angle = -3.14 / 2
    x = center[0] + (radius * cos(shift_angle + angle))
    y = center[1] + (radius * sin(shift_angle + angle))

    return int(x), int(y)

def get_degrees(radians):
    return (radians * 180.0) / 3.14

def overlay_angle(img_path, angle):
    center=(320, 400)
    radius=50
    cv_image = cv2.imread(img_path)
    cv2.circle(cv_image, center, radius, (255, 255, 255), thickness=4, lineType=8)
    x, y = point_on_circle(center, radius, -angle)
    cv2.circle(cv_image, (x,y), 6, (255, 0, 0), thickness=6, lineType=8)
    cv2.putText(
        cv_image,
        'angle: %.5f' % get_degrees(angle),
        (50, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255))

    return cv_image

def safe_makedirs(path):
    try: os.makedirs(path)
    except: pass

if __name__=='__main__':
    angles = [((x - 500.0) / 500) for x in range(1000)]
    images_path = '/data/extracted-jan27/center'
    video_path = '/data/extracted-jan27/movies/center.mp4'
    temp_dir = '/data/extracted-jan27/center_temp_dir'
    generate_video(angles, images_path, video_path, temp_dir)

