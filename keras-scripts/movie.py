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
                   true_angles=None):
    """
    Let N be length of `angles` (array of floats).
    Assumes first N images in images_path correspond to each angle.
    Creates or overwrites a video at video_path by joining N images.
    """
    if true_angles is not None:
        assert len(angles) == len(true_angles), "True angles length should match angles"
    assert video_path.endswith('.mp4'), 'h264 pls'
    safe_makedirs(os.path.dirname(video_path))

    temp_dir = tempfile.mkdtemp()
    try:
        filename_angles = []
        for f,angle in zip(list(sorted(os.listdir(images_path))), angles):
            if f.endswith('.jpg'):
                f= f[:-4]
            filename_angles.append((f, angle))

        progress_bar = IncrementalBar(
            'Generating overlay',
            max=len(filename_angles),
        suffix='%(percent).1f%% - %(eta)ds')

        for i, (filename, angle) in enumerate(filename_angles):
            img_path = os.path.join(images_path, filename + '.jpg')
            png_path = filename + '.png'

            cv_image = cv2.imread(img_path)

            if true_angles is not None:
                cv_image = overlay_angle(cv_image, float(true_angles[i]), (0, 255, 0))

            cv_image = overlay_angle(cv_image, float(angle))
            cv2.imwrite(os.path.join(temp_dir, png_path), cv_image)
            progress_bar.next()

        print '\nGenerating mpg video'
        _, mpg_path = tempfile.mkstemp()
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
    finally:
        shutil.rmtree(temp_dir)

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

def overlay_angle(cv_image, angle, color=(255, 0, 0), text=True):
    center=(320, 400)
    radius=50
    cv2.circle(cv_image, center, radius, (255, 255, 255), thickness=4, lineType=8)
    x, y = point_on_circle(center, radius, -angle)
    cv2.circle(cv_image, (x,y), 6, color, thickness=6, lineType=8)
    if text:
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
    generate_video(angles, images_path, video_path, [0.2] * 1000)

