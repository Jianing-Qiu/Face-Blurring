import cv2
import argparse
import os
from deepface.detectors import FaceDetector 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Blurring')
    parser.add_argument('--root_path', type=str, default="")
    args = parser.parse_args()

    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
    detector_backend = backends[4]
    face_detector = FaceDetector.build_model(detector_backend)

    save_dir = args.root_path+'_face_blurred'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_files = os.listdir(args.root_path)

    for im_file in image_files:
        im = cv2.imread(os.path.join(args.root_path, im_file))
        face_region_pairs = FaceDetector.detect_faces(face_detector, detector_backend, im)
        for face, region in face_region_pairs:
            x, y, w, h = region
            print(x, y, w, h)

            # Extract the region of the image that contains the face
            face_image = im[y:y+h, x:x+w]

            # Blur the face image
            face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

            # Put the blurred face region back into the im image
            im[y:y+h, x:x+w] = face_image

            cv2.imwrite(os.path.join(save_dir, im_file), im)





