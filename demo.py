import argparse
import time

import cv2
import mediapipe as mp

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--shape', '-s', type=int, nargs=2, default=[960, 540])

  args = parser.parse_args()
  return args

def main():
  # args
  args = get_args()

  mp_face_mesh = mp.solutions.face_mesh
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles

  cap_width, cap_height = args.shape

  # set webcam.
  cap = cv2.VideoCapture(0)

  cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

  print(f'width, height: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}, {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')

  # init model
  face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5
  )

  while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
      break

    start_time = time.time()
    # preprocess
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(frame)
    inference_time = time.time() - start_time

    # draw results
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
          image=frame,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp_drawing.draw_landmarks(
          image=frame,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        mp_drawing.draw_landmarks(
          image=frame,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    processing_time = time.time() - start_time

    text = 'Inference time: %.0fms' % (inference_time * 1000)
    frame = cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        thickness=2,
    )

    text = 'Processing time: %.0fms' % (processing_time * 1000)
    frame = cv2.putText(
        frame,
        text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        thickness=2,
    )
    text = 'FPS: %.1f' % (1 / processing_time)
    frame = cv2.putText(
        frame,
        text,
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        thickness=2,
    )

    cv2.imshow('MediaPipe Face Mesh', frame)
    if cv2.waitKey(1) == 27: # ESC
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()