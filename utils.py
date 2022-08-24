import time

class FPS:
  def __init__(self):
    self.fps = 'Now measuring.'
    self.fps_start = time.time()
    self.fps_counter = 0

  def __call__(self):
    return self.fps

  def count(self):
    end_time = time.time()
    if end_time - self.fps_start < 1.:
      self.fps_counter += 1
    else:
      self.fps = self.fps_counter
      self.fps_start = end_time
      self.fps_counter = 0